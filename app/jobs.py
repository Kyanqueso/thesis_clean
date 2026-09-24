"""Single-worker subprocess job queue for pipeline runs launched from the dashboard."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from app import paths
from app.paths import write_run_dir
from app.runs import MODES, PIPELINES, ROOT, data_path

SRC_DIR = ROOT / "src"


def log_dir() -> Path:
    """Current write tree's log folder. A function, not a constant: app.paths
    re-resolves the tree while the server runs."""
    return paths.RUNS_WRITE_DIR / "_logs"


class ValidationError(Exception):
    pass


@dataclass
class Job:
    id: str
    stage: str
    params: dict
    cmds: list[list[str]] = field(default_factory=list)
    status: str = "queued"  # queued | running | done | failed | cancelled
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    returncode: int | None = None
    cancel_requested: bool = False
    # pinned when the job is created, so a tree that moves mid-session cannot
    # strand a running job's log somewhere the reader no longer looks
    logs_root: Path = field(default_factory=log_dir)

    @property
    def log_path(self) -> Path:
        return self.logs_root / f"{self.id}.log"

    def public(self) -> dict:
        return {
            "id": self.id, "stage": self.stage, "params": self.params,
            "status": self.status, "created": self.created,
            "started": self.started, "finished": self.finished,
            "returncode": self.returncode,
            "cmds": [" ".join(c) for c in self.cmds],
        }


def _limited_copy(src: Path, limit: int, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "r", encoding="utf-8") as fin, open(dest, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i >= limit:
                break
            fout.write(line)
    return dest


def build_cmds(stage: str, params: dict) -> list[list[str]]:
    py = sys.executable

    if stage == "dataset":
        which = params.get("which")
        if which == "alamsabi":
            return [[py, str(SRC_DIR / "download_dataset.py"), "--out-dir", str(paths.DATA_DIR)]]
        if which == "organic":
            return [[py, str(SRC_DIR / "organic_dataset.py"), "--out-dir", str(paths.DATA_DIR)]]
        raise ValidationError("dataset job needs which=alamsabi|organic")

    if stage == "sweep":
        cmd = [py, str(SRC_DIR / "run_all_pipelines.py"),
               "--data-dir", str(paths.DATA_DIR), "--runs-dir", str(paths.RUNS_WRITE_DIR)]
        if params.get("limit"):
            cmd += ["--limit", str(int(params["limit"]))]
        if params.get("models"):
            cmd += ["--models", str(params["models"])]
        if params.get("force"):
            cmd.append("--force")
        if params.get("save_models", True):
            cmd.append("--save-models")
        if params.get("skip_projections"):
            cmd.append("--skip-projections")
        return [cmd]

    if stage == "run":
        pipeline, mode = params.get("pipeline"), params.get("mode")
        if pipeline not in PIPELINES:
            raise ValidationError(f"unknown pipeline: {pipeline}")
        if mode not in MODES:
            raise ValidationError(f"unknown mode: {mode}")
        src = data_path(pipeline)
        if not src.is_file():
            raise ValidationError(f"dataset missing: {src.name} - build/download it first")

        # a --limit run is a smoke test: keep it out of the real run dir so it
        # cannot overwrite delivered results with a toy evaluation
        smoke = bool(params.get("limit"))
        run_data = src
        if smoke:
            run_data = _limited_copy(src, int(params["limit"]),
                                     paths.RUNS_WRITE_DIR / "_smoke_data" / f"{pipeline}.jsonl")

        d = write_run_dir(pipeline, mode, smoke=smoke)
        gen_cmd = [py, str(SRC_DIR / "generate_embeddings.py"),
                   "--data", str(run_data), "--mode", mode,
                   "--embed-dir", str(d / "embeddings")]
        if params.get("force"):
            gen_cmd.append("--force")
        if params.get("models"):
            gen_cmd += ["--models", str(params["models"])]

        eval_cmd = [py, str(SRC_DIR / "train_evaluate_visualize.py"),
                    "--data", str(run_data),
                    "--embed-dir", str(d / "embeddings"),
                    "--results-dir", str(d / "results"),
                    "--figures-dir", str(d / "figures")]
        if params.get("save_models", True):
            eval_cmd += ["--save-models", "--models-dir", str(d / "models")]
        if params.get("skip_projections"):
            eval_cmd.append("--skip-projections")
        return [gen_cmd, eval_cmd]

    raise ValidationError(f"unknown stage: {stage}")


class JobQueue:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.order: list[str] = []
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._current: tuple[Job, subprocess.Popen] | None = None
        threading.Thread(target=self._worker, daemon=True).start()

    def enqueue(self, stage: str, params: dict) -> Job:
        cmds = build_cmds(stage, params)
        job = Job(id=uuid.uuid4().hex[:12], stage=stage, params=params, cmds=cmds)
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        job.log_path.write_text("", encoding="utf-8")
        with self._lock:
            self.jobs[job.id] = job
            self.order.append(job.id)
        self._event.set()
        return job

    def list(self) -> list[dict]:
        with self._lock:
            return [self.jobs[j].public() for j in reversed(self.order)]

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None:
            return False
        if job.status == "queued":
            job.status = "cancelled"
            job.finished = time.time()
            return True
        if job.status == "running":
            job.cancel_requested = True
            cur = self._current
            if cur and cur[0] is job:
                proc = cur[1]
                if os.name == "nt":
                    # /T kills the whole tree (run_all_pipelines spawns children)
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                                   capture_output=True)
                else:
                    proc.kill()
            return True
        return False

    def _next(self) -> Job | None:
        with self._lock:
            for jid in self.order:
                if self.jobs[jid].status == "queued":
                    return self.jobs[jid]
        return None

    def _worker(self):
        while True:
            job = self._next()
            if job is None:
                self._event.wait(timeout=2.0)
                self._event.clear()
                continue
            job.status = "running"
            job.started = time.time()
            with open(job.log_path, "a", encoding="utf-8", errors="replace") as logf:
                ok = True
                for cmd in job.cmds:
                    if job.cancel_requested:
                        job.status, ok = "cancelled", False
                        break
                    logf.write("$ " + " ".join(cmd) + "\n")
                    logf.flush()
                    # the pipeline scripts print emoji; force UTF-8 stdio on Windows
                    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
                    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                            cwd=ROOT, env=env)
                    self._current = (job, proc)
                    rc = proc.wait()
                    self._current = None
                    if job.cancel_requested:
                        job.status, ok = "cancelled", False
                        break
                    if rc != 0:
                        job.status, job.returncode, ok = "failed", rc, False
                        logf.write(f"\n[job failed: exit {rc}]\n")
                        break
                if ok:
                    job.status = "done"
                    job.returncode = 0
            job.finished = time.time()


QUEUE = JobQueue()
