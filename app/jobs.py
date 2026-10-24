"""Single-worker subprocess job queue for pipeline runs launched from the dashboard."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from app import paths, settings, vast
from app.paths import write_run_dir
from app.runs import MODES, PIPELINES, ROOT, data_path
from pipeline_paths import SMOKE_SUBDIR

SRC_DIR = ROOT / "src"

# one of these in a job's command list is not a process: the worker stops the
# Vast.ai instance there, once the last step that needs the GPU is behind it
VAST_STOP = "@vast-stop"

# the lines src/progress.py prints
MARKER = re.compile(r"##(step|progress) (\{.*\})")


def log_dir() -> Path:
    """Current write tree's log folder. A function, not a constant: app.paths
    re-resolves the tree while the server runs."""
    return paths.RUNS_WRITE_DIR / "_logs"


class ValidationError(Exception):
    pass


@dataclass
class Cmd:
    argv: list[str]
    # announced by the worker as one step of the overall bar. Empty for
    # run_all_pipelines.py, which announces its own (`steps` of them).
    label: str = ""
    steps: int = 1


@dataclass
class Job:
    id: str
    stage: str
    params: dict
    cmds: list[Cmd] = field(default_factory=list)
    env: dict = field(default_factory=dict)
    status: str = "queued"  # queued | running | done | failed | cancelled
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    returncode: int | None = None
    cancel_requested: bool = False
    # pinned when the job is created, so a tree that moves mid-session cannot
    # strand a running job's log somewhere the reader no longer looks
    logs_root: Path = field(default_factory=log_dir)
    # progress, scraped from the ##step / ##progress lines in the log
    step: int = 0                 # steps STARTED so far
    step_label: str = ""
    phase: dict | None = None     # the newest ##progress of the current step
    vast_stopped: bool = False
    _off: int = 0
    _tail: str = ""
    _scan_lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def log_path(self) -> Path:
        return self.logs_root / f"{self.id}.log"

    @property
    def remote(self) -> bool:
        return bool(self.env.get("THESIS_QWEN3_REMOTE"))

    def scan(self) -> None:
        """Read whatever the log gained since the last call and fold its marker
        lines (src/progress.py) into step / phase."""
        with self._scan_lock:
            try:
                with open(self.log_path, "rb") as f:
                    f.seek(self._off)
                    chunk = f.read()
            except OSError:
                return
            if not chunk:
                return
            self._off += len(chunk)
            # tqdm redraws with \r, so a marker can follow one on the same "line"
            lines = re.split(r"[\r\n]", self._tail + chunk.decode("utf-8", "replace"))
            self._tail = lines.pop()
            for line in lines:
                # searched, not anchored: tqdm draws to stderr without a newline,
                # so a marker often lands on the end of a progress-bar line
                m = MARKER.search(line)
                if not m:
                    continue
                kind = m.group(1)
                try:
                    data = json.loads(m.group(2))
                except ValueError:
                    continue
                if kind == "step":
                    self.step += 1
                    self.step_label = str(data.get("label", ""))
                    self.phase = None
                else:
                    self.phase = data

    def public(self) -> dict:
        if self.status == "running":
            self.scan()
        return {
            "id": self.id, "stage": self.stage, "params": self.params,
            "status": self.status, "created": self.created,
            "started": self.started, "finished": self.finished,
            "returncode": self.returncode,
            "cmds": [" ".join(c.argv) for c in self.cmds],
            "progress": {"step": self.step, "steps": sum(c.steps for c in self.cmds),
                         "label": self.step_label, "phase": self.phase},
        }


def _limited_copy(src: Path, limit: int, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "r", encoding="utf-8") as fin, open(dest, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i >= limit:
                break
            fout.write(line)
    return dest


def _wants_remote(params: dict) -> bool:
    """qwen3 on the Vast.ai instance: asked for, and qwen3 is actually in the run."""
    models = params.get("models")
    return bool(params.get("qwen3_remote")) and (not models or "qwen3" in str(models).split(","))


def job_env(params: dict) -> dict:
    if not _wants_remote(params):
        return {}
    try:
        vast.target()
    except vast.VastError as e:
        raise ValidationError(str(e))
    return {"THESIS_QWEN3_REMOTE": "1"}


def _dataset_cmds(py: str, which: str) -> list[Cmd]:
    if which == "alamsabi":
        return [Cmd([py, "-u", str(SRC_DIR / "download_dataset.py"), "--out-dir", str(paths.DATA_DIR)],
                    "Download the Alamsabi dataset")]
    if which == "organic":
        return [Cmd([py, "-u", str(SRC_DIR / "organic_dataset.py"), "--out-dir", str(paths.DATA_DIR)],
                    "Build the organic datasets (collect, clean, de-identify)")]
    raise ValidationError("dataset job needs which=alamsabi|organic")


def _sweep_cmds(py: str, params: dict) -> list[Cmd]:
    n_runs = len(PIPELINES) * len(MODES)
    # a --limit sweep is a smoke test: keep it out of the real run dirs so it
    # cannot overwrite delivered results with a toy evaluation
    runs_dir = paths.RUNS_WRITE_DIR / SMOKE_SUBDIR if params.get("limit") else paths.RUNS_WRITE_DIR
    base = [py, "-u", str(SRC_DIR / "run_all_pipelines.py"),
            "--data-dir", str(paths.DATA_DIR), "--runs-dir", str(runs_dir)]
    if params.get("limit"):
        base += ["--limit", str(int(params["limit"]))]
    if params.get("models"):
        base += ["--models", str(params["models"])]
    if params.get("skip_done"):
        base.append("--skip-done")
    force = ["--force"] if params.get("force") else []
    train = []
    if params.get("save_models", True):
        train.append("--save-models")
    if params.get("skip_projections"):
        train.append("--skip-projections")

    if not _wants_remote(params):
        return [Cmd(base + force + train, steps=2 * n_runs + 1)]
    # With a rented GPU in play, do every embedding first and hand the instance
    # back before the hours of CPU-only training start.
    return [Cmd(base + force + ["--embed-only"], steps=n_runs),
            Cmd([VAST_STOP], steps=0),
            Cmd(base + train + ["--train-only"], steps=n_runs + 1)]


def build_cmds(stage: str, params: dict) -> list[Cmd]:
    py = sys.executable

    if stage == "dataset":
        return _dataset_cmds(py, params.get("which"))

    if stage == "sweep":
        return _sweep_cmds(py, params)

    if stage == "full":
        # data -> embeddings -> training -> aggregate. Datasets already on disk
        # are kept unless asked otherwise: the existing results were trained on
        # exactly those files.
        rebuild = bool(params.get("rebuild_data"))
        cmds = []
        if rebuild or not data_path("pipeline1_alamsabi").is_file():
            cmds += _dataset_cmds(py, "alamsabi")
        if rebuild or not all(data_path(p).is_file() for p in PIPELINES if p != "pipeline1_alamsabi"):
            cmds += _dataset_cmds(py, "organic")
        return cmds + _sweep_cmds(py, params)

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
        gen_cmd = [py, "-u", str(SRC_DIR / "generate_embeddings.py"),
                   "--data", str(run_data), "--mode", mode,
                   "--embed-dir", str(d / "embeddings")]
        if params.get("force"):
            gen_cmd.append("--force")
        if params.get("models"):
            gen_cmd += ["--models", str(params["models"])]

        eval_cmd = [py, "-u", str(SRC_DIR / "train_evaluate_visualize.py"),
                    "--data", str(run_data),
                    "--embed-dir", str(d / "embeddings"),
                    "--results-dir", str(d / "results"),
                    "--figures-dir", str(d / "figures")]
        if params.get("save_models", True):
            eval_cmd += ["--save-models", "--models-dir", str(d / "models")]
        if params.get("skip_projections"):
            eval_cmd.append("--skip-projections")
        name = f"{pipeline} / {mode}"
        stop = [Cmd([VAST_STOP], steps=0)] if _wants_remote(params) else []
        return [Cmd(gen_cmd, f"{name} — generate_embeddings"), *stop,
                Cmd(eval_cmd, f"{name} — train_evaluate_visualize")]

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
        job = Job(id=uuid.uuid4().hex[:12], stage=stage, params=params, cmds=cmds,
                  env=job_env(params))
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

    def _vast_stop(self, job: Job, logf) -> None:
        """Stop the rented instance, unless something still needs it. Never
        raises: failing to stop must not turn a finished job into a failed one,
        but it is said loudly, because it is the user's money."""
        if job.vast_stopped:
            return
        job.vast_stopped = True
        with self._lock:
            waiting = any(j.status == "queued" and j.remote for j in self.jobs.values())
        if waiting:
            msg = "left running: another queued job needs it"
        elif settings.get("VAST_AUTO_STOP") != "1" or not settings.get("VAST_API_KEY"):
            msg = "auto-stop is off. STOP THE INSTANCE YOURSELF in the Vast console to end billing"
        else:
            try:
                msg = vast.stop_instance()
            except Exception as e:
                msg = f"AUTO-STOP FAILED ({e}). Stop the instance yourself in the Vast console"
        logf.write(f"\n[Vast.ai] {msg}\n")
        logf.flush()

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
                    if cmd.argv == [VAST_STOP]:
                        self._vast_stop(job, logf)
                        continue
                    if cmd.label:
                        logf.write("##step " + json.dumps({"label": cmd.label}) + "\n")
                    logf.write("$ " + " ".join(cmd.argv) + "\n")
                    logf.flush()
                    # the pipeline scripts print emoji; force UTF-8 stdio on Windows
                    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
                           **job.env}
                    proc = subprocess.Popen(cmd.argv, stdout=logf, stderr=subprocess.STDOUT,
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
                if not ok and job.remote and not job.vast_stopped:
                    # the remote run is detached on purpose, so killing our side
                    # leaves it going; end it, then stop paying for the GPU
                    try:
                        vast.kill_remote()
                    except Exception as e:
                        logf.write(f"\n[Vast.ai] could not reach the instance to end the run: {e}\n")
                    self._vast_stop(job, logf)
                if ok:
                    job.status = "done"
                    job.returncode = 0
            job.scan()
            job.finished = time.time()


QUEUE = JobQueue()
