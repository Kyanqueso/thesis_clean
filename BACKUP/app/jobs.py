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

import pandas as pd

from app import paths, settings, vast
from app.paths import ROOT, write_run_dir
from app.runs import EMB_MODELS, MODES, PIPELINES, data_path
from pipeline_paths import SMOKE_SUBDIR

SRC_DIR = ROOT / "src"
LOG_DIR = paths.RUNS_DIR / "_logs"
ALL_RUNS = [(p, m) for p in PIPELINES for m in MODES]

# a pseudo-command: the worker stops the Vast.ai instance here, once no later step needs the GPU
VAST_STOP = "@vast-stop"

# the marker lines src/progress.py prints
MARKER = re.compile(r"##(step|progress) (\{.*\})")


class ValidationError(Exception):
    pass


@dataclass
class Cmd:
    argv: list[str]
    label: str = ""   # this command's step on the overall bar
    run: str = ""     # "pipeline / mode" it belongs to; a failure skips the rest of that run, not the job


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
    logs_root: Path = LOG_DIR
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
        """Fold the marker lines the log gained since the last call into step / phase."""
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
            # tqdm redraws with \r, so split on both; the last partial line waits for the next scan
            lines = re.split(r"[\r\n]", self._tail + chunk.decode("utf-8", "replace"))
            self._tail = lines.pop()
            for line in lines:
                # searched, not anchored: a marker often lands on the end of a tqdm bar
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
            "progress": {"step": self.step, "steps": sum(c.argv != [VAST_STOP] for c in self.cmds),
                         "label": self.step_label, "phase": self.phase},
        }


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


def _limited_copy(src: Path, limit: int, dest: Path) -> Path:
    """Write the first `limit` rows of src to dest (for smoke testing)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "r", encoding="utf-8") as fin, open(dest, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i >= limit:
                break
            fout.write(line)
    return dest


def _already_trained(results_dir: Path, models: str | None) -> bool:
    """True when this run's results already cover every requested embedding, from the current pipeline."""
    csv_path = results_dir / "full_evaluation_results.csv"
    # operating_points.csv is only written by the current code, so older results count as unfinished
    if not csv_path.is_file() or not (results_dir / "operating_points.csv").is_file():
        return False
    wanted = {m.strip().upper() for m in (models.split(",") if models else EMB_MODELS)}
    try:
        have = set(pd.read_csv(csv_path, usecols=["Embeddings"])["Embeddings"].astype(str))
    except (ValueError, OSError):
        return False
    return wanted <= have


def _dataset_cmds(py: str, which: str) -> list[Cmd]:
    if which == "alamsabi":
        return [Cmd([py, "-u", str(SRC_DIR / "download_dataset.py")], "Download the Alamsabi dataset")]
    return [Cmd([py, "-u", str(SRC_DIR / "organic_dataset.py")],
                "Build the organic datasets (collect, clean, de-identify)")]


def _run_cmds(py: str, params: dict, runs: list[tuple[str, str]], data_later: bool = False) -> list[Cmd]:
    """Embed + train commands per (pipeline, mode); data_later = the job builds the datasets first."""
    limit = None if data_later else params.get("limit")
    embed_extra = ["--force"] if params.get("force") else []
    if params.get("models"):
        embed_extra += ["--models", str(params["models"])]
    copies: dict[str, Path] = {}
    embeds, trains = [], []
    for pipeline, mode in runs:
        data = data_path(pipeline)
        if not data_later and not data.is_file():
            raise ValidationError(f"dataset missing: {data.name} - build/download it first")
        if limit:  # a smoke test: a row-limited copy, written under runs/_smoke so real results are safe
            if pipeline not in copies:
                copies[pipeline] = _limited_copy(
                    data, int(limit), paths.RUNS_DIR / SMOKE_SUBDIR / "_smoke_data" / f"{pipeline}.jsonl")
            data = copies[pipeline]
        d = write_run_dir(pipeline, mode, smoke=bool(limit))
        if params.get("skip_done") and not params.get("force") and _already_trained(d / "results", params.get("models")):
            continue
        name = f"{pipeline} / {mode}"
        embeds.append(Cmd([py, "-u", str(SRC_DIR / "generate_embeddings.py"), "--data", str(data),
                           "--mode", mode, "--embed-dir", str(d / "embeddings"), *embed_extra],
                          f"{name} — generate_embeddings", run=name))
        trains.append(Cmd([py, "-u", str(SRC_DIR / "train_evaluate_visualize.py"),
                           "--data", str(data), "--run-dir", str(d)],
                          f"{name} — train_evaluate_visualize", run=name))
    if _wants_remote(params):
        # with a rented GPU, embed everything first and stop the instance before the CPU-only training
        return embeds + [Cmd([VAST_STOP])] + trains
    return [c for pair in zip(embeds, trains) for c in pair]


def _resource_cmds(py: str, params: dict) -> list[Cmd]:
    """One benchmark_resources.py call per pipeline whose dataset is on disk; results merge into one file."""
    stages = str(params.get("stages") or "embed,classify")
    if not set(stages.split(",")) <= {"embed", "classify"}:
        raise ValidationError(f"unknown stages: {stages}")
    extra = ["--stages", stages]
    if params.get("models"):
        extra += ["--models", str(params["models"])]
    cmds = [Cmd([py, "-u", str(SRC_DIR / "benchmark_resources.py"), "--pipelines", p, *extra],
                f"{p} — benchmark_resources", run=f"{p} / resources")
            for p in PIPELINES if data_path(p).is_file()]
    if not cmds:
        raise ValidationError("no dataset on disk to measure - build/download them first")
    return cmds


def build_cmds(stage: str, params: dict) -> list[Cmd]:
    py = sys.executable

    if stage == "sweep":
        cmds = _run_cmds(py, params, ALL_RUNS)

    elif stage == "full":
        rebuild = bool(params.get("rebuild_data"))
        cmds = []
        if rebuild or not data_path("pipeline1_alamsabi").is_file():
            cmds += _dataset_cmds(py, "alamsabi")
        if rebuild or not all(data_path(p).is_file() for p in PIPELINES if p != "pipeline1_alamsabi"):
            cmds += _dataset_cmds(py, "organic")
        if rebuild:
            # new datasets make every cached embedding and result stale
            params = {**params, "force": True, "skip_done": False}
        cmds += _run_cmds(py, params, ALL_RUNS, data_later=bool(cmds))

    elif stage == "run":
        pipeline, mode = params.get("pipeline"), params.get("mode")
        if pipeline not in PIPELINES:
            raise ValidationError(f"unknown pipeline: {pipeline}")
        if mode not in MODES:
            raise ValidationError(f"unknown mode: {mode}")
        cmds = _run_cmds(py, params, [(pipeline, mode)])

    elif stage == "resources":
        cmds = _resource_cmds(py, params)

    else:
        raise ValidationError(f"unknown stage: {stage}")

    if not any(c.argv != [VAST_STOP] for c in cmds):
        raise ValidationError("nothing to run: every picked run already has results "
                              "(untick “skip finished runs” to redo them)")
    return cmds


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
        self.add(job)
        return job

    def add(self, job: Job) -> None:
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        job.log_path.write_text("", encoding="utf-8")
        with self._lock:
            self.jobs[job.id] = job
            self.order.append(job.id)
        self._event.set()

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
                    # /T kills the whole tree (generate_embeddings spawns one process per model)
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
        """Stop the rented instance unless a queued job still needs it; never raises, always logs."""
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

    def _run(self, job: Job, logf) -> None:
        """Run a job's commands; a failed step skips the rest of its run, a failed dataset step ends the job."""
        failed_runs: list[str] = []
        for cmd in job.cmds:
            if job.cancel_requested:
                break
            if cmd.argv == [VAST_STOP]:
                self._vast_stop(job, logf)
                continue
            logf.write("##step " + json.dumps({"label": cmd.label}) + "\n")
            if cmd.run in failed_runs:
                logf.write(f"skipped: an earlier step of {cmd.run} failed\n")
                logf.flush()
                continue
            logf.write("$ " + " ".join(cmd.argv) + "\n")
            logf.flush()
            # the scripts print non-ASCII text; force UTF-8 stdio on Windows
            env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", **job.env}
            proc = subprocess.Popen(cmd.argv, stdout=logf, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
            self._current = (job, proc)
            rc = proc.wait()
            self._current = None
            if job.cancel_requested:
                break
            if rc != 0:
                job.returncode = rc
                logf.write(f"\n[step failed: exit {rc}]\n")
                if not cmd.run:   # a dataset step: everything after it needs its output
                    job.status = "failed"
                    return
                failed_runs.append(cmd.run)

        if job.cancel_requested:
            job.status = "cancelled"
        elif failed_runs:
            job.status = "failed"
            logf.write(f"\n[{len(failed_runs)} run(s) failed, the rest completed: {', '.join(failed_runs)}]\n")
        else:
            job.status, job.returncode = "done", 0

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
                self._run(job, logf)
                if job.status != "done" and job.remote and not job.vast_stopped:
                    # the remote run is detached, so killing our side leaves it going: end it, then stop the GPU
                    try:
                        vast.kill_remote()
                    except Exception as e:
                        logf.write(f"\n[Vast.ai] could not reach the instance to end the run: {e}\n")
                    self._vast_stop(job, logf)
            job.scan()
            job.finished = time.time()


QUEUE = JobQueue()
