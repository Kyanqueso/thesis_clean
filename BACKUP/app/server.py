"""Dashboard server. From the repo root: venv\\Scripts\\python.exe -m uvicorn app.server:app --port 8756"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import detect, paths, runs, settings, vast
from app.jobs import QUEUE, ValidationError

APP_DIR = Path(__file__).resolve().parent
app = FastAPI(title="thesis_clean dashboard")


# ---------------- state & results ----------------

@app.get("/api/state")
def get_state():
    return runs.scan_state()


@app.get("/api/results")
def get_results():
    return {"rows": runs.all_results()}


@app.get("/api/master")
def get_master(reference: str = Query(default="normal")):
    if reference not in runs.MODES:
        raise HTTPException(404, f"unknown mode: {reference}")
    return runs.master_table(reference)


@app.get("/api/resources")
def get_resources():
    """Cost of running the detector on the benchmark PC: GPU, CPU, RAM, time and disk."""
    return runs.resources()


@app.get("/api/figure")
def get_figure(pipeline: str, mode: str, name: str):
    if pipeline not in runs.PIPELINES or mode not in runs.MODES:
        raise HTTPException(404, "unknown run")
    fig_dir = (runs.run_dir(pipeline, mode) / "figures").resolve()
    path = (fig_dir / name).resolve()
    if fig_dir not in path.parents or path.suffix != ".png" or not path.is_file():
        raise HTTPException(404, "figure not found")
    return FileResponse(path)


@app.get("/api/sample")
def get_sample(pipeline: str, n: int = Query(default=1, ge=1, le=25)):
    if pipeline not in runs.PIPELINES:
        raise HTTPException(404, "unknown pipeline")
    rows = runs.sample_rows(pipeline, n=n)
    if not rows:
        raise HTTPException(404, f"{runs.PIPELINES[pipeline]['data']} is not in "
                                 f"{paths.DATA_PREFIX}/ — download or build it from the Run tab")
    return {"rows": rows}


# ---------------- file explorer ----------------

def _safe_path(path: str) -> Path:
    """Map a browser-supplied relative path onto disk, refusing anything outside the data and runs trees."""
    if "\\" in path or ".." in path.split("/"):
        raise HTTPException(400, "bad path")
    p = paths.resolve(path)
    if p is None:
        raise HTTPException(404, "path outside allowed roots")
    if not p.is_file():
        raise HTTPException(404, "file not found")
    return p


@app.get("/api/files")
def get_files():
    return runs.file_tree()


@app.get("/api/file/inspect")
def inspect_file(path: str):
    p = _safe_path(path)
    suf = p.suffix.lower()
    if suf == ".npy":
        arr = np.load(p, mmap_mode="r")
        return {"type": "text", "text": f"numpy array · shape {tuple(arr.shape)} · dtype {arr.dtype}"}
    if suf == ".npz":
        z = np.load(p)
        lines = [f"{k}: shape {tuple(z[k].shape)} · dtype {z[k].dtype}" for k in z.files]
        return {"type": "text", "text": "numpy archive\n" + "\n".join(lines)}
    if suf == ".jsonl":
        with open(p, encoding="utf-8", errors="replace") as f:
            lines = [line[:600].rstrip() for _, line in zip(range(3), f)]
        return {"type": "text", "text": "\n".join(lines) + "\n… (first 3 rows)"}
    if suf == ".csv":
        df = pd.read_csv(p, nrows=10)
        data = json.loads(df.to_json(orient="split"))  # numpy scalars are not JSON
        return {"type": "table", "cols": data["columns"], "rows": data["data"],
                "note": f"first {len(df)} rows"}
    if suf in (".json", ".log", ".txt", ".md", ".html"):
        text = p.read_text(encoding="utf-8", errors="replace")
        cap = 4000  # a peek, not a viewer
        return {"type": "text", "text": text[:cap] + ("\n… (truncated)" if len(text) > cap else "")}
    if suf == ".png":
        return {"type": "image"}
    return {"type": "text", "text": f"binary file · {p.stat().st_size / 1e6:.2f} MB"}


@app.get("/api/file/raw")
def raw_file(path: str):
    p = _safe_path(path)
    if p.suffix.lower() != ".png":
        raise HTTPException(400, "raw serving is limited to .png")
    return FileResponse(p)


# ---------------- jobs ----------------

class JobRequest(BaseModel):
    stage: str
    params: dict = {}


@app.post("/api/jobs")
def post_job(req: JobRequest):
    try:
        job = QUEUE.enqueue(req.stage, req.params)
    except ValidationError as e:
        raise HTTPException(400, str(e))
    return job.public()


@app.get("/api/jobs")
def get_jobs():
    return {"jobs": QUEUE.list()}


@app.get("/api/jobs/{job_id}/log")
def get_log(job_id: str, offset: int = 0):
    job = QUEUE.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    offset, data = max(0, offset), b""
    # bytes, not text: Windows logs end lines in \r\n, and counting decoded characters drifted the offset
    if job.log_path.is_file():
        with open(job.log_path, "rb") as f:
            f.seek(offset)
            data = f.read()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        if e.start >= len(data) - 3:  # a character cut in half at the end: send it with the next poll
            data = data[:e.start]
        text = data.decode("utf-8", "replace")
    return {"text": text.replace("\r\n", "\n").replace("\r", "\n"),
            "offset": offset + len(data), "status": job.status}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    if not QUEUE.cancel(job_id):
        raise HTTPException(400, "job not cancellable")
    return {"ok": True}


# ---------------- settings ----------------

@app.get("/api/settings")
def get_settings():
    return settings.public()


@app.post("/api/settings")
def post_settings(values: dict):
    try:
        settings.save(values)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return settings.public()


@app.post("/api/settings/test/{what}")
def test_settings(what: str):
    if what == "openai":
        return settings.test_openai()
    if what == "vast":
        return vast.test()
    raise HTTPException(404, "unknown test")


# ---------------- detect ----------------

class DetectRequest(BaseModel):
    pipeline: str
    mode: str
    emb: str
    clf: str
    user_intent: str = ""
    context: str = ""


@app.get("/api/detect/options")
def detect_options():
    return detect.options()


@app.post("/api/detect")
def post_detect(req: DetectRequest):
    try:
        out = detect.score(req.pipeline, req.mode, req.emb, req.clf, req.user_intent, req.context)
    except Exception as e:  # embedding backends fail in many ways (no key, no memory...)
        return JSONResponse(status_code=500, content={"error": f"{type(e).__name__}: {e}"})
    if "error" in out:
        return JSONResponse(status_code=400, content=out)
    return out


# ---------------- static ----------------

@app.middleware("http")
async def no_cache_static(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache"  # edits to static files show on reload
    return response


app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(APP_DIR / "static" / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server:app", host="127.0.0.1", port=8756)
