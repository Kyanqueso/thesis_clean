"""thesis_clean dashboard.

Run from the repo root:
    venv\\Scripts\\python.exe -m app.server        (http://127.0.0.1:8756)
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import detect, paths, runs
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
    """One row per pipeline x embedding x classifier: reference-mode metrics,
    ablation degradation and McNemar significance in a single grid."""
    if reference not in runs.MODES:
        raise HTTPException(404, f"unknown mode: {reference}")
    return runs.master_table(reference)


@app.get("/api/curves")
def get_curves(pipeline: str, mode: str):
    """ROC/PR curves for a run. A run without predictions.npz answers 200 with
    available:false and how to get them, rather than failing."""
    if pipeline not in runs.PIPELINES or mode not in runs.MODES:
        raise HTTPException(404, "unknown run")
    return runs.curves(pipeline, mode)


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
    """Map a browser-supplied relative path onto disk (see paths.INSPECT_ROOTS)."""
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
    import numpy as np
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
        lines = []
        with open(p, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                lines.append(line[:600].rstrip())
        return {"type": "text", "text": "\n".join(lines) + "\n… (first 3 rows)"}
    if suf in (".json", ".csv", ".log", ".txt", ".html"):
        text = p.read_text(encoding="utf-8", errors="replace")
        cap = 20000
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


# ---------------- breakdowns ----------------

@app.get("/api/breakdown/fields")
def get_breakdown_fields(pipeline: str):
    if pipeline not in runs.PIPELINES:
        raise HTTPException(404, "unknown pipeline")
    return runs.breakdown_fields(pipeline)


@app.get("/api/breakdown")
def get_breakdown(pipeline: str, mode: str, key: str, by: str):
    if pipeline not in runs.PIPELINES or mode not in runs.MODES:
        raise HTTPException(404, "unknown run")
    return runs.breakdown(pipeline, mode, key, by)


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
    text = ""
    if job.log_path.is_file():
        with open(job.log_path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(max(0, offset))
            text = f.read()
    return {"text": text, "offset": offset + len(text.encode("utf-8", errors="replace")),
            "status": job.status}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    if not QUEUE.cancel(job_id):
        raise HTTPException(400, "job not cancellable")
    return {"ok": True}


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
        out = detect.score(req.pipeline, req.mode, req.emb, req.clf,
                           req.user_intent, req.context)
    except Exception as e:  # embedding backends can fail in many ways (no key, no VRAM...)
        return JSONResponse(status_code=500, content={"error": f"{type(e).__name__}: {e}"})
    if "error" in out:
        return JSONResponse(status_code=400, content=out)
    return out


# ---------------- static ----------------

@app.middleware("http")
async def no_cache_static(request, call_next):
    response = await call_next(request)
    # dev server: always revalidate so edits to static assets show up on reload
    response.headers["Cache-Control"] = "no-cache"
    return response


app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(APP_DIR / "static" / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server:app", host="127.0.0.1", port=8756)
