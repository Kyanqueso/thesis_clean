"""Progress markers the dashboard reads out of a job's log.

One line per marker, so a log stays greppable and nothing has to parse tqdm:

    ##step {"label": "pipeline1_alamsabi / normal — generate_embeddings"}
    ##progress {"phase": "embed", "item": "openai", "done": 120, "total": 900}

A step is one unit of the job's overall bar; progress is where the current step
is. total=0 means "running, size unknown". app/jobs.py is the reader.
"""

import json
import time


def step(label: str) -> None:
    print("##step " + json.dumps({"label": label}), flush=True)


def mark(phase: str, item: str = "", done: int = 0, total: int = 0) -> None:
    print("##progress " + json.dumps(
        {"phase": phase, "item": item, "done": int(done), "total": int(total)}), flush=True)


def track(iterable, phase: str, item: str = "", total: int | None = None, every: float = 1.0):
    """Yield from iterable, marking progress at most once per `every` seconds
    (and always at the start and the end)."""
    if total is None:
        total = len(iterable)
    mark(phase, item, 0, total)
    last, done = time.monotonic(), 0
    for x in iterable:
        yield x
        done += 1
        now = time.monotonic()
        if now - last >= every or done == total:
            last = now
            mark(phase, item, done, total)
