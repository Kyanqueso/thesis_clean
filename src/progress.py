"""Progress marker lines (##step / ##progress) that app/jobs.py reads out of a job's log."""

import json
import time


def step(label: str) -> None:
    print("##step " + json.dumps({"label": label}), flush=True)


# total=0 means "running, size unknown"
def mark(phase: str, item: str = "", done: int = 0, total: int = 0) -> None:
    print("##progress " + json.dumps(
        {"phase": phase, "item": item, "done": int(done), "total": int(total)}), flush=True)


def track(iterable, phase: str, item: str = "", total: int | None = None, every: float = 1.0):
    """Yield from iterable, marking progress at most once per `every` seconds (and at start/end)."""
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
