"""qwen3 embeddings on a Vast.ai instance over ssh/scp: upload the dataset, embed there, download the .npy."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import progress  # noqa: E402  (src/progress.py)

REMOTE_DIR = "thesis_embed"          # relative to the remote home
NPY = "qwen3_prompt.npy"
API = "https://console.vast.ai/api"
# generate_embeddings.py imports all of these at the top, even for qwen3 alone
REMOTE_PACKAGES = ["sentence-transformers", "transformers", "pandas", "tqdm",
                   "openai", "python-dotenv", "tiktoken"]
# pinned to requirements.txt; torch is whatever CUDA build the image ships
PINNED = ("sentence-transformers", "transformers")


class VastError(Exception):
    pass


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


# ---------------- ssh ----------------

def parse_target(cmd: str) -> tuple[str, str, int]:
    """(user, host, port) from the console's 'ssh -p 12345 root@ssh4.vast.ai'; strict, since it lands in an argv."""
    m = re.search(r"(?<![\w.-])([A-Za-z_][\w-]*)@([A-Za-z0-9][A-Za-z0-9.-]*)", cmd)
    if not m:
        raise ValueError("VAST_SSH: expected the SSH command from the Vast console, "
                         "like: ssh -p 12345 root@ssh4.vast.ai")
    p = re.search(r"(?:^|\s)-p\s*(\d+)", cmd)
    return m.group(1), m.group(2), int(p.group(1)) if p else 22


def target() -> tuple[str, str, int]:
    raw = _env("VAST_SSH")
    if not raw:
        raise VastError("Vast.ai is not set up: add the instance's SSH command in Settings")
    try:
        return parse_target(raw)
    except ValueError as e:
        raise VastError(str(e))


def _opts() -> list[str]:
    o = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
         "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30"]
    key = _env("VAST_SSH_KEY")
    if key:
        o += ["-i", os.path.expanduser(key)]
    return o


def _bin(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise VastError(f"'{name}' was not found on PATH (install OpenSSH: Windows Settings "
                        f"→ Optional features → OpenSSH Client)")
    return path


def ssh(remote_cmd: str, timeout: float = 120, stream: bool = False) -> subprocess.CompletedProcess:
    """Run one command on the instance; stream=True sends its output to our stdout (the job log)."""
    user, host, port = target()
    argv = [_bin("ssh"), "-p", str(port), *_opts(), f"{user}@{host}", remote_cmd]
    if stream:
        sys.stdout.flush()
        return subprocess.run(argv, timeout=timeout, stdin=subprocess.DEVNULL)
    return subprocess.run(argv, timeout=timeout, capture_output=True, stdin=subprocess.DEVNULL)


def _scp(src: str, dst: str, timeout: float = 3600) -> None:
    _, _, port = target()
    r = subprocess.run([_bin("scp"), "-P", str(port), "-q", *_opts(), src, dst],
                       timeout=timeout, capture_output=True, stdin=subprocess.DEVNULL)
    if r.returncode != 0:
        raise VastError(f"scp failed: {r.stderr.decode('utf-8', 'replace').strip()}")


def _remote(path: str) -> str:
    user, host, _ = target()
    return f"{user}@{host}:{path}"


def _ok(r: subprocess.CompletedProcess, what: str) -> str:
    out = (r.stdout or b"").decode("utf-8", "replace")
    if r.returncode != 0:
        err = (r.stderr or b"").decode("utf-8", "replace").strip()
        hint = ""
        if "HOST IDENTIFICATION HAS CHANGED" in err:
            _, host, port = target()
            hint = (f"\nVast reuses addresses between instances. If this is your new "
                    f"instance, run:  ssh-keygen -R \"[{host}]:{port}\"")
        raise VastError(f"{what} failed (ssh exit {r.returncode}): {err or out.strip()}{hint}")
    return out


# the venv Vast's own images ship, else the system python
_PY = "$( [ -x /venv/main/bin/python ] && echo /venv/main/bin/python || command -v python3 )"


def test() -> dict:
    """Settings' Test button: can we log in, and is there a GPU?"""
    try:
        gpu = _ok(ssh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
                      timeout=45), "connect").strip()
    except (VastError, subprocess.TimeoutExpired) as e:
        return {"ok": False, "detail": str(e)}
    detail = f"connected · GPU: {gpu or 'none reported'}"
    if _env("VAST_API_KEY"):
        try:
            inst = find_instance()
            detail += f" · auto-stop will target instance {inst}"
        except Exception as e:
            return {"ok": False, "detail": f"{detail} · but auto-stop will NOT work: {e}"}
    return {"ok": True, "detail": detail}


def kill_remote() -> None:
    """End a detached embed run (on cancel); one job runs at a time, so every generate_embeddings goes."""
    ssh("pkill -f generate_embeddings.py", timeout=45)


# ---------------- Vast API (auto-stop) ----------------

def _api(method: str, path: str, **kw):
    import requests
    key = _env("VAST_API_KEY")
    if not key:
        raise VastError("no Vast API key saved")
    r = requests.request(method, API + path, timeout=30,
                         headers={"Authorization": f"Bearer {key}"}, **kw)
    if r.status_code >= 400:
        raise VastError(f"Vast API {method} {path}: HTTP {r.status_code} {r.text[:200]}")
    return r.json()


def find_instance() -> int:
    """The instance to stop: the ID from Settings, else the account's instance the SSH command points at."""
    if _env("VAST_INSTANCE_ID"):
        return int(_env("VAST_INSTANCE_ID"))
    _, host, port = target()
    try:
        listing = _api("GET", "/v0/instances/", params={"owner": "me"})
    except VastError:
        listing = _api("GET", "/v1/instances")
    for inst in listing.get("instances", []):
        if inst.get("ssh_host") == host and str(inst.get("ssh_port")) == str(port):
            return int(inst["id"])
        direct = [m.get("HostPort") for m in (inst.get("ports") or {}).get("22/tcp") or []]
        if inst.get("public_ipaddr") == host and str(port) in map(str, direct):
            return int(inst["id"])
    raise VastError(f"none of this account's instances matches {host}:{port}; "
                    f"put the instance ID in Settings")


def stop_instance() -> str:
    """Stop (not destroy) the instance: GPU billing ends, the disk is kept."""
    inst = find_instance()
    res = _api("PUT", f"/v0/instances/{inst}/", json={"state": "stopped"})
    if not res.get("success", False):
        raise VastError(f"Vast refused to stop instance {inst}: {res}")
    return f"instance {inst} stopped"


# ---------------- the embed run ----------------

def _pins() -> list[str]:
    """REMOTE_PACKAGES, with the model-code ones at requirements.txt's versions."""
    pins = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        name, _, ver = line.strip().partition("==")
        if name in PINNED and ver:
            pins[name] = f"{name}=={ver}"
    return [pins.get(p, p) for p in REMOTE_PACKAGES]


def embed(data: Path, mode: str, embed_dir: Path, force: bool) -> int:
    """Embed one dataset with qwen3 on the instance; the run is detached, so re-running re-attaches to it."""
    out = embed_dir / NPY
    if out.exists() and not force:
        print(f"  {NPY} already exists. Use --force to overwrite.")
        return 0

    size = data.stat().st_size
    # a remote dataset is identified by name + byte size, not a content hash
    rdata = f"{REMOTE_DIR}/data/{size}_{data.name}"
    run = f"{REMOTE_DIR}/runs/{data.stem}_{size}_{mode}"
    if not re.fullmatch(r"[\w./-]+", rdata):
        raise VastError(f"unexpected characters in dataset name: {data.name}")

    progress.mark("connect", "Vast.ai instance")
    state, have = _ok(ssh(
        f"mkdir -p {REMOTE_DIR}/data {run}; "
        f"cat {run}/exit 2>/dev/null || "
        f"{{ kill -0 $(cat {run}/pid 2>/dev/null) 2>/dev/null && echo running; }} || echo none; "
        f"stat -c %s {rdata} 2>/dev/null || echo 0"), "connect").split()[-2:]

    if state == "running":
        print("  re-attaching to the run already in progress on the instance")
    elif state == "0":
        print("  the instance already finished this run; downloading it")
    else:
        progress.mark("upload", data.name)
        for f in ("generate_embeddings.py", "progress.py"):
            _scp(str(SRC / f), _remote(f"{REMOTE_DIR}/{f}"))
        if have != str(size):
            print(f"  uploading {data.name} ({size / 1e6:.1f} MB)...", flush=True)
            _scp(str(data), _remote(rdata))

        progress.mark("setup", "Python packages on the instance")
        r = ssh(f"cd {REMOTE_DIR} && PY={_PY} && "
                f"$PY -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 3)' && "
                f"{{ test -f .deps || {{ $PY -m pip install -q {' '.join(map(shlex.quote, _pins()))} "
                f"&& touch .deps; }}; }}", timeout=1800, stream=True)
        if r.returncode != 0:
            raise VastError(f"setting up the instance failed (exit {r.returncode}). It needs "
                            "PyTorch with a working GPU (rent it with a PyTorch template) "
                            "and internet access for pip; see the output above.")

        rel = f"runs/{Path(run).name}"   # the run dir, as seen from REMOTE_DIR
        inner = (f"$PY -u generate_embeddings.py --data data/{size}_{data.name} --mode {mode} "
                 f"--embed-dir {rel}/out --models qwen3 --_single qwen3 --force; "
                 f"echo $? > {rel}/exit")
        # only the nohup'd shell is backgrounded, every stream redirected, or ssh would never return
        _ok(ssh(f"cd {REMOTE_DIR} && export PY={_PY} && rm -f {rel}/exit && "
                f"( nohup sh -c {shlex.quote(inner)} > {rel}/log 2>&1 < /dev/null & "
                f"echo $! > {rel}/pid )"), "start")
        state = "running"

    # follow the remote log; its ##progress lines pass straight through to ours
    offset, failures = 0, 0
    while state == "running":
        time.sleep(5)
        try:
            r = ssh(f"cat {run}/exit 2>/dev/null || echo running; "
                    f"tail -c +{offset + 1} {run}/log 2>/dev/null", timeout=90)
        except subprocess.TimeoutExpired:
            r = None
        if r is None or r.returncode != 0:
            failures += 1
            if failures > 60:   # ~10 minutes of silence
                raise VastError("lost the connection to the instance. The run is still going "
                                "there; start this again to re-attach.")
            time.sleep(5)
            continue
        failures = 0
        head, _, body = r.stdout.partition(b"\n")
        state = head.decode().strip() or "running"
        if body:
            offset += len(body)
            sys.stdout.write(body.decode("utf-8", "replace"))
            sys.stdout.flush()

    if state != "0":
        print(f"\n  the remote embedding run failed (exit {state}); see its output above.")
        return 1

    progress.mark("download", NPY)
    embed_dir.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(".part")
    try:
        _scp(_remote(f"{run}/out/{NPY}"), str(part))
    except VastError as e:
        # clear the exit marker so the next attempt re-runs instead of re-downloading nothing
        ssh(f"rm -f {run}/exit", timeout=45)
        print(f"\n  no embeddings came back from the instance: {e}")
        return 1
    part.replace(out)
    ssh(f"rm -rf {run}", timeout=45)
    print(f"  Saved embeddings to: {out.name} ({out.stat().st_size / 1e6:.2f} MB, from Vast.ai)")
    return 0


def main() -> int:
    # run by src/generate_embeddings.py as: python -m app.vast embed --data ... --mode ... --embed-dir ...
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed", help="Embed one dataset with qwen3 on the instance.")
    e.add_argument("--data", type=Path, required=True)
    e.add_argument("--mode", choices=["normal", "user_intent_only", "context_only"], required=True)
    e.add_argument("--embed-dir", type=Path, required=True)
    e.add_argument("--force", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    try:
        return embed(args.data, args.mode, args.embed_dir, args.force)
    except VastError as e:
        print(f"Vast.ai error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
