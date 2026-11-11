"""Dashboard settings in the gitignored .env; saving updates os.environ too, so the next job sees them."""

from __future__ import annotations

import os

from dotenv import load_dotenv, set_key, unset_key

from app.paths import ROOT

ENV_PATH = ROOT / ".env"

# name -> is it a secret (secrets are write-only: the browser only ever sees a masked tail)
FIELDS = {
    "OPENAI_API_KEY": True,
    "OPENAI_TPM": False,
    "VAST_SSH": False,
    "VAST_SSH_KEY": False,
    "VAST_API_KEY": True,
    "VAST_INSTANCE_ID": False,
    "VAST_AUTO_STOP": False,
}

load_dotenv(ENV_PATH)


def get(name: str) -> str:
    return os.environ.get(name, "").strip()


def _mask(v: str) -> str:
    return ("…" + v[-4:]) if len(v) > 8 else "…"


def public() -> dict:
    out = {}
    for name, secret in FIELDS.items():
        v = get(name)
        out[name] = {"set": bool(v), "value": (_mask(v) if v else "") if secret else v}
    return out


def _check(name: str, v: str) -> None:
    if "\n" in v or "\r" in v:
        raise ValueError(f"{name}: must be a single line")
    if name in ("OPENAI_TPM", "VAST_INSTANCE_ID") and not v.isdigit():
        raise ValueError(f"{name}: must be a whole number")
    if name == "VAST_AUTO_STOP" and v not in ("0", "1"):
        raise ValueError(f"{name}: must be 0 or 1")
    if name == "VAST_SSH":
        from app import vast
        vast.parse_target(v)  # raises ValueError with what is wrong
    if name == "VAST_SSH_KEY" and not os.path.isfile(os.path.expanduser(v)):
        raise ValueError(f"VAST_SSH_KEY: no file at {v}")


def save(values: dict) -> None:
    """Apply {name: value}; "" clears a field, an omitted name is untouched, and nothing is written unless all are valid."""
    clean = {}
    for name, v in values.items():
        if name not in FIELDS:
            raise ValueError(f"unknown setting: {name}")
        v = str(v).strip()
        if v:
            _check(name, v)
        clean[name] = v

    ENV_PATH.touch(exist_ok=True)
    for name, v in clean.items():
        if v:
            set_key(ENV_PATH, name, v)
            os.environ[name] = v
        else:
            if os.environ.pop(name, None) is not None:
                unset_key(ENV_PATH, name)

    if "OPENAI_API_KEY" in clean:
        # the Simulation tab caches an OpenAI client built from the old key
        from app import detect
        detect._embedder_cache.pop("openai", None)


def test_openai() -> dict:
    if not get("OPENAI_API_KEY"):
        return {"ok": False, "detail": "no key saved"}
    from openai import OpenAI
    try:
        # free, and proves both the key and access to the model the pipeline uses
        OpenAI(timeout=20).models.retrieve("text-embedding-3-small")
    except Exception as e:
        return {"ok": False, "detail": f"{type(e).__name__}: {e}"}
    return {"ok": True, "detail": "key works, text-embedding-3-small is available"}
