"""The job runner's own logic: marker parsing, what a job is made of, settings.

Run: python -m app.test_jobs   (or under pytest)
"""

import os
import tempfile
from pathlib import Path

from app import jobs, settings, vast
from app.jobs import VAST_STOP, Job, build_cmds, job_env


def test_scan_reads_markers_across_partial_writes():
    with tempfile.TemporaryDirectory() as d:
        job = Job(id="t", stage="run", params={}, logs_root=Path(d))
        log = job.log_path
        log.write_bytes(b'##step {"label": "A"}\nnoise\n##progress {"phase": "embed", "it')
        job.scan()
        assert (job.step, job.step_label, job.phase) == (1, "A", None)   # half a line is held back
        with open(log, "ab") as f:
            # the rest of that line, then a marker on the end of a tqdm bar (tqdm
            # draws to stderr with \r and no newline, so the two share a line)
            f.write(b'em": "openai", "done": 3, "total": 9}\n')
            f.write(b'\rBatches: 67%|## | 2/3 [00:04]##progress {"phase": "embed", "item": "openai", "done": 5, "total": 9}\n')
        job.scan()
        assert job.phase == {"phase": "embed", "item": "openai", "done": 5, "total": 9}
        with open(log, "ab") as f:
            f.write(b'##progress not json\n##step {"label": "B"}\n')
        job.scan()
        assert (job.step, job.step_label, job.phase) == (2, "B", None)   # a new step clears the phase


def test_sweep_shape():
    local = build_cmds("sweep", {"models": "minilm,qwen3", "force": True})
    assert len(local) == 1 and local[0].steps == 19 and "--force" in local[0].argv

    # a smoke sweep must not write into the real run dirs
    smoke = build_cmds("sweep", {"models": "minilm", "limit": 50})[0].argv
    assert Path(smoke[smoke.index("--runs-dir") + 1]).name == "_smoke"

    remote = build_cmds("sweep", {"models": "minilm,qwen3", "force": True, "qwen3_remote": True})
    embed, stop, train = remote
    assert stop.argv == [VAST_STOP] and stop.steps == 0
    assert "--embed-only" in embed.argv and "--force" in embed.argv
    # the instance is gone by now, so the training pass must never re-embed
    assert "--train-only" in train.argv and "--force" not in train.argv
    assert sum(c.steps for c in remote) == 19

    # asking for Vast without qwen3 in the run is a plain local sweep
    assert len(build_cmds("sweep", {"models": "minilm", "qwen3_remote": True})) == 1
    assert job_env({"models": "minilm", "qwen3_remote": True}) == {}


def test_full_builds_only_missing_datasets(monkeypatch=None):
    real = jobs.data_path
    try:
        jobs.data_path = lambda p: Path("/nope") / p          # nothing on disk
        scripts = [Path(c.argv[2]).name for c in build_cmds("full", {"models": "minilm"})]
        assert scripts == ["download_dataset.py", "organic_dataset.py", "run_all_pipelines.py"]
        jobs.data_path = lambda p: Path(__file__)             # everything on disk
        assert len(build_cmds("full", {"models": "minilm"})) == 1
        assert len(build_cmds("full", {"models": "minilm", "rebuild_data": True})) == 3
    finally:
        jobs.data_path = real


def test_vast_target_parsing():
    assert vast.parse_target("ssh -p 12345 root@ssh4.vast.ai -L 8080:localhost:8080") == ("root", "ssh4.vast.ai", 12345)
    assert vast.parse_target("ssh root@1.2.3.4 -p 40022") == ("root", "1.2.3.4", 40022)
    assert vast.parse_target("root@host.example") == ("root", "host.example", 22)
    for bad in ("ssh4.vast.ai", "ssh root@-oProxyCommand=x", ""):
        try:
            vast.parse_target(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")


def test_settings_roundtrip():
    real, saved = settings.ENV_PATH, {k: os.environ.get(k) for k in settings.FIELDS}
    with tempfile.TemporaryDirectory() as d:
        try:
            settings.ENV_PATH = Path(d) / ".env"
            key = Path(d) / "id key"          # a path with a space and backslashes
            key.write_text("x")
            settings.save({"OPENAI_API_KEY": "sk-test-1234567890", "OPENAI_TPM": "5000",
                           "VAST_SSH": "ssh -p 2 root@h.example", "VAST_SSH_KEY": str(key)})
            pub = settings.public()
            assert pub["OPENAI_API_KEY"] == {"set": True, "value": "…7890"}   # never the key itself
            assert pub["VAST_SSH_KEY"]["value"] == str(key)

            # what a restarted server would load back from the file
            from dotenv import dotenv_values
            on_disk = dotenv_values(settings.ENV_PATH)
            assert on_disk["VAST_SSH_KEY"] == str(key) and on_disk["OPENAI_API_KEY"] == "sk-test-1234567890"

            for bad in ({"OPENAI_TPM": "lots"}, {"VAST_SSH": "nonsense"}, {"NOPE": "1"},
                        {"OPENAI_TPM": "7", "VAST_SSH_KEY": str(key) + ".missing"}):
                try:
                    settings.save(bad)
                except ValueError:
                    continue
                raise AssertionError(f"accepted {bad}")
            assert os.environ["OPENAI_TPM"] == "5000"   # a rejected save changes nothing

            settings.save({"OPENAI_API_KEY": ""})
            assert "OPENAI_API_KEY" not in os.environ
            assert "OPENAI_API_KEY" not in dotenv_values(settings.ENV_PATH)
        finally:
            settings.ENV_PATH = real
            for k, v in saved.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
