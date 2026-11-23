"""Job runner tests: marker parsing, job commands, settings. Run: python -m app.test_jobs (or pytest)"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from app import jobs, paths, runs, settings, vast
from app.jobs import VAST_STOP, Cmd, Job, build_cmds, job_env


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


def _script(cmd) -> str:
    return Path(cmd.argv[2]).name if len(cmd.argv) > 2 else cmd.argv[0]


def test_sweep_shape():
    real_data = jobs.data_path
    try:
        jobs.data_path = lambda p: Path(__file__)             # every dataset "on disk"
        local = build_cmds("sweep", {"models": "minilm,qwen3", "force": True})
        # embed then train, run by run, for all 9 runs
        assert [_script(c) for c in local[:2]] == ["generate_embeddings.py", "train_evaluate_visualize.py"]
        assert len(local) == 18 and all(c.run for c in local)
        assert all("--force" in c.argv for c in local if _script(c) == "generate_embeddings.py")

        remote = build_cmds("sweep", {"models": "minilm,qwen3", "force": True, "qwen3_remote": True})
        # every embed first, then the instance stops, then every train (which never re-embeds)
        assert [_script(c) for c in remote].index(VAST_STOP) == 9
        assert all(_script(c) == "train_evaluate_visualize.py" for c in remote[10:])

        # asking for Vast without qwen3 in the run is a plain local sweep
        assert VAST_STOP not in [_script(c) for c in build_cmds("sweep", {"models": "minilm", "qwen3_remote": True})]
        assert job_env({"models": "minilm", "qwen3_remote": True}) == {}
    finally:
        jobs.data_path = real_data


def test_smoke_stays_out_of_real_runs():
    real_runs, real_data = paths.RUNS_DIR, jobs.data_path
    with tempfile.TemporaryDirectory() as d:
        try:
            paths.RUNS_DIR = Path(d)
            data = Path(d) / "data.jsonl"
            data.write_text("".join(f'{{"i": {i}}}\n' for i in range(100)), encoding="utf-8")
            jobs.data_path = lambda p: data
            cmds = build_cmds("sweep", {"models": "minilm", "limit": 10})
            for c in cmds:
                run_dir = c.argv[c.argv.index("--run-dir") + 1] if "--run-dir" in c.argv else \
                    c.argv[c.argv.index("--embed-dir") + 1]
                assert Path(run_dir).is_relative_to(Path(d) / "_smoke")
                copy = Path(c.argv[c.argv.index("--data") + 1])
                assert len(copy.read_text(encoding="utf-8").splitlines()) == 10
        finally:
            jobs.data_path = real_data
            paths.RUNS_DIR = real_runs


def test_full_builds_only_missing_datasets():
    real = jobs.data_path
    try:
        jobs.data_path = lambda p: Path("/nope") / p          # nothing on disk
        scripts = [_script(c) for c in build_cmds("full", {"models": "minilm"})]
        assert scripts[:3] == ["download_dataset.py", "organic_dataset.py", "generate_embeddings.py"]
        jobs.data_path = lambda p: Path(__file__)             # everything on disk
        assert len(build_cmds("full", {"models": "minilm"})) == 18
        rebuilt = build_cmds("full", {"models": "minilm", "rebuild_data": True, "skip_done": True})
        # new datasets make cached embeddings stale, so a rebuild always re-embeds
        assert len(rebuilt) == 20 and all("--force" in c.argv for c in rebuilt if _script(c) == "generate_embeddings.py")
    finally:
        jobs.data_path = real


def test_resources_read_from_the_runs():
    real = paths.RUNS_DIR
    with tempfile.TemporaryDirectory() as d:
        try:
            paths.RUNS_DIR = Path(d)
            run = Path(d) / "pipeline1_alamsabi" / "normal"
            for sub in ("results", "embeddings", "models"):
                (run / sub).mkdir(parents=True)
            (run / "results" / "full_evaluation_results.csv").write_text(
                "Embeddings,Classifier,Train-Time(s),Train-CPU(s),Train-RAM(MB),Latency-P50(ms),Latency-P99(ms)\n"
                "MINILM,SVM,12.5,3.0,210.0,0.7,1.1\n"
                "OPENAI,SVM,15.0,4.0,,0.8,1.2\n", encoding="utf-8")
            (run / "results" / "run_meta.json").write_text(json.dumps(
                {"environment": {"cpu": "Test CPU", "total_ram_gb": 64}, "finished_at": "2026-11-07T23:00:00"}))
            np.save(run / "embeddings" / "minilm_prompt.npy", np.zeros((10, 4), np.float32))
            (run / "embeddings" / "minilm_gpu.json").write_text(json.dumps(
                {"device": "Test GPU", "vram_peak_mb": 1017.0, "gpu_util_mean": 47.8, "gpu_util_max": 99.0}))
            (run / "models" / "minilm_SVM.joblib").write_bytes(b"x" * 2_000_000)

            res = runs.resources()
            mini, oai = res["rows"]
            assert (mini["fit_s"], mini["cpu_s"], mini["ram_mb"], mini["p99_ms"]) == (12.5, 3.0, 210.0, 1.1)
            assert (mini["vram_mb"], mini["gpu_mean"], mini["model_mb"], mini["emb_rows"]) == (1017.0, 47.8, 2.0, 10)
            assert oai["ram_mb"] is None and oai["vram_mb"] is None and oai["model_mb"] is None
            assert res["machine"] == {"cpu": "Test CPU", "ram_gb": 64, "gpu": "Test GPU"}
            assert (res["n_embedding"], res["n_classifier"], res["n_benchmark"]) == (1, 2, 0)

            # Measure resources ran later: its numbers win, and its blanks are filled from the run
            (Path(d) / runs.BENCH_FILE).write_text(json.dumps({"classifier": {
                "pipeline1_alamsabi|minilm|SVM": {"fit_s": 9.0, "cpu_s": 8.0, "measured_at": "2026-11-08T01:00:00"},
                "pipeline1_alamsabi|openai|SVM": {"fit_s": 1.0, "ram_mb": 99.0, "measured_at": "2026-11-01T00:00:00"},
            }}))
            mini, oai = runs.resources()["rows"]
            assert (mini["fit_s"], mini["cpu_s"], mini["ram_mb"], mini["clf_src"]) == (9.0, 8.0, 210.0, "benchmark")
            # an older benchmark loses to the run, but still fills what the run lacks
            assert (oai["fit_s"], oai["ram_mb"], oai["clf_src"]) == (15.0, 99.0, "run")
        finally:
            paths.RUNS_DIR = real


def test_resources_one_step_per_dataset_on_disk():
    real = jobs.data_path
    try:
        jobs.data_path = lambda p: Path(__file__) if p != "pipeline2_organic_bipia" else Path("/nope")
        cmds = build_cmds("resources", {"models": "minilm,qwen3", "stages": "classify"})
        assert [c.argv[c.argv.index("--pipelines") + 1] for c in cmds] == \
            ["pipeline1_alamsabi", "pipeline3_organic_injected"]
        assert all(_script(c) == "benchmark_resources.py" and c.run for c in cmds)
        assert all(c.argv[-4:] == ["--stages", "classify", "--models", "minilm,qwen3"] for c in cmds)
        for bad in ({"stages": "embed,nope"},):
            try:
                build_cmds("resources", bad)
            except jobs.ValidationError:
                continue
            raise AssertionError(f"accepted {bad}")
        jobs.data_path = lambda p: Path("/nope")
        try:
            build_cmds("resources", {})
        except jobs.ValidationError:
            pass
        else:
            raise AssertionError("queued a benchmark with no dataset on disk")
    finally:
        jobs.data_path = real


def test_failed_step_skips_only_its_own_run():
    py = sys.executable
    with tempfile.TemporaryDirectory() as d:
        job = Job(id="fail-test", stage="run", params={}, logs_root=Path(d), cmds=[
            Cmd([py, "-c", "raise SystemExit(3)"], "A embed", run="A"),
            Cmd([py, "-c", "print('A train ran')"], "A train", run="A"),
            Cmd([py, "-c", "print('B ran')"], "B embed", run="B"),
        ])
        jobs.QUEUE.add(job)
        for _ in range(300):
            if job.finished:
                break
            time.sleep(0.1)
        log = job.log_path.read_text(encoding="utf-8")
        assert job.status == "failed" and job.returncode == 3
        assert "A train ran" not in log and "skipped: an earlier step of A failed" in log
        assert "B ran" in log and job.step == 3


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
