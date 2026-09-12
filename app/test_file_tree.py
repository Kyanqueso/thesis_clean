"""Shape and arithmetic of runs.file_tree(), which the Files tab renders blind.

Run: python -m app.test_file_tree   (or under pytest)
"""

from pathlib import Path

from app import paths, runs

GROUPS = ["Input datasets", "Embeddings", "Results", "Figures", "Models", "Other"]


def test_file_tree():
    groups = runs.file_tree()["groups"]
    assert [g["name"] for g in groups] == GROUPS

    for g in groups:
        # a group either has children or its own rows, never both
        assert bool(g["children"]) != bool(g["entries"]) or not g["n_files"]
        if g["children"]:
            for key in ("n_expected", "n_present", "n_files"):
                assert g[key] == sum(c[key] for c in g["children"]), (g["name"], key)
        for node in g["children"] or [g]:
            assert node["n_present"] <= node["n_expected"]
            for e in node["entries"]:
                assert paths.resolve(e["path"]) is not None, e["path"]
                assert e["label"] and e["status"] in ("present", "missing")
                # the eye is offered only where /api/file/inspect can render something
                assert e["preview"] == (e["status"] == "present"
                                        and Path(e["path"]).suffix.lower() in runs.PREVIEW_EXT)
                assert (e["mtime"] is None) == (e["status"] == "missing")

    by_name = {g["name"]: g for g in groups}
    # the schema the dashboard promises: 3 datasets, 3 pipelines x 3 modes x 3 models
    assert by_name["Input datasets"]["n_expected"] == len(runs.PIPELINES)
    assert by_name["Embeddings"]["n_expected"] == (
        len(runs.PIPELINES) * len(runs.MODES) * len(runs.EMB_MODELS))
    assert by_name["Results"]["n_expected"] == (
        len(runs.PIPELINES) * len(runs.MODES) * len(runs.EXPECTED_RESULTS))


if __name__ == "__main__":
    test_file_tree()
    print("ok")
