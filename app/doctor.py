"""Print where the dashboard is looking and what it found there.

    venv/Scripts/python.exe -m app.doctor

Run this first on a new machine: it shows which directories were picked, how
they were picked, and which artifacts each run does and does not have — so a
layout the auto-detection got wrong is obvious rather than silent.
"""

from __future__ import annotations

from app import paths, runs


def main() -> None:
    d = paths.describe()
    print("repo root :", d["root"])
    print("datasets  :", d["data_dir"])
    print("            ", d["data_source"], "·", "found" if d["data_exists"] else "MISSING")
    print("runs      :", d["runs_dir"])
    print("            ", d["runs_source"], "·", "found" if d["runs_exists"] else "MISSING")
    if d["split_write"]:
        print("job output:", d["runs_write_dir"])
    print("explorer roots:", ", ".join(sorted(paths.INSPECT_ROOTS)))

    state = runs.scan_state()
    print("\ndatasets")
    for key, ds in state["datasets"].items():
        info = ds["info"]
        print(f"  {'ok ' if info else '-- '} {ds['file']:<34}"
              f"{str(info['size_mb']) + ' MB' if info else 'not found'}")

    order = "/".join(runs.EMB_MODELS)
    print(f"\nruns (E {order}: letter = present, dot = missing · "
          "R results · P predictions · F figures · M models)")
    for r in state["runs"]:
        emb = "".join(m[0].upper() if ok else "." for m, ok in r["embeddings"].items())
        print(f"  {r['pipeline']:<28} {r['mode']:<18} E:{emb} "
              f"R:{'y' if r['has_results'] else '.'} P:{'y' if r['has_predictions'] else '.'} "
              f"F:{len(r['figures']):<2} M:{len(r['models'])}")

    cap = state["capabilities"]
    print(f"\nsweep summary: {'found' if state['has_sweep_summary'] else 'not found'}")
    print(f"runs with results: {cap['n_runs_with_results']}/9 · "
          f"with predictions: {cap['n_runs_with_predictions']}/9 · "
          f"with models: {cap['n_runs_with_models']}/9")
    if not cap["n_runs_with_predictions"]:
        print("  -> no per-sample probabilities: /api/curves and /api/breakdown "
              "report unavailable until a run is redone locally")
    if not cap["n_runs_with_models"]:
        print("  -> no saved classifiers: the Detect tab stays disabled until a run "
              "is redone with save models")
    if not d["runs_exists"] or not cap["n_runs_with_results"]:
        print("\nNothing found. Point the dashboard at the tree explicitly, e.g.:")
        print('  set THESIS_RUNS_DIR=D:/path/to/runs_from_vastai')
        print('  set THESIS_DATA_DIR=D:/path/to/data_from_vastai')


if __name__ == "__main__":
    main()
