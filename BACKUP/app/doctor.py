"""Print which data and runs folders the dashboard reads and what each run has. Run: python -m app.doctor"""

from __future__ import annotations

from app import paths, runs


def main() -> None:
    d = paths.describe()
    print("repo root :", d["root"])
    print("datasets  :", d["data_dir"], "" if d["data_exists"] else "(MISSING)")
    print("runs      :", d["runs_dir"], "" if d["runs_exists"] else "(MISSING)")

    state = runs.scan_state()
    print("\ndatasets")
    for ds in state["datasets"].values():
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
    print(f"\nruns with results: {cap['n_runs_with_results']}/9 · "
          f"with predictions: {cap['n_runs_with_predictions']}/9 · "
          f"with models: {cap['n_runs_with_models']}/9")
    if not cap["n_runs_with_models"]:
        print("  -> no saved classifiers: the Simulation tab stays disabled until a run is trained")
    if not d["runs_exists"] or not cap["n_runs_with_results"]:
        print("\nNothing found. Put data/ and runs/ in the repo root, or point at them with "
              "THESIS_DATA_DIR and THESIS_RUNS_DIR.")


if __name__ == "__main__":
    main()
