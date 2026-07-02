# Throwaway overnight sweep aggregator (#555): summarize the workers' JSONL records.
import glob
import json
import statistics
import sys
from collections import Counter


def main():
    paths = sys.argv[1:] or glob.glob(
        "/home/andrew/feat-proton-ingredients/.overnight/worker_*.jsonl")
    paths = [p for p in paths if ".progress." not in p]   # verdicts only, not the trend stream
    records = []
    for path in sorted(paths):
        with open(path) as f:
            records += [json.loads(line) for line in f if line.strip()]
    ok = [r for r in records if "error" not in r]
    errors = [r for r in records if "error" in r]
    print(f"attempts: {len(records)}  (clean {len(ok)}, errors {len(errors)})")
    if errors:
        print(f"  first error: {errors[0]['error']!r} (worker {errors[0]['worker']})")
    if not ok:
        return
    converged = [r for r in ok if r["converged"]]
    stationary = [r for r in ok if r["stationary"]]
    persistent = [r for r in ok if r["persistent"]]
    print(f"converged (stationary AND persistent): {len(converged)}")
    for r in converged[:20]:
        print(f"  seed {r['base_seed']}: holes={r['holes']} betti={r['betti']} "
              f"singlet={r['singlet']:.4g} F={r['F']:.4g} edges={r['edges']}")
    print(f"stationary: {len(stationary)}/{len(ok)}   persistent: "
          f"{len(persistent)}/{len(ok)}")
    print(f"holes histogram: {dict(sorted(Counter(r['holes'] for r in ok).items()))}")
    print(f"b3 histogram:    "
          f"{dict(sorted(Counter(r['betti'][3] for r in ok if len(r['betti']) > 3).items()))}")
    singlets = [r["singlet"] for r in ok]
    print(f"singlet diagnostic: min={min(singlets):.4g}  "
          f"median={statistics.median(singlets):.4g}  max={max(singlets):.4g}")
    fs = [r["F"] for r in ok]
    print(f"final F: min={min(fs):.4g}  median={statistics.median(fs):.4g}  "
          f"max={max(fs):.4g}")
    print(f"cells: median={statistics.median(r['cells'] for r in ok):.0f}  "
          f"max={max(r['cells'] for r in ok)}   edges: "
          f"median={statistics.median(r['edges'] for r in ok):.0f}  "
          f"max={max(r['edges'] for r in ok)}")
    causal = [r for r in ok if r["im_max"] > 0 or r["re_min"] < 0]
    print(f"attempts with ANY causal content (Im≠0 or Re<0): {len(causal)}")
    best = min(ok, key=lambda r: r["singlet"])
    print(f"lowest singlet diagnostic: seed {best['base_seed']} -> "
          f"singlet={best['singlet']:.4g} holes={best['holes']} betti={best['betti']} "
          f"F={best['F']:.4g} stationary={best['stationary']} "
          f"persistent={best['persistent']}")
    elapsed = [r["elapsed_s"] for r in ok]
    print(f"attempt time: median={statistics.median(elapsed):.0f}s  "
          f"max={max(elapsed):.0f}s")


if __name__ == "__main__":
    main()
