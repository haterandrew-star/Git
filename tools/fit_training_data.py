"""
fit_training_data.py
--------------------
Loads the scraped historical tournament dataset and fits a double-sigmoid model
to each tournament with enough timestamped data.

Produces:
  1. Per-tournament fitted parameters
  2. A "meta-parameters" summary — distributions of L1, k1, m1, L2, k2, m2
     across all tournaments (useful for Bayesian priors in the hierarchical model)
  3. A ranked quality report showing best-fit tournaments

Usage:
    python tools/fit_training_data.py \\
        --training .tmp/past_events_training.json \\
        --out .tmp/all_fitted_params.json \\
        --report .tmp/fit_report.json \\
        --min-points 15 \\
        --min-entries 30

Output:
    .tmp/all_fitted_params.json    — [{tournament, tid, params, r2, rmse}, ...]
    .tmp/fit_report.json           — summary stats, prior distributions, quality ranking
"""

import argparse
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path


# ── Model ─────────────────────────────────────────────────────────────────────

def double_sigmoid(t, L1, k1, m1, L2, k2, m2):
    def safe_sig(k, t, m):
        x = -k * (t - m)
        if x > 500: return 0.0
        if x < -500: return 1.0
        return 1.0 / (1.0 + math.exp(x))
    return L1 * safe_sig(k1, t, m1) + L2 * safe_sig(k2, t, m2)


# ── Fitting ───────────────────────────────────────────────────────────────────

def fit_scipy(data: list) -> dict | None:
    try:
        import numpy as np
        from scipy.optimize import curve_fit
    except ImportError:
        return None

    ts = np.array([d[0] for d in data], dtype=float)
    ys = np.array([d[1] for d in data], dtype=float)
    y_max = float(ys.max())
    t_max = float(ts.max())

    def t_at_frac(frac):
        target = frac * y_max
        for i, y in enumerate(ys):
            if y >= target:
                return ts[i]
        return t_max * frac

    # Try multiple initializations to avoid local minima
    init_sets = [
        [y_max * 0.12, 0.05, t_at_frac(0.25), y_max * 0.88, 0.25, t_at_frac(0.75)],
        [y_max * 0.20, 0.03, t_max * 0.20, y_max * 0.80, 0.15, t_max * 0.80],
        [y_max * 0.05, 0.10, t_max * 0.30, y_max * 0.95, 0.35, t_max * 0.85],
    ]

    bounds = (
        [0, 0.001, 0, 0, 0.001, 0],
        [y_max * 3, 2, t_max * 2, y_max * 5, 2, t_max * 3]
    )

    def model(t, L1, k1, m1, L2, k2, m2):
        return np.array([double_sigmoid(ti, L1, k1, m1, L2, k2, m2) for ti in t])

    best = None
    best_r2 = -999

    for p0 in init_sets:
        try:
            popt, _ = curve_fit(model, ts, ys, p0=p0, bounds=bounds,
                                maxfev=50000, method="trf")
            L1, k1, m1, L2, k2, m2 = popt
            preds = np.array([double_sigmoid(t, L1, k1, m1, L2, k2, m2) for t in ts])
            ss_res = float(np.sum((ys - preds) ** 2))
            ss_tot = float(np.sum((ys - ys.mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            rmse = float(np.sqrt(ss_res / len(ys)))

            if r2 > best_r2:
                best_r2 = r2
                best = {
                    "L1": round(L1, 4), "k1": round(k1, 6), "m1": round(m1, 2),
                    "L2": round(L2, 4), "k2": round(k2, 6), "m2": round(m2, 2),
                    "r2": round(r2, 4), "rmse": round(rmse, 3),
                    "n_points": len(data), "converged": True,
                }
        except RuntimeError:
            continue

    return best


def fit_record(record: dict, min_points: int, min_entries: int) -> dict | None:
    """Fit one training record. Returns fitted params dict or None."""
    if record.get("data_points", 0) < min_points:
        return None
    if record.get("total_entries", 0) < min_entries:
        return None

    data = record.get("cumulative", [])
    if not data:
        return None

    result = fit_scipy(data)
    if result is None:
        return None

    # Sanity checks — reject nonsensical fits
    if result["r2"] < 0.5:
        return None  # too poor
    if result["L1"] <= 0 or result["L2"] <= 0:
        return None
    if result["m1"] >= result["m2"]:
        return None  # wrong ordering

    return {
        "tournament": record["tournament"],
        "tid": record["tid"],
        "total_entries": record["total_entries"],
        "data_points": record["data_points"],
        "params": result,
        "r2": result["r2"],
        "rmse": result["rmse"],
        "years": record.get("years", []),
    }


# ── Prior distribution summary ────────────────────────────────────────────────

def compute_priors(fitted: list[dict]) -> dict:
    """
    Compute summary statistics for each parameter across all fitted tournaments.
    These become Bayesian priors for the hierarchical model.
    """
    if not fitted:
        return {}

    def stats(vals):
        if not vals:
            return {}
        vals = sorted(vals)
        mean = statistics.mean(vals)
        try:
            sd = statistics.stdev(vals)
        except statistics.StatisticsError:
            sd = 0.0
        median = statistics.median(vals)
        p10 = vals[max(0, int(len(vals) * 0.10))]
        p90 = vals[min(len(vals) - 1, int(len(vals) * 0.90))]
        return {
            "mean": round(mean, 4),
            "sd": round(sd, 4),
            "median": round(median, 4),
            "p10": round(p10, 4),
            "p90": round(p90, 4),
            "min": round(vals[0], 4),
            "max": round(vals[-1], 4),
            "n": len(vals),
        }

    params = [f["params"] for f in fitted]

    # Normalize L values as fraction of total (L2/total)
    fractions = []
    for p in params:
        total = p["L1"] + p["L2"]
        if total > 0:
            fractions.append(p["L2"] / total)

    return {
        "L1": stats([p["L1"] for p in params]),
        "k1": stats([p["k1"] for p in params]),
        "m1": stats([p["m1"] for p in params]),
        "L2": stats([p["L2"] for p in params]),
        "k2": stats([p["k2"] for p in params]),
        "m2": stats([p["m2"] for p in params]),
        "r2": stats([f["r2"] for f in fitted]),
        "rmse": stats([f["rmse"] for f in fitted]),
        "L2_fraction": stats(fractions),
        "_note": (
            "L2_fraction is L2/(L1+L2) — fraction of total capacity in the late surge. "
            "k1/k2 are growth rates. m1/m2 are midpoints in days-since-first-registration."
        ),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Fit double-sigmoid models to historical tournament data")
    parser.add_argument("--training", default=".tmp/past_events_training.json",
                        help="Path to training data JSON from scrape_past_events.py")
    parser.add_argument("--out", default=".tmp/all_fitted_params.json",
                        help="Output: all fitted parameters")
    parser.add_argument("--report", default=".tmp/fit_report.json",
                        help="Output: summary report with priors")
    parser.add_argument("--min-points", type=int, default=15,
                        help="Minimum data points to attempt fit (default: 15)")
    parser.add_argument("--min-entries", type=int, default=30,
                        help="Minimum total entries to include (default: 30)")
    parser.add_argument("--top", type=int, default=20,
                        help="Number of top fits to display in report (default: 20)")
    return parser.parse_args()


def main():
    args = parse_args()

    training_path = Path(args.training)
    if not training_path.exists():
        print(f"ERROR: Training data not found at {training_path}", file=sys.stderr)
        print("Run first: python tools/scrape_past_events.py --index-only", file=sys.stderr)
        sys.exit(1)

    with open(training_path) as f:
        training = json.load(f)

    records = training.get("records", [])
    print(f"Loaded {len(records)} training records")
    print(f"Fitting models (min_points={args.min_points}, min_entries={args.min_entries})...")

    try:
        import numpy
        import scipy
        print(f"scipy {scipy.__version__} / numpy {numpy.__version__} available")
    except ImportError:
        print("ERROR: scipy required. Run: pip install scipy numpy", file=sys.stderr)
        sys.exit(1)

    fitted = []
    skipped_points = 0
    skipped_entries = 0
    failed = 0

    for i, record in enumerate(records):
        name = record.get("tournament", "?")[:50]
        n_pts = record.get("data_points", 0)
        n_ent = record.get("total_entries", 0)

        if n_pts < args.min_points:
            skipped_points += 1
            continue
        if n_ent < args.min_entries:
            skipped_entries += 1
            continue

        result = fit_record(record, args.min_points, args.min_entries)

        if result:
            fitted.append(result)
            if (i + 1) % 10 == 0 or len(fitted) <= 5:
                print(f"  [{len(fitted)}] {name} — R²={result['r2']:.3f}, "
                      f"RMSE={result['rmse']:.2f}, n={n_pts}")
        else:
            failed += 1

    print(f"\nFitting complete:")
    print(f"  Fitted:          {len(fitted)}")
    print(f"  Skipped (<pts):  {skipped_points}")
    print(f"  Skipped (<ents): {skipped_entries}")
    print(f"  Failed/poor fit: {failed}")

    # Sort by R²
    fitted.sort(key=lambda x: x["r2"], reverse=True)

    # Save all fitted params
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "count": len(fitted),
            "tournaments": fitted,
        }, f, indent=2)
    print(f"\nAll params → {out_path}")

    # Compute priors
    priors = compute_priors(fitted)

    # Top fits
    top = fitted[:args.top]

    # Build report
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "summary": {
            "total_training_records": len(records),
            "tournaments_fitted": len(fitted),
            "skipped_insufficient_points": skipped_points,
            "skipped_insufficient_entries": skipped_entries,
            "failed_fits": failed,
        },
        "prior_distributions": priors,
        "top_fits": top,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report → {report_path}")

    # Print prior summary
    if priors:
        print(f"\n{'='*60}")
        print("EMPIRICAL PRIORS (use for Bayesian model initialization)")
        print(f"{'='*60}")
        for param in ["L1", "k1", "m1", "L2", "k2", "m2"]:
            s = priors.get(param, {})
            if s:
                print(f"  {param}: mean={s['mean']:.4f}  sd={s['sd']:.4f}  "
                      f"[{s['p10']:.4f} – {s['p90']:.4f}]")

        print(f"\n  R²:   mean={priors['r2']['mean']:.3f}  sd={priors['r2']['sd']:.3f}")
        print(f"  RMSE: mean={priors['rmse']['mean']:.2f}  sd={priors['rmse']['sd']:.2f}")
        print(f"\n  L2 fraction (late surge / total): mean={priors['L2_fraction']['mean']:.3f}")

    print(f"\nTop {min(5, len(top))} fits:")
    for t in top[:5]:
        p = t["params"]
        print(f"  {t['tournament'][:45]:45s} R²={t['r2']:.3f} "
              f"L1={p['L1']:.0f} L2={p['L2']:.0f} k2={p['k2']:.3f}")


if __name__ == "__main__":
    main()
