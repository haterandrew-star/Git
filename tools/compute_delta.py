"""
compute_delta.py
----------------
Computes the delta between actual registration count and model prediction,
and generates a full prediction summary including final-count forecast.

Usage:
    python tools/compute_delta.py \\
        --params .tmp/params_world_open.json \\
        --actual 20 \\
        --t-today 154 \\
        --t-final 247 \\
        --tournament "World Open 2026"

    # Or use date-based inputs (requires first-reg-date):
    python tools/compute_delta.py \\
        --params .tmp/params_world_open.json \\
        --actual 20 \\
        --first-reg-date 2025-09-01 \\
        --tournament-date 2026-07-02 \\
        --tournament "World Open 2026"

Output:
    {
        "tournament": "World Open 2026",
        "as_of": "2026-03-10",
        "actual_count": 20,
        "predicted_now": 19.4,
        "delta_pct": 3.1,
        "status": "GREEN",
        "predicted_final": 277.3,
        "ci_low": 235.7,
        "ci_high": 318.9,
        "params": {...}
    }
"""

import argparse
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path


# ── Model ─────────────────────────────────────────────────────────────────────

def double_sigmoid(t, L1, k1, m1, L2, k2, m2):
    def safe_sig(k, t, m):
        x = -k * (t - m)
        if x > 500: return 0.0
        if x < -500: return 1.0
        return 1.0 / (1.0 + math.exp(x))
    return L1 * safe_sig(k1, t, m1) + L2 * safe_sig(k2, t, m2)


# ── Delta computation ─────────────────────────────────────────────────────────

def compute_delta(actual: int, predicted: float) -> float:
    if predicted == 0:
        return 0.0
    return (actual - predicted) / predicted * 100.0


def get_status(delta_pct: float) -> str:
    if delta_pct > 10:
        return "GREEN"
    elif delta_pct < -10:
        return "RED"
    else:
        return "YELLOW"


def compute_ci(predicted_final: float, rmse: float = None) -> tuple[float, float]:
    """
    Confidence interval for final prediction.
    If RMSE is available, propagate it; otherwise use ±15% heuristic.
    """
    if rmse and rmse > 0:
        # Simple propagation: CI ≈ ±2*RMSE at the plateau
        margin = max(predicted_final * 0.10, 2 * rmse)
    else:
        margin = predicted_final * 0.15
    return round(predicted_final - margin, 1), round(predicted_final + margin, 1)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Compute delta and generate prediction summary")

    parser.add_argument("--params", required=True,
                        help="Path to params JSON file (from fit_double_sigmoid.py)")
    parser.add_argument("--actual", type=int, required=True,
                        help="Current actual registration count")
    parser.add_argument("--tournament", default="Tournament",
                        help="Tournament name for output label")

    # t-value inputs (direct)
    parser.add_argument("--t-today", type=float,
                        help="t-value for today (days since first registration)")
    parser.add_argument("--t-final", type=float,
                        help="t-value for tournament date")

    # Date-based inputs (converted to t)
    parser.add_argument("--first-reg-date",
                        help="Date of first registration (YYYY-MM-DD)")
    parser.add_argument("--tournament-date",
                        help="Tournament start date (YYYY-MM-DD)")

    parser.add_argument("--out", default=None,
                        help="Output JSON file path (default: stdout)")
    return parser.parse_args()


def dates_to_t(first_reg_date_str: str, target_date_str: str) -> float:
    first = date.fromisoformat(first_reg_date_str)
    target = date.fromisoformat(target_date_str)
    return (target - first).days


def main():
    args = parse_args()

    # Load params
    with open(args.params) as f:
        params = json.load(f)

    if "error" in params:
        print(f"ERROR: Params file contains error: {params['error']}", file=sys.stderr)
        sys.exit(1)

    L1 = params["L1"]
    k1 = params["k1"]
    m1 = params["m1"]
    L2 = params["L2"]
    k2 = params["k2"]
    m2 = params["m2"]
    rmse = params.get("rmse")

    # Resolve t values
    today_str = date.today().isoformat()

    if args.t_today is not None:
        t_today = args.t_today
    elif args.first_reg_date:
        t_today = dates_to_t(args.first_reg_date, today_str)
    else:
        print("ERROR: Provide either --t-today or --first-reg-date", file=sys.stderr)
        sys.exit(1)

    if args.t_final is not None:
        t_final = args.t_final
    elif args.first_reg_date and args.tournament_date:
        t_final = dates_to_t(args.first_reg_date, args.tournament_date)
    else:
        print("ERROR: Provide either --t-final or (--first-reg-date + --tournament-date)", file=sys.stderr)
        sys.exit(1)

    # Compute predictions
    predicted_now = double_sigmoid(t_today, L1, k1, m1, L2, k2, m2)
    predicted_final = double_sigmoid(t_final, L1, k1, m1, L2, k2, m2)
    delta_pct = compute_delta(args.actual, predicted_now)
    status = get_status(delta_pct)
    ci_low, ci_high = compute_ci(predicted_final, rmse)

    result = {
        "tournament": args.tournament,
        "as_of": today_str,
        "t_today": round(t_today, 1),
        "t_final": round(t_final, 1),
        "actual_count": args.actual,
        "predicted_now": round(predicted_now, 1),
        "delta_pct": round(delta_pct, 1),
        "status": status,
        "predicted_final": round(predicted_final, 1),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "params": {
            "L1": L1, "k1": k1, "m1": m1,
            "L2": L2, "k2": k2, "m2": m2,
            "r2": params.get("r2"), "rmse": rmse,
        },
    }

    output = json.dumps(result, indent=2)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            f.write(output)
        print(f"Prediction saved to {args.out}", file=sys.stderr)
    else:
        print(output)

    # Human-readable summary to stderr
    print(
        f"\n{'='*50}\n"
        f"  {args.tournament} — as of {today_str}\n"
        f"  Actual:    {args.actual:,}\n"
        f"  Expected:  {predicted_now:.1f}\n"
        f"  Delta:     {delta_pct:+.1f}%  [{status}]\n"
        f"  Final est: {predicted_final:.0f} ({ci_low:.0f}–{ci_high:.0f})\n"
        f"{'='*50}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
