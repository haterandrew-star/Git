"""
detect_wave_pattern.py
----------------------
Automatically determines whether a tournament registration curve is:

  1-wave  — single logistic sigmoid (no early bird discount)
  2-wave  — double sigmoid (early-bird spike + late-registration surge)

Uses four signals weighted into a score:
  1. Fee deadline structure  — early bird >45 days before event: +3 / >21d: +1
  2. Registration velocity   — plateau between two surge peaks: +2
  3. L1 fraction             — early phase ≥8% of asymptote: +2 / ≥4%: +1
  4. AIC model comparison    — double sigmoid wins by >10: +3 / >4: +1 / single wins: -2
  5. Historical prior        — optional empirical base-rate from fitted_params.json

score ≥ 3  →  2-wave (double sigmoid recommended)
score < 3  →  1-wave (single sigmoid recommended)

Usage:
    python tools/detect_wave_pattern.py \\
        --data .tmp/cumulative_WO2026.json \\
        --config-id WO2026 \\
        --historical .tmp/all_fitted_params.json

    python tools/detect_wave_pattern.py \\
        --data .tmp/cumulative_CHI2022.json \\
        --deadline-date 2022-03-15 \\
        --event-date 2022-05-27

Output:
    {
        "wave_count": 2,
        "model": "double_sigmoid",
        "confidence": 0.87,
        "score": 8,
        "l1_fraction": 0.217,
        "signals": { ... },
        "reasoning": "Early Bird 73d before event (+3); Velocity plateau (+2); ...",
        "description": "2-Wave (Early Bird): 87% confidence"
    }
"""

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parent.parent


# ── Model functions ────────────────────────────────────────────────────────────

def single_sigmoid(t: float, L: float, k: float, m: float) -> float:
    x = -k * (t - m)
    if x > 500: return 0.0
    if x < -500: return L
    return L / (1.0 + math.exp(x))


def double_sigmoid(t: float, L1: float, k1: float, m1: float,
                   L2: float, k2: float, m2: float) -> float:
    def sig(k, t, m):
        x = -k * (t - m)
        if x > 500: return 0.0
        if x < -500: return 1.0
        return 1.0 / (1.0 + math.exp(x))
    return L1 * sig(k1, t, m1) + L2 * sig(k2, t, m2)


# ── Fitting ────────────────────────────────────────────────────────────────────

def _fit_single(data: list) -> dict | None:
    """Fit single sigmoid via scipy curve_fit, returns params + ssr."""
    try:
        import numpy as np
        from scipy.optimize import curve_fit
    except ImportError:
        return None

    ts = np.array([d[0] for d in data], dtype=float)
    ys = np.array([d[1] for d in data], dtype=float)
    y_max, t_max = float(ys.max()), float(ts.max())

    def model(t, L, k, m):
        return np.array([single_sigmoid(float(ti), L, k, m) for ti in t])

    best, best_r2 = None, -999.0
    for p0 in [
        [y_max * 1.15, 0.05, t_max * 0.65],
        [y_max * 1.30, 0.03, t_max * 0.75],
        [y_max * 1.05, 0.10, t_max * 0.55],
    ]:
        try:
            popt, _ = curve_fit(
                model, ts, ys, p0=p0,
                bounds=([1, 0.001, 0], [y_max * 10, 2, t_max * 3]),
                maxfev=15000, method="trf",
            )
            L, k, m = popt
            preds = model(ts, L, k, m)
            ssr = float(np.sum((ys - preds) ** 2))
            ss_tot = float(np.sum((ys - ys.mean()) ** 2))
            r2 = 1.0 - ssr / ss_tot if ss_tot > 0 else 0.0
            if r2 > best_r2:
                best_r2 = r2
                best = {
                    "L": round(float(L), 2), "k": round(float(k), 6),
                    "m": round(float(m), 2), "r2": round(r2, 4),
                    "ssr": round(ssr, 2), "n": len(data),
                }
        except Exception:
            continue
    return best


def _fit_double(data: list) -> dict | None:
    """Fit double sigmoid via scipy curve_fit, returns params + ssr."""
    try:
        import numpy as np
        from scipy.optimize import curve_fit
    except ImportError:
        return None

    ts = np.array([d[0] for d in data], dtype=float)
    ys = np.array([d[1] for d in data], dtype=float)
    y_max, t_max = float(ys.max()), float(ts.max())

    def t_frac(frac):
        target = frac * y_max
        for i, y in enumerate(ys):
            if y >= target:
                return float(ts[i])
        return t_max * frac

    def model(t, L1, k1, m1, L2, k2, m2):
        return np.array([double_sigmoid(float(ti), L1, k1, m1, L2, k2, m2) for ti in t])

    best, best_r2 = None, -999.0
    for p0 in [
        [y_max * 0.12, 0.05, t_frac(0.25), y_max * 0.88, 0.25, t_frac(0.75)],
        [y_max * 0.20, 0.03, t_max * 0.20, y_max * 0.80, 0.15, t_max * 0.80],
        [y_max * 0.05, 0.10, t_max * 0.25, y_max * 0.95, 0.30, t_max * 0.85],
    ]:
        try:
            popt, _ = curve_fit(
                model, ts, ys, p0=p0,
                bounds=([0, 0.001, 0, 0, 0.001, 0],
                        [y_max * 3, 2, t_max * 2, y_max * 5, 2, t_max * 3]),
                maxfev=30000, method="trf",
            )
            L1, k1, m1, L2, k2, m2 = popt
            if m1 >= m2:
                continue
            preds = model(ts, L1, k1, m1, L2, k2, m2)
            ssr = float(np.sum((ys - preds) ** 2))
            ss_tot = float(np.sum((ys - ys.mean()) ** 2))
            r2 = 1.0 - ssr / ss_tot if ss_tot > 0 else 0.0
            if r2 > best_r2:
                best_r2 = r2
                best = {
                    "L1": round(float(L1), 4), "k1": round(float(k1), 6),
                    "m1": round(float(m1), 2), "L2": round(float(L2), 4),
                    "k2": round(float(k2), 6), "m2": round(float(m2), 2),
                    "r2": round(r2, 4), "ssr": round(ssr, 2), "n": len(data),
                }
        except Exception:
            continue
    return best


def _aic(n: int, ssr: float, n_params: int) -> float:
    """AIC = n·ln(SSR/n) + 2·p  (lower = better fit)."""
    if ssr <= 0 or n <= 0:
        return 0.0
    return n * math.log(ssr / n) + 2 * n_params


# ── Signal: velocity plateau ───────────────────────────────────────────────────

def detect_velocity_plateau(data: list) -> bool:
    """
    Returns True if registration velocity shows two peaks separated by a dip —
    characteristic of an early-bird rush followed by late-registration surge.
    """
    if len(data) < 10:
        return False

    vels = []
    for i in range(2, len(data) - 2):
        dt = data[i + 2][0] - data[i - 2][0]
        dy = data[i + 2][1] - data[i - 2][1]
        if dt > 0:
            vels.append(dy / dt)

    if len(vels) < 6:
        return False

    max_v = max(vels)
    if max_v <= 0:
        return False

    # First significant peak (>50% of max velocity)
    peak1_idx = next((i for i, v in enumerate(vels) if v > max_v * 0.50), None)
    if peak1_idx is None:
        return False

    after = vels[peak1_idx + 1:]
    if len(after) < 4:
        return False

    mid = len(after) // 2
    dip_region = after[:mid]
    late_region = after[mid:]

    min_dip = min(dip_region) if dip_region else max_v
    max_late = max(late_region) if late_region else 0

    # Plateau = dip below 40% of max AND second surge above 25%
    return min_dip < max_v * 0.40 and max_late > max_v * 0.25


# ── Main detection function ────────────────────────────────────────────────────

def detect_wave_pattern(
    data: list,
    metadata: dict | None = None,
    fitted_double_params: dict | None = None,
    historical_params: list | None = None,
) -> dict:
    """
    Classify tournament registration as 1-wave or 2-wave.

    Args:
        data:                 [[t, count], ...] cumulative registration series
        metadata:             {fee_deadlines, tournament_date, first_reg_date, expected_total}
        fitted_double_params: Pre-fitted double sigmoid params {L1, k1, m1, L2, k2, m2, ...}
                              Pass this to skip re-fitting from scratch when params are available.
        historical_params:    [{params: {L1, L2, ...}, ...}] from all_fitted_params.json

    Returns:
        {wave_count, model, confidence, score, l1_fraction, signals, reasoning, description}
    """
    meta = metadata or {}
    signals: dict = {}
    score = 0
    reasons: list[str] = []

    # ── Signal 1: Fee Deadline Structure ──────────────────────────────────────
    fee_deadlines = meta.get("fee_deadlines", [])
    t_date_str = meta.get("tournament_date")
    early_bird = None

    if fee_deadlines:
        try:
            early_bird = sorted(fee_deadlines, key=lambda d: d.get("date", "9999"))[0]
        except (TypeError, KeyError):
            early_bird = fee_deadlines[0] if fee_deadlines else None

    if early_bird and t_date_str:
        try:
            eb_date = date.fromisoformat(str(early_bird.get("date", "")))
            ev_date = date.fromisoformat(t_date_str)
            days_before = (ev_date - eb_date).days
            signals["has_fee_deadline"] = days_before > 21
            signals["deadline_days_before_event"] = days_before
            if days_before > 45:
                score += 3
                reasons.append(f"Early Bird deadline {days_before}d before event (+3)")
            elif days_before > 21:
                score += 1
                reasons.append(f"Fee deadline {days_before}d before event (+1)")
            else:
                reasons.append(f"Late fee deadline only {days_before}d before event (0)")
        except (ValueError, AttributeError):
            signals["has_fee_deadline"] = False
    else:
        signals["has_fee_deadline"] = False
        score -= 1
        reasons.append("No early fee deadline structure (-1)")

    # ── Signal 2: Velocity Plateau ─────────────────────────────────────────────
    plateau = detect_velocity_plateau(data)
    signals["velocity_plateau_detected"] = plateau
    if plateau:
        score += 2
        reasons.append("Registration velocity plateau detected (+2)")

    # ── Signal 3: L1 Fraction ─────────────────────────────────────────────────
    l1_frac = 0.0
    ds_params = fitted_double_params

    if not ds_params and len(data) >= 8:
        ds_params = _fit_double(data)

    if ds_params:
        total = ds_params.get("L1", 0) + ds_params.get("L2", 0)
        l1_frac = ds_params["L1"] / total if total > 0 else 0.0
        signals["l1_fraction"] = round(l1_frac, 3)
        signals["double_r2"] = ds_params.get("r2")

        if l1_frac > 0.08:
            score += 2
            reasons.append(f"L1 = {l1_frac * 100:.1f}% of total (early phase significant, +2)")
        elif l1_frac > 0.04:
            score += 1
            reasons.append(f"L1 = {l1_frac * 100:.1f}% of total (+1)")
        else:
            score -= 1
            reasons.append(f"L1 = {l1_frac * 100:.1f}% (weak early phase, -1)")

    # ── Signal 4: AIC Comparison ───────────────────────────────────────────────
    if len(data) >= 8:
        ss = _fit_single(data)
        if ss and ds_params and "ssr" in ds_params:
            aic_s = _aic(len(data), ss["ssr"], 3)
            aic_d = _aic(len(data), ds_params["ssr"], 6)
            delta = aic_s - aic_d  # positive → double sigmoid fits better

            signals["aic_single"] = round(aic_s, 1)
            signals["aic_double"] = round(aic_d, 1)
            signals["aic_delta"] = round(delta, 1)
            signals["aic_winner"] = (
                "double" if delta > 2 else "single" if delta < -2 else "tie"
            )
            signals["single_r2"] = ss.get("r2")

            if delta > 10:
                score += 3
                reasons.append(f"AIC strongly favors double sigmoid (Δ={delta:.1f}, +3)")
            elif delta > 4:
                score += 1
                reasons.append(f"AIC slightly favors double sigmoid (Δ={delta:.1f}, +1)")
            elif delta < -4:
                score -= 2
                reasons.append(f"AIC favors single sigmoid (Δ={delta:.1f}, -2)")

    # ── Signal 5: Historical Prior ─────────────────────────────────────────────
    if historical_params and len(historical_params) >= 3:
        two_wave = sum(
            1 for p in historical_params
            if (p.get("params", {}).get("L1", 0) /
                max(1, p.get("params", {}).get("L1", 0) + p.get("params", {}).get("L2", 1))) > 0.08
        )
        base_rate = two_wave / len(historical_params)
        signals["historical_two_wave_rate"] = round(base_rate, 2)
        signals["historical_sample_size"] = len(historical_params)

        if base_rate > 0.55:
            score += 1
            reasons.append(
                f"Historical prior: {base_rate * 100:.0f}% of tournaments are 2-wave (+1)"
            )
        elif base_rate < 0.30:
            score -= 1
            reasons.append(
                f"Historical prior: only {base_rate * 100:.0f}% are 2-wave (-1)"
            )

    # ── Classification ─────────────────────────────────────────────────────────
    wave_count = 2 if score >= 3 else 1
    confidence = min(0.99, max(0.50, 0.50 + abs(score) * 0.07))
    model = "double_sigmoid" if wave_count == 2 else "single_sigmoid"

    return {
        "wave_count": wave_count,
        "model": model,
        "confidence": round(confidence, 2),
        "score": score,
        "l1_fraction": round(l1_frac, 3),
        "signals": signals,
        "reasoning": "; ".join(reasons) if reasons else "No signals available",
        "description": (
            f"{'2-Wave (Early Bird)' if wave_count == 2 else '1-Wave (No Early Bird)'}: "
            f"{round(confidence * 100)}% confidence"
        ),
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def _load_config(tid: str) -> dict | None:
    config_path = ROOT / "config.json"
    if not config_path.exists():
        return None
    with open(config_path) as f:
        cfg = json.load(f)
    for t in cfg.get("tournaments", []):
        if t["id"] == tid:
            return t
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Detect 1-wave vs 2-wave registration pattern"
    )
    parser.add_argument("--data", required=True,
                        help="Path to cumulative JSON [[t, count], ...]")
    parser.add_argument("--config-id",
                        help="Tournament ID to look up metadata in config.json")
    parser.add_argument("--deadline-date",
                        help="Early bird deadline date YYYY-MM-DD")
    parser.add_argument("--event-date",
                        help="Tournament start date YYYY-MM-DD")
    parser.add_argument("--historical",
                        help="Path to all_fitted_params.json for historical prior")
    parser.add_argument("--out",
                        help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    # Load registration data
    with open(args.data) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("data", data.get("points", []))

    # Build metadata from config or CLI args
    meta: dict = {}
    if args.config_id:
        cfg = _load_config(args.config_id)
        if cfg:
            meta["fee_deadlines"] = cfg.get("fee_deadlines", [])
            meta["tournament_date"] = cfg.get("tournament_date")
            meta["first_reg_date"] = cfg.get("first_reg_date")
            meta["expected_total"] = cfg.get("expected_total")

    if args.deadline_date:
        meta.setdefault("fee_deadlines", []).append(
            {"date": args.deadline_date, "label": "Early Bird"}
        )
    if args.event_date:
        meta["tournament_date"] = args.event_date

    # Load historical priors
    hist = None
    if args.historical and Path(args.historical).exists():
        with open(args.historical) as f:
            hist_raw = json.load(f)
        # all_fitted_params.json has shape {generated_at, count, tournaments: [...]}
        # detect_wave_pattern expects a flat list of param dicts
        if isinstance(hist_raw, dict):
            hist = hist_raw.get("tournaments", [])
        else:
            hist = hist_raw

    result = detect_wave_pattern(data, metadata=meta, historical_params=hist)

    print(f"\nWave Pattern Detection: {args.data}", file=sys.stderr)
    print(f"  Result:     {result['description']}", file=sys.stderr)
    print(f"  Score:      {result['score']:+d}", file=sys.stderr)
    print(f"  L1 fraction:{result['l1_fraction']:.3f}", file=sys.stderr)
    print(f"  Reasoning:  {result['reasoning']}", file=sys.stderr)

    output = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            f.write(output)
        print(f"  Saved → {args.out}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
