"""
fit_double_sigmoid.py
---------------------
Fits a double-sigmoid model to cumulative tournament registration data.

Model:
    f(t) = L1 / (1 + exp(-k1*(t-m1))) + L2 / (1 + exp(-k2*(t-m2)))

Where:
    L1, L2 = asymptotic heights of each sigmoid phase
    k1, k2 = growth rates (steepness)
    m1, m2 = midpoints (inflection points)

Usage:
    python tools/fit_double_sigmoid.py \\
        --data .tmp/cumulative_world_open.json \\
        --out .tmp/params_world_open.json

    # Or pipe data as JSON array of [t, count] pairs:
    echo '[[0,1],[10,3],[30,8]]' | python tools/fit_double_sigmoid.py --stdin --out params.json

Input format (.json):
    [[t0, count0], [t1, count1], ...]
    or {"data": [[t0, count0], ...]}

Output format:
    {
        "L1": float, "k1": float, "m1": float,
        "L2": float, "k2": float, "m2": float,
        "r2": float, "rmse": float,
        "n_points": int,
        "converged": bool
    }
"""

import argparse
import json
import math
import sys
from pathlib import Path


# ── Model function ────────────────────────────────────────────────────────────

def double_sigmoid(t, L1, k1, m1, L2, k2, m2):
    """Evaluate the double-sigmoid at time t."""
    def safe_sigmoid(k, t, m):
        x = -k * (t - m)
        if x > 500:
            return 0.0
        if x < -500:
            return 1.0
        return 1.0 / (1.0 + math.exp(x))

    return L1 * safe_sigmoid(k1, t, m1) + L2 * safe_sigmoid(k2, t, m2)


# ── Fitting ───────────────────────────────────────────────────────────────────

def fit_scipy(data: list[list[float]]) -> dict:
    """
    Fit using scipy.optimize.curve_fit (Levenberg-Marquardt).
    Preferred method — more robust than gradient descent.
    """
    try:
        import numpy as np
        from scipy.optimize import curve_fit
    except ImportError:
        raise ImportError("scipy and numpy are required. Run: pip install scipy numpy")

    ts = np.array([d[0] for d in data], dtype=float)
    ys = np.array([d[1] for d in data], dtype=float)

    y_max = float(ys.max())
    t_max = float(ts.max())

    # Initial guess: split the total into two phases
    # m1 ~ where 25% of max is reached, m2 ~ 75%
    def t_at_frac(frac):
        target = frac * y_max
        for i, y in enumerate(ys):
            if y >= target:
                return ts[i]
        return t_max * frac

    m1_guess = t_at_frac(0.25)
    m2_guess = t_at_frac(0.75)

    p0 = [
        y_max * 0.12,  # L1 — small first phase
        0.05,          # k1 — slow early growth
        m1_guess,      # m1
        y_max * 0.88,  # L2 — large second phase
        0.25,          # k2 — faster late surge
        m2_guess,      # m2
    ]

    bounds = (
        [0,    0.001, 0,       0,    0.001, 0],
        [y_max * 2, 2, t_max * 2, y_max * 5, 2, t_max * 3],
    )

    def model(t, L1, k1, m1, L2, k2, m2):
        return np.array([double_sigmoid(ti, L1, k1, m1, L2, k2, m2) for ti in t])

    try:
        popt, _ = curve_fit(
            model, ts, ys,
            p0=p0,
            bounds=bounds,
            maxfev=50000,
            method="trf",
        )
        L1, k1, m1, L2, k2, m2 = popt
        converged = True
    except RuntimeError:
        # Fallback: return initial guess as params
        L1, k1, m1, L2, k2, m2 = p0
        converged = False

    # Compute fit quality
    preds = np.array([double_sigmoid(t, L1, k1, m1, L2, k2, m2) for t in ts])
    ss_res = float(np.sum((ys - preds) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(ss_res / len(ys)))

    return {
        "L1": round(L1, 4), "k1": round(k1, 6), "m1": round(m1, 2),
        "L2": round(L2, 4), "k2": round(k2, 6), "m2": round(m2, 2),
        "r2": round(r2, 4), "rmse": round(rmse, 3),
        "n_points": len(data),
        "converged": converged,
    }


def fit_gradient_descent(data: list[list[float]]) -> dict:
    """
    Pure-Python gradient descent fallback (no scipy required).
    Less accurate but works without dependencies.
    """
    ts = [d[0] for d in data]
    ys = [d[1] for d in data]
    n = len(data)
    y_max = max(ys)
    t_max = max(ts)

    # Initial params
    L1, k1, m1 = y_max * 0.12, 0.05, t_max * 0.3
    L2, k2, m2 = y_max * 0.88, 0.25, t_max * 0.75

    def loss():
        return sum(
            (double_sigmoid(ts[i], L1, k1, m1, L2, k2, m2) - ys[i]) ** 2
            for i in range(n)
        ) / n

    eps = 1e-4
    lr = 1e-3
    prev_loss = loss()

    for iteration in range(60000):
        # Numerical gradients
        grads = {}
        for param_name, val in [("L1", L1), ("k1", k1), ("m1", m1),
                                  ("L2", L2), ("k2", k2), ("m2", m2)]:
            params = {"L1": L1, "k1": k1, "m1": m1, "L2": L2, "k2": k2, "m2": m2}
            params[param_name] = val + eps
            l_plus = sum(
                (double_sigmoid(ts[i], **params) - ys[i]) ** 2 for i in range(n)
            ) / n
            params[param_name] = val - eps
            l_minus = sum(
                (double_sigmoid(ts[i], **params) - ys[i]) ** 2 for i in range(n)
            ) / n
            grads[param_name] = (l_plus - l_minus) / (2 * eps)

        L1 = max(0.01, L1 - lr * grads["L1"])
        k1 = max(0.001, k1 - lr * grads["k1"])
        m1 = max(0.0, m1 - lr * grads["m1"])
        L2 = max(0.01, L2 - lr * grads["L2"])
        k2 = max(0.001, k2 - lr * grads["k2"])
        m2 = max(m1 + 1, m2 - lr * grads["m2"])

        # Adaptive learning rate
        curr_loss = loss()
        if curr_loss < prev_loss * 0.9999:
            lr *= 1.05
        else:
            lr *= 0.5
        prev_loss = curr_loss

        if iteration % 10000 == 0 and iteration > 0:
            lr *= 0.1  # decay in stages

        if curr_loss < 1e-8:
            break

    # Compute R²
    preds = [double_sigmoid(ts[i], L1, k1, m1, L2, k2, m2) for i in range(n)]
    y_mean = sum(ys) / n
    ss_res = sum((ys[i] - preds[i]) ** 2 for i in range(n))
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = math.sqrt(ss_res / n)

    return {
        "L1": round(L1, 4), "k1": round(k1, 6), "m1": round(m1, 2),
        "L2": round(L2, 4), "k2": round(k2, 6), "m2": round(m2, 2),
        "r2": round(r2, 4), "rmse": round(rmse, 3),
        "n_points": n,
        "converged": True,  # gradient descent always "converges" to something
    }


def fit(data: list[list[float]]) -> dict:
    """Try scipy first; fall back to gradient descent."""
    try:
        return fit_scipy(data)
    except ImportError:
        print("scipy not available — using gradient descent fallback", file=sys.stderr)
        return fit_gradient_descent(data)


# ── CLI entry point ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Fit double-sigmoid model to registration data")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data", help="Path to JSON file with [[t, count], ...] data")
    group.add_argument("--stdin", action="store_true", help="Read data from stdin")
    parser.add_argument("--out", default=None, help="Output JSON file path (default: stdout)")
    parser.add_argument("--min-points", type=int, default=10,
                        help="Minimum data points required to fit (default: 10)")
    return parser.parse_args()


def load_data(args) -> list[list[float]]:
    if args.stdin:
        raw = sys.stdin.read()
    else:
        with open(args.data) as f:
            raw = f.read()

    parsed = json.loads(raw)

    # Accept both bare array and {"data": [...]} format
    if isinstance(parsed, dict):
        parsed = parsed.get("data", parsed.get("points", []))

    return [[float(row[0]), float(row[1])] for row in parsed]


def main():
    args = parse_args()
    data = load_data(args)

    if len(data) < args.min_points:
        result = {
            "error": f"Insufficient data: {len(data)} points (minimum {args.min_points})",
            "n_points": len(data),
            "converged": False,
        }
    else:
        print(f"Fitting model to {len(data)} data points...", file=sys.stderr)
        result = fit(data)
        print(
            f"Fit complete — R²={result['r2']:.4f}, RMSE={result['rmse']:.3f}, "
            f"converged={result['converged']}",
            file=sys.stderr,
        )

    output = json.dumps(result, indent=2)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            f.write(output)
        print(f"Params saved to {args.out}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
