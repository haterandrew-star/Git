"""
model_ensemble.py
-----------------
Multi-model ensemble for chess tournament entry prediction.

Implements five model families that run in tandem and are combined into
a single ensemble prediction:

  1. SingleSigmoid    — f(t) = L / (1 + exp(-k*(t-m)))
                        Best for most CCA events (no early bird)

  2. DoubleSigmoid    — f(t) = L1*sig(k1,m1) + L2*sig(k2,m2)
                        Best for early-bird tournaments (two registration phases)

  3. MonteCarlo       — Parametric bootstrap from scipy parameter covariance.
                        Produces percentile bands (p10/p50/p90) rather than a
                        point estimate. Inherits the sigmoid shape.

  4. BayesianMCMC     — Metropolis-Hastings MCMC with empirical priors derived
                        from historical data. More robust early in registration
                        when data is sparse. Uses Nonlinear Bayesian Regression.

  5. ThompsonSampling — Sequential Bayesian update on final count estimate.
                        Maintains a Normal posterior over N_final and updates it
                        each time we observe n_t entries at time t.

Ensemble output is a weighted combination that emphasizes data-rich models when
many observations exist and prior-heavy models when data is sparse.

Usage:
    python tools/model_ensemble.py --id WO2026
    python tools/model_ensemble.py --id CHI2022 --data .tmp/cumulative_CHI2022.json
    python tools/model_ensemble.py --id NAO2024 --data .tmp/cumulative_NAO2024.json
    python tools/model_ensemble.py --compare   # run on all available .tmp/cumulative_*.json
    python tools/model_ensemble.py --backtest  # evaluate on Chicago and NAO historical data

Outputs:
    .tmp/ensemble_{tid}.json    — full ensemble result with all model predictions
"""

import argparse
import json
import math
import random
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
TMP = ROOT / ".tmp"
TMP.mkdir(exist_ok=True)

# ── Shared utilities ───────────────────────────────────────────────────────────

def safe_sigmoid(k: float, t: float, m: float) -> float:
    x = -k * (t - m)
    if x > 500: return 0.0
    if x < -500: return 1.0
    return 1.0 / (1.0 + math.exp(x))


def single_sigmoid(t: float, L: float, k: float, m: float) -> float:
    return L * safe_sigmoid(k, t, m)


def double_sigmoid(t: float, L1: float, k1: float, m1: float,
                   L2: float, k2: float, m2: float) -> float:
    return L1 * safe_sigmoid(k1, t, m1) + L2 * safe_sigmoid(k2, t, m2)


def r_squared(ys_actual, ys_pred) -> float:
    mean_y = sum(ys_actual) / len(ys_actual)
    ss_tot = sum((y - mean_y) ** 2 for y in ys_actual)
    ss_res = sum((a - p) ** 2 for a, p in zip(ys_actual, ys_pred))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def rmse(ys_actual, ys_pred) -> float:
    n = len(ys_actual)
    return math.sqrt(sum((a - p) ** 2 for a, p in zip(ys_actual, ys_pred)) / n)


# ── Model 1: Single Sigmoid ────────────────────────────────────────────────────

class SingleSigmoidModel:
    """
    Simple logistic growth curve. Best for tournaments without an early bird.
    f(t) = L / (1 + exp(-k*(t-m)))
    """

    def __init__(self):
        self.params: dict = {}
        self.fitted = False
        self.r2 = 0.0
        self.rmse_val = 0.0
        self.pcov = None

    def fit(self, data: list[list]) -> bool:
        """Fit to [[t, count], ...] data."""
        try:
            import numpy as np
            from scipy.optimize import curve_fit
        except ImportError:
            print("  scipy/numpy required for SingleSigmoidModel.fit()")
            return False

        if len(data) < 3:
            return False

        ts = np.array([d[0] for d in data], dtype=float)
        ys = np.array([d[1] for d in data], dtype=float)

        y_max = float(ys.max())
        t_max = float(ts.max())

        def model_fn(t, L, k, m):
            return np.array([single_sigmoid(float(ti), float(L), float(k), float(m)) for ti in t])

        best_params = None
        best_r2 = -999.0
        best_pcov = None

        # Multiple initializations
        init_candidates = [
            [y_max * 1.15, 0.05, t_max * 0.65],
            [y_max * 1.30, 0.03, t_max * 0.75],
            [y_max * 1.05, 0.10, t_max * 0.55],
            [y_max * 1.50, 0.02, t_max * 0.80],
        ]

        bounds = ([1, 0.001, 0], [y_max * 10, 2.0, t_max * 3])

        for p0 in init_candidates:
            try:
                popt, pcov = curve_fit(
                    model_fn, ts, ys, p0=p0, bounds=bounds, maxfev=20000, method="trf"
                )
                L, k, m = popt
                preds = model_fn(ts, L, k, m)
                r2 = r_squared(ys.tolist(), preds.tolist())
                if r2 > best_r2:
                    best_r2 = r2
                    best_params = {"L": float(L), "k": float(k), "m": float(m)}
                    best_pcov = pcov
            except Exception:
                continue

        if best_params is None or best_r2 < 0.0:
            return False

        self.params = best_params
        self.fitted = True
        self.pcov = best_pcov

        preds = [single_sigmoid(t, best_params["L"], best_params["k"], best_params["m"])
                 for t, _ in data]
        ys_list = [d[1] for d in data]
        self.r2 = r_squared(ys_list, preds)
        self.rmse_val = rmse(ys_list, preds)
        return True

    def predict(self, t: float) -> float:
        if not self.fitted:
            return 0.0
        return single_sigmoid(t, self.params["L"], self.params["k"], self.params["m"])

    def predict_with_ci(self, t: float, sigma_mult: float = 1.96) -> dict:
        pred = self.predict(t)
        margin = max(pred * 0.12, 2 * self.rmse_val) if self.fitted else pred * 0.15
        return {
            "predicted": round(pred, 1),
            "ci_low": round(max(0, pred - margin), 1),
            "ci_high": round(pred + margin, 1),
        }


# ── Model 2: Double Sigmoid ────────────────────────────────────────────────────

class DoubleSigmoidModel:
    """
    Two-phase sigmoid for tournaments with an early bird discount.
    f(t) = L1*sig(k1,m1) + L2*sig(k2,m2)
    Phase 1 captures the early-bird rush; phase 2 captures the final surge.
    """

    def __init__(self):
        self.params: dict = {}
        self.fitted = False
        self.r2 = 0.0
        self.rmse_val = 0.0

    def fit(self, data: list[list]) -> bool:
        try:
            import numpy as np
            from scipy.optimize import curve_fit
        except ImportError:
            return False

        if len(data) < 6:
            return False

        ts = np.array([d[0] for d in data], dtype=float)
        ys = np.array([d[1] for d in data], dtype=float)
        y_max = float(ys.max())
        t_max = float(ts.max())

        def t_at_frac(frac):
            target = frac * y_max
            for i, y in enumerate(ys):
                if y >= target:
                    return float(ts[i])
            return t_max * frac

        def model_fn(t, L1, k1, m1, L2, k2, m2):
            return np.array([double_sigmoid(float(ti), L1, k1, m1, L2, k2, m2) for ti in t])

        inits = [
            [y_max * 0.12, 0.05, t_at_frac(0.25), y_max * 0.88, 0.25, t_at_frac(0.75)],
            [y_max * 0.20, 0.03, t_max * 0.20, y_max * 0.80, 0.15, t_max * 0.80],
            [y_max * 0.05, 0.10, t_max * 0.30, y_max * 0.95, 0.35, t_max * 0.85],
        ]
        bounds = (
            [0, 0.001, 0, 0, 0.001, 0],
            [y_max * 3, 2, t_max * 2, y_max * 5, 2, t_max * 3]
        )

        best_r2 = -999.0
        best_params = None

        for p0 in inits:
            try:
                popt, _ = curve_fit(model_fn, ts, ys, p0=p0, bounds=bounds,
                                    maxfev=50000, method="trf")
                L1, k1, m1, L2, k2, m2 = popt
                if m1 >= m2:
                    continue
                preds = model_fn(ts, L1, k1, m1, L2, k2, m2)
                r2 = r_squared(ys.tolist(), preds.tolist())
                if r2 > best_r2:
                    best_r2 = r2
                    best_params = {
                        "L1": float(L1), "k1": float(k1), "m1": float(m1),
                        "L2": float(L2), "k2": float(k2), "m2": float(m2),
                    }
            except Exception:
                continue

        if best_params is None or best_r2 < 0.0:
            return False

        self.params = best_params
        self.fitted = True

        ys_list = [d[1] for d in data]
        preds = [double_sigmoid(t, **best_params) for t, _ in data]
        self.r2 = r_squared(ys_list, preds)
        self.rmse_val = rmse(ys_list, preds)
        return True

    def predict(self, t: float) -> float:
        if not self.fitted:
            return 0.0
        return double_sigmoid(t, **self.params)

    def predict_with_ci(self, t: float) -> dict:
        pred = self.predict(t)
        margin = max(pred * 0.10, 2 * self.rmse_val) if self.fitted else pred * 0.15
        return {
            "predicted": round(pred, 1),
            "ci_low": round(max(0, pred - margin), 1),
            "ci_high": round(pred + margin, 1),
        }


# ── Model 3: Monte Carlo Bootstrap ────────────────────────────────────────────

class MonteCarloModel:
    """
    Parametric bootstrap: sample parameter distributions from the fitted
    covariance matrix to generate an ensemble of plausible final counts.

    Produces distribution statistics (p10/p50/p90) rather than a point estimate.
    Works with either single or double sigmoid depending on has_early_bird.
    """

    def __init__(self, has_early_bird: bool = False, n_samples: int = 2000):
        self.has_early_bird = has_early_bird
        self.n_samples = n_samples
        self.base_model: SingleSigmoidModel | DoubleSigmoidModel | None = None
        self.fitted = False
        self._samples_cache: dict = {}

    def fit(self, data: list[list]) -> bool:
        if self.has_early_bird:
            self.base_model = DoubleSigmoidModel()
        else:
            self.base_model = SingleSigmoidModel()

        ok = self.base_model.fit(data)
        self.fitted = ok
        self._samples_cache = {}
        return ok

    def _get_samples(self, t: float) -> list[float]:
        """Sample from parameter posterior and evaluate at t."""
        try:
            import numpy as np
        except ImportError:
            return []

        if not self.fitted or self.base_model is None:
            return []

        cache_key = round(t, 1)
        if cache_key in self._samples_cache:
            return self._samples_cache[cache_key]

        samples = []

        if isinstance(self.base_model, SingleSigmoidModel) and self.base_model.pcov is not None:
            p = self.base_model.params
            mu = [p["L"], p["k"], p["m"]]
            try:
                cov = np.array(self.base_model.pcov)
                # Regularize covariance if needed
                cov = cov + np.eye(3) * 1e-6
                draws = np.random.multivariate_normal(mu, cov, size=self.n_samples)
                for L, k, m in draws:
                    if L > 0 and k > 0:
                        val = single_sigmoid(t, L, k, m)
                        if 0 < val < L * 3:
                            samples.append(val)
            except Exception:
                pass

        # Fallback: perturb parameters by ±20% relative noise
        if len(samples) < self.n_samples // 2:
            params = self.base_model.params
            samples = []
            for _ in range(self.n_samples):
                if isinstance(self.base_model, SingleSigmoidModel):
                    L = params["L"] * (1 + random.gauss(0, 0.12))
                    k = params["k"] * (1 + random.gauss(0, 0.15))
                    m = params["m"] * (1 + random.gauss(0, 0.08))
                    if L > 0 and k > 0:
                        samples.append(single_sigmoid(t, L, k, m))
                else:
                    p = params
                    L1 = p["L1"] * (1 + random.gauss(0, 0.15))
                    k1 = p["k1"] * (1 + random.gauss(0, 0.20))
                    m1 = p["m1"] * (1 + random.gauss(0, 0.10))
                    L2 = p["L2"] * (1 + random.gauss(0, 0.12))
                    k2 = p["k2"] * (1 + random.gauss(0, 0.15))
                    m2 = p["m2"] * (1 + random.gauss(0, 0.08))
                    if L1 > 0 and L2 > 0 and k1 > 0 and k2 > 0 and m1 < m2:
                        samples.append(double_sigmoid(t, L1, k1, m1, L2, k2, m2))

        self._samples_cache[cache_key] = samples
        return samples

    def predict(self, t: float) -> float:
        samples = self._get_samples(t)
        if not samples:
            return self.base_model.predict(t) if self.base_model else 0.0
        return sum(samples) / len(samples)

    def predict_distribution(self, t: float) -> dict:
        """Return full distribution statistics at time t."""
        samples = self._get_samples(t)
        if not samples:
            pt = self.base_model.predict(t) if self.base_model else 0.0
            return {
                "mean": round(pt, 1), "median": round(pt, 1),
                "p10": round(pt * 0.85, 1), "p50": round(pt, 1), "p90": round(pt * 1.15, 1),
                "p5": round(pt * 0.80, 1), "p95": round(pt * 1.20, 1),
                "n_samples": 0,
            }
        samples_sorted = sorted(samples)
        n = len(samples_sorted)

        def pct(p):
            idx = min(int(p / 100 * n), n - 1)
            return round(samples_sorted[idx], 1)

        return {
            "mean": round(sum(samples) / n, 1),
            "median": pct(50),
            "p10": pct(10), "p50": pct(50), "p90": pct(90),
            "p5": pct(5), "p95": pct(95),
            "n_samples": n,
        }


# ── Model 4: Bayesian MCMC (Metropolis-Hastings) ───────────────────────────────

class BayesianMCMCModel:
    """
    Nonlinear Bayesian regression using Metropolis-Hastings MCMC.

    Fits a single sigmoid with a full posterior distribution over L, k, m.
    Priors are loaded from historical fit data when available; otherwise
    uses sensible defaults derived from Chicago 2022 and NAO 2024.

    This model is particularly useful when data is sparse (< 15 points)
    because the prior regularizes the estimate toward historically plausible values.
    """

    def __init__(self, n_iter: int = 4000, burn_in_frac: float = 0.5,
                 priors: dict | None = None):
        self.n_iter = n_iter
        self.burn_frac = burn_in_frac
        self.priors = priors or {}
        self.chain_L: list[float] = []
        self.chain_k: list[float] = []
        self.chain_m: list[float] = []
        self.fitted = False
        self.acceptance_rate = 0.0

    def _log_prior(self, L: float, k: float, m: float, sigma: float,
                   t_max: float) -> float:
        if L <= 0 or k <= 0 or sigma <= 0:
            return -1e18

        # L: Normal prior with mean from priors or data-driven estimate
        L_mean = self.priors.get("L_mean", 500.0)
        L_sd = self.priors.get("L_sd", 400.0)
        lp = -0.5 * ((L - L_mean) / L_sd) ** 2

        # k: Log-Normal prior (log(k) ~ Normal(log(0.05), 1.5))
        k_prior_mean = self.priors.get("k_mean", 0.05)
        lp += -0.5 * ((math.log(k) - math.log(k_prior_mean)) / 1.5) ** 2

        # m: Normal prior centered at ~65% of the time range
        m_mean = self.priors.get("m_mean", t_max * 0.65)
        m_sd = self.priors.get("m_sd", t_max * 0.30)
        lp += -0.5 * ((m - m_mean) / max(m_sd, 10)) ** 2

        # sigma: half-Normal
        lp += -0.5 * (sigma / max(L * 0.15, 20)) ** 2

        return lp

    def _log_likelihood(self, L: float, k: float, m: float, sigma: float,
                        ts: list, ys: list) -> float:
        ll = 0.0
        for t, y in zip(ts, ys):
            y_pred = single_sigmoid(t, L, k, m)
            ll += -0.5 * ((y - y_pred) / sigma) ** 2 - math.log(sigma)
        return ll

    def fit(self, data: list[list]) -> bool:
        if len(data) < 2:
            return False

        ts = [float(d[0]) for d in data]
        ys = [float(d[1]) for d in data]
        t_max = max(ts)
        y_max = max(ys)

        # Initial parameter values
        L = y_max * 1.3
        k = 0.05
        m = t_max * 0.65
        sigma = max(y_max * 0.10, 5.0)

        current_lp = (
            self._log_prior(L, k, m, sigma, t_max)
            + self._log_likelihood(L, k, m, sigma, ts, ys)
        )

        # Adaptive step sizes
        step_L = max(y_max * 0.08, 20.0)
        step_k = k * 0.15
        step_m = t_max * 0.06
        step_s = sigma * 0.20

        chain_L, chain_k, chain_m = [], [], []
        accepted = 0
        burn_in = int(self.n_iter * self.burn_frac)

        for i in range(self.n_iter):
            # Propose new parameters
            L_new = L + random.gauss(0, step_L)
            k_new = k * math.exp(random.gauss(0, 0.15))   # log-scale proposal
            m_new = m + random.gauss(0, step_m)
            s_new = abs(sigma + random.gauss(0, step_s))

            if L_new <= 0 or k_new <= 0 or s_new <= 0:
                continue

            new_lp = (
                self._log_prior(L_new, k_new, m_new, s_new, t_max)
                + self._log_likelihood(L_new, k_new, m_new, s_new, ts, ys)
            )

            log_alpha = new_lp - current_lp
            if log_alpha > 0 or random.random() < math.exp(max(log_alpha, -700)):
                L, k, m, sigma = L_new, k_new, m_new, s_new
                current_lp = new_lp
                accepted += 1

            if i >= burn_in:
                chain_L.append(L)
                chain_k.append(k)
                chain_m.append(m)

            # Adaptive tuning in first quarter of burn-in
            if i > 0 and i < burn_in // 2 and i % 200 == 0:
                rate = accepted / i
                if rate > 0.5:
                    step_L *= 1.3; step_k *= 1.2; step_m *= 1.2
                elif rate < 0.15:
                    step_L *= 0.7; step_k *= 0.8; step_m *= 0.8

        self.chain_L = chain_L
        self.chain_k = chain_k
        self.chain_m = chain_m
        self.fitted = len(chain_L) > 10
        self.acceptance_rate = accepted / self.n_iter
        return self.fitted

    def predict(self, t: float) -> float:
        if not self.fitted:
            return 0.0
        preds = [single_sigmoid(t, L, k, m) for L, k, m in zip(self.chain_L, self.chain_k, self.chain_m)]
        return sum(preds) / len(preds)

    def predict_distribution(self, t: float) -> dict:
        if not self.fitted:
            return {}
        preds = sorted([single_sigmoid(t, L, k, m)
                        for L, k, m in zip(self.chain_L, self.chain_k, self.chain_m)])
        n = len(preds)

        def pct(p):
            return round(preds[min(int(p / 100 * n), n - 1)], 1)

        return {
            "mean": round(sum(preds) / n, 1),
            "median": pct(50),
            "p10": pct(10), "p50": pct(50), "p90": pct(90),
            "p5": pct(5), "p95": pct(95),
            "acceptance_rate": round(self.acceptance_rate, 3),
            "n_posterior_samples": n,
        }

    def get_param_posteriors(self) -> dict:
        """Return posterior statistics for each parameter."""
        if not self.fitted:
            return {}

        def stats(vals):
            if not vals:
                return {}
            s = sorted(vals)
            n = len(s)
            mean = sum(s) / n
            variance = sum((v - mean) ** 2 for v in s) / n
            return {
                "mean": round(mean, 4),
                "sd": round(variance ** 0.5, 4),
                "p10": round(s[max(0, int(0.10 * n))], 4),
                "p50": round(s[int(0.50 * n)], 4),
                "p90": round(s[min(n - 1, int(0.90 * n))], 4),
            }

        return {
            "L": stats(self.chain_L),
            "k": stats(self.chain_k),
            "m": stats(self.chain_m),
        }


# ── Model 5: Thompson Sampling ─────────────────────────────────────────────────

class ThompsonSamplingModel:
    """
    Sequential Bayesian prediction using Thompson Sampling.

    Maintains a Normal posterior over the final count N_final.
    As new registration observations arrive, the posterior is updated
    using a Normal-Normal conjugate model.

    The likelihood is: observed_at_t ~ Normal(f(t/T) * N_final, noise)
    where f(t/T) is the expected fraction registered by time t given the
    sigmoid shape.

    Thompson Sampling: draw N_final ~ Posterior, use as prediction.
    """

    def __init__(self, prior_mean: float = 500.0, prior_std: float = 300.0):
        self.mu_prior = prior_mean
        self.sigma_prior = prior_std
        self.mu_post = prior_mean
        self.sigma_post = prior_std
        self.n_updates = 0
        self.sigmoid_params: dict | None = None

    def set_sigmoid_shape(self, params: dict):
        """
        Provide sigmoid parameters to convert current count → expected final.
        params: dict with L, k, m  (single sigmoid)
             or dict with L1, k1, m1, L2, k2, m2 (double sigmoid)
        """
        self.sigmoid_params = params

    def _fraction_registered(self, t_now: float, t_final: float) -> float:
        """
        Expected fraction of total registered by t_now, given sigmoid shape.
        Returns a value in (0, 1].
        """
        if self.sigmoid_params is None:
            # Fallback: linear interpolation
            return max(0.01, min(1.0, t_now / max(t_final, 1)))

        p = self.sigmoid_params

        if "L" in p:
            # Single sigmoid
            total = single_sigmoid(t_final, p["L"], p["k"], p["m"])
            current = single_sigmoid(t_now, p["L"], p["k"], p["m"])
        elif "L1" in p:
            # Double sigmoid
            total = double_sigmoid(t_final, p["L1"], p["k1"], p["m1"],
                                   p["L2"], p["k2"], p["m2"])
            current = double_sigmoid(t_now, p["L1"], p["k1"], p["m1"],
                                     p["L2"], p["k2"], p["m2"])
        else:
            return t_now / max(t_final, 1)

        if total <= 0:
            return 0.01
        return max(0.01, min(1.0, current / total))

    def update(self, t_now: float, count_now: float, t_final: float):
        """
        Bayesian update using Normal-Normal conjugate model.
        Observing count_now at t_now implies N_final ≈ count_now / frac.
        """
        frac = self._fraction_registered(t_now, t_final)

        # Implied final count from this observation
        implied_final = count_now / frac

        # Observation noise: scale with current count and uncertainty in fraction
        obs_noise = max(30, count_now * 0.15 + (count_now * 0.20 / max(frac, 0.05)))

        # Normal-Normal conjugate update
        prior_precision = 1.0 / (self.sigma_post ** 2)
        likelihood_precision = 1.0 / (obs_noise ** 2)

        self.mu_post = (
            (self.mu_post * prior_precision + implied_final * likelihood_precision)
            / (prior_precision + likelihood_precision)
        )
        self.sigma_post = math.sqrt(1.0 / (prior_precision + likelihood_precision))
        self.n_updates += 1

    def sample(self, n_samples: int = 2000) -> dict:
        """Draw samples from the posterior distribution over N_final."""
        samples = [max(0, random.gauss(self.mu_post, self.sigma_post))
                   for _ in range(n_samples)]
        samples.sort()
        n = len(samples)

        def pct(p):
            return round(samples[min(int(p / 100 * n), n - 1)], 1)

        return {
            "mean": round(sum(samples) / n, 1),
            "median": pct(50),
            "p10": pct(10), "p50": pct(50), "p90": pct(90),
            "posterior_mean": round(self.mu_post, 1),
            "posterior_std": round(self.sigma_post, 1),
            "n_updates": self.n_updates,
            "n_samples": n_samples,
        }

    def predict(self, _t: float = None) -> float:
        return self.mu_post


# ── Model 6: Historical Ratio ──────────────────────────────────────────────────

class HistoricalRatioModel:
    """
    Ratio-based prediction using same-family historical registration curves.

    At the current relative cycle point (t_today / t_final), looks up how many
    entries were registered in each historical year at the same relative point,
    computes fraction = count_at_point / final_count for each year, then:

        predicted_final = current_count / median_fraction

    CI is derived from the min/max fraction range across historical years.

    This model dominates early-cycle when sigmoid models fit a flat plateau and
    underpredict. Its weight fades once enough data exists to fit a real curve.

    Historical data format (loaded from .tmp/{family}_historical.json):
        { "2022": {"cumulative": [[t, count], ...], "total_entries": N}, ... }
    """

    def __init__(self, historical_data: dict | None = None):
        self.historical = historical_data or {}
        self.fitted = False
        self.fractions: list[float] = []
        self.implied_finals: list[float] = []
        self.year_details: list[dict] = []
        self.pct_elapsed: float = 0.0

    def fit(self, t_today: float, t_final: float, count_now: float) -> bool:
        if not self.historical or count_now <= 0 or t_final <= 0:
            return False

        self.pct_elapsed = t_today / t_final
        self.fractions = []
        self.implied_finals = []
        self.year_details = []

        for year, data in self.historical.items():
            pts = data.get("cumulative", [])
            final = data.get("total_entries", 0)
            if not pts or final <= 0:
                continue

            # t_equiv: same relative point in this year's cycle
            yr_t_max = max(p[0] for p in pts)
            t_equiv = self.pct_elapsed * yr_t_max

            # Interpolate count at t_equiv
            count_at = 0.0
            for i, (t, c) in enumerate(pts):
                if t <= t_equiv:
                    count_at = float(c)
                else:
                    # Linear interpolation between previous and current point
                    if i > 0:
                        t_prev, c_prev = pts[i - 1]
                        if t > t_prev:
                            frac = (t_equiv - t_prev) / (t - t_prev)
                            count_at = float(c_prev) + frac * (float(c) - float(c_prev))
                    break

            if count_at <= 0:
                continue

            fraction = count_at / final
            implied = count_now / fraction
            self.fractions.append(fraction)
            self.implied_finals.append(implied)
            self.year_details.append({
                "year": year, "fraction": round(fraction, 4),
                "count_at_equiv": round(count_at, 1),
                "final": final, "implied": round(implied, 0),
            })

        self.fitted = len(self.fractions) >= 1
        return self.fitted

    def predict(self) -> float:
        if not self.fitted:
            return 0.0
        s = sorted(self.implied_finals)
        n = len(s)
        return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2

    def predict_with_ci(self) -> dict:
        if not self.fitted:
            return {"predicted": 0.0, "ci_low": 0.0, "ci_high": 0.0}
        s = sorted(self.implied_finals)
        n = len(s)
        pred = s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2
        # IQR if ≥4 years, else min/max
        ci_low  = s[n // 4]     if n >= 4 else s[0]
        ci_high = s[3 * n // 4] if n >= 4 else s[-1]
        # Ensure ci contains pred
        ci_low  = min(ci_low,  pred)
        ci_high = max(ci_high, pred)
        return {
            "predicted": round(pred, 1),
            "ci_low": round(ci_low, 1),
            "ci_high": round(ci_high, 1),
            "pct_elapsed": round(self.pct_elapsed * 100, 1),
            "n_years": n,
            "year_details": self.year_details,
        }


# ── Ensemble Combiner ──────────────────────────────────────────────────────────

class EnsemblePredictor:
    """
    Combines predictions from all model families into a single estimate.

    Weighting strategy:
    - With >= 20 data points: emphasize data-fitted models (sigmoid, MC, MCMC)
    - With 10-19 points: balanced weighting
    - With < 10 points: emphasize prior-heavy models (Thompson, MCMC)
    - If a model failed to fit, its weight is set to zero and others rebalanced.

    Final prediction: weighted mean of model predictions.
    CI bounds: weighted combination of individual model CI bounds.
    """

    def __init__(self, n_data_points: int = 0):
        self.n_data = n_data_points
        self.model_results: dict = {}

    def add_result(self, name: str, predicted: float, ci_low: float, ci_high: float,
                   weight: float = 1.0, fitted: bool = True):
        self.model_results[name] = {
            "predicted": predicted,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "weight": weight,
            "fitted": fitted,
        }

    def _compute_weights(self) -> dict[str, float]:
        """Compute normalized weights based on data availability and fit quality."""
        n = self.n_data

        # Base weights by data density
        if n >= 50:
            # Late cycle: sigmoid models are well-calibrated, ratio fades out
            base = {
                "single_sigmoid": 2.0,
                "double_sigmoid": 2.0,
                "monte_carlo": 2.5,
                "bayesian_mcmc": 2.0,
                "thompson": 1.0,
                "historical_ratio": 0.5,
            }
        elif n >= 20:
            # Mid cycle: balance data-fitted and historical
            base = {
                "single_sigmoid": 1.5,
                "double_sigmoid": 1.5,
                "monte_carlo": 2.0,
                "bayesian_mcmc": 2.0,
                "thompson": 1.5,
                "historical_ratio": 2.5,
            }
        elif n >= 10:
            base = {
                "single_sigmoid": 0.5,
                "double_sigmoid": 0.5,
                "monte_carlo": 1.0,
                "bayesian_mcmc": 1.5,
                "thompson": 1.5,
                "historical_ratio": 5.0,
            }
        else:
            # Sparse data: historical ratio dominates
            base = {
                "single_sigmoid": 0.5,
                "double_sigmoid": 0.5,
                "monte_carlo": 0.5,
                "bayesian_mcmc": 1.5,
                "thompson": 1.5,
                "historical_ratio": 4.0,
            }

        # Zero out models that didn't fit or are implausibly low
        # When a data-fitted model predicts far below the historical_ratio,
        # it's fitting a flat early-cycle plateau rather than the true curve.
        hist_pred = self.model_results.get("historical_ratio", {}).get("predicted", 0)
        for name, result in self.model_results.items():
            if not result["fitted"] or result["predicted"] <= 0:
                base[name] = 0.0
            elif (name not in ("historical_ratio", "thompson", "bayesian_mcmc")
                  and hist_pred > 0
                  and result["predicted"] < hist_pred * 0.15):
                # Sigmoid/MC models predicting <15% of historical ratio → implausible
                base[name] = 0.0

        # Normalize
        total = sum(base.get(n, 0) for n in self.model_results if base.get(n, 0) > 0)
        if total <= 0:
            return {n: 1.0 / len(self.model_results) for n in self.model_results}

        return {n: base.get(n, 0) / total for n in self.model_results}

    def predict(self) -> dict:
        if not self.model_results:
            return {}

        weights = self._compute_weights()

        predicted = 0.0
        ci_low = 0.0
        ci_high = 0.0
        total_w = 0.0

        for name, result in self.model_results.items():
            w = weights.get(name, 0.0)
            if w <= 0 or not result["fitted"]:
                continue
            predicted += w * result["predicted"]
            ci_low += w * result["ci_low"]
            ci_high += w * result["ci_high"]
            total_w += w

        if total_w == 0:
            return {}

        return {
            "predicted_final": round(predicted, 1),
            "ci_low": round(ci_low, 1),
            "ci_high": round(ci_high, 1),
            "weights": {n: round(w, 3) for n, w in weights.items()},
            "n_models": sum(1 for r in self.model_results.values() if r["fitted"]),
        }


# ── Workflow: run all models on one tournament ─────────────────────────────────

def load_tournament_config(tid: str) -> dict | None:
    config_path = ROOT / "config.json"
    if not config_path.exists():
        return None
    with open(config_path) as f:
        cfg = json.load(f)
    for t in cfg.get("tournaments", []):
        if t["id"] == tid:
            return t
    return None


def load_historical_data(tournament_cfg: dict | None) -> dict:
    """
    Load same-family historical registration curves from .tmp/{family}_historical.json.
    Returns dict keyed by year, or empty dict if not available.
    """
    if not tournament_cfg:
        return {}
    family = tournament_cfg.get("tournament_family", "")
    if not family:
        return {}
    # Map family name to file: "world_open" → "wo_historical.json", etc.
    family_file_map = {
        "world_open": "wo_historical.json",
        "chicago_open": "chi_historical.json",
        "north_american_open": "nao_historical.json",
    }
    fname = family_file_map.get(family, f"{family}_historical.json")
    path = TMP / fname
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def load_priors_from_history() -> dict:
    """
    Load empirical prior distributions from fit_report.json if available.
    Falls back to reasonable defaults derived from Chicago 2022 / NAO 2024.
    """
    report_path = TMP / "fit_report.json"
    if report_path.exists():
        try:
            with open(report_path) as f:
                report = json.load(f)
            pd = report.get("prior_distributions", {})
            return {
                "L_mean": pd.get("L2", {}).get("mean", 500),
                "L_sd": pd.get("L2", {}).get("sd", 400),
                "k_mean": pd.get("k2", {}).get("mean", 0.10),
                "m_mean": None,  # will be set per-tournament
                "m_sd": None,
            }
        except Exception:
            pass

    # Defaults: average of Chicago 2022 (L=943, k≈0.10) and NAO 2024 (L=1102, k≈0.07)
    return {
        "L_mean": 700.0,
        "L_sd": 350.0,
        "k_mean": 0.07,
        "m_mean": None,
        "m_sd": None,
    }


def run_ensemble(
    tid: str,
    cumulative_data: list[list],
    tournament_cfg: dict | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run all five model families and return a combined ensemble result.

    Args:
        tid:              Tournament identifier
        cumulative_data:  [[t, cumulative_count], ...] sorted by t
        tournament_cfg:   From config.json (optional but recommended)
        verbose:          Print progress

    Returns:
        Full ensemble result dict, also saved to .tmp/ensemble_{tid}.json
    """
    if not cumulative_data:
        return {"error": "No cumulative data provided"}

    ts = [d[0] for d in cumulative_data]
    ys = [d[1] for d in cumulative_data]
    n_points = len(cumulative_data)
    t_today = max(ts)
    count_now = max(ys)

    # Tournament metadata
    has_early_bird = False
    t_final = None
    expected_total = None

    if tournament_cfg:
        # Prefer explicit has_early_bird flag; fall back to fee_deadlines presence
        if "has_early_bird" in tournament_cfg:
            has_early_bird = bool(tournament_cfg["has_early_bird"])
        else:
            has_early_bird = bool(tournament_cfg.get("fee_deadlines"))
        if tournament_cfg.get("tournament_date") and tournament_cfg.get("first_reg_date"):
            td = date.fromisoformat(tournament_cfg["tournament_date"])
            fr = date.fromisoformat(tournament_cfg["first_reg_date"])
            t_final = (td - fr).days
        expected_total = tournament_cfg.get("expected_total")

    if t_final is None:
        # Estimate: assume we're ~60% through registration
        t_final = int(t_today / 0.6) if t_today > 0 else 200

    if verbose:
        print(f"\n{'='*60}")
        print(f"ENSEMBLE: {tid}")
        print(f"  Data points: {n_points}  |  Current count: {count_now}")
        print(f"  t_today: {t_today}  |  t_final: {t_final}")
        print(f"  has_early_bird: {has_early_bird}")
        print(f"{'='*60}")

    priors = load_priors_from_history()
    priors["m_mean"] = t_final * 0.65
    priors["m_sd"] = t_final * 0.25

    ensemble = EnsemblePredictor(n_data_points=n_points)
    results = {}

    # ── Model 1: Single Sigmoid ───────────────────────────────────────────────
    if verbose: print("\n[1/5] Single Sigmoid...")
    m1 = SingleSigmoidModel()
    ok1 = m1.fit(cumulative_data)
    if ok1:
        pred1 = m1.predict_with_ci(t_final)
        if verbose:
            print(f"     R²={m1.r2:.3f}  RMSE={m1.rmse_val:.1f}")
            print(f"     Predicted final: {pred1['predicted']} "
                  f"[{pred1['ci_low']} – {pred1['ci_high']}]")
        results["single_sigmoid"] = {
            "predicted_final": pred1["predicted"],
            "ci_low": pred1["ci_low"],
            "ci_high": pred1["ci_high"],
            "params": m1.params,
            "r2": round(m1.r2, 4),
            "rmse": round(m1.rmse_val, 2),
            "fitted": True,
        }
        ensemble.add_result("single_sigmoid", pred1["predicted"], pred1["ci_low"],
                            pred1["ci_high"], fitted=True)
    else:
        if verbose: print("     FAILED (insufficient data or convergence)")
        results["single_sigmoid"] = {"fitted": False, "reason": "fit failed"}
        ensemble.add_result("single_sigmoid", 0, 0, 0, fitted=False)

    # ── Model 2: Double Sigmoid ───────────────────────────────────────────────
    if verbose: print("\n[2/5] Double Sigmoid...")
    m2 = DoubleSigmoidModel()
    ok2 = m2.fit(cumulative_data)
    if ok2:
        pred2 = m2.predict_with_ci(t_final)
        if verbose:
            print(f"     R²={m2.r2:.3f}  RMSE={m2.rmse_val:.1f}")
            print(f"     Predicted final: {pred2['predicted']} "
                  f"[{pred2['ci_low']} – {pred2['ci_high']}]")
        results["double_sigmoid"] = {
            "predicted_final": pred2["predicted"],
            "ci_low": pred2["ci_low"],
            "ci_high": pred2["ci_high"],
            "params": m2.params,
            "r2": round(m2.r2, 4),
            "rmse": round(m2.rmse_val, 2),
            "fitted": True,
        }
        ensemble.add_result("double_sigmoid", pred2["predicted"], pred2["ci_low"],
                            pred2["ci_high"], fitted=True)
    else:
        if verbose: print("     FAILED (need ≥6 points; double-sigmoid harder to fit early on)")
        results["double_sigmoid"] = {"fitted": False, "reason": "fit failed"}
        ensemble.add_result("double_sigmoid", 0, 0, 0, fitted=False)

    # ── Model 3: Monte Carlo Bootstrap ───────────────────────────────────────
    if verbose: print("\n[3/5] Monte Carlo Bootstrap...")
    m3 = MonteCarloModel(has_early_bird=has_early_bird, n_samples=2000)
    ok3 = m3.fit(cumulative_data)
    if ok3:
        dist3 = m3.predict_distribution(t_final)
        if verbose:
            print(f"     n_samples={dist3['n_samples']}")
            print(f"     p10={dist3['p10']}  median={dist3['median']}  p90={dist3['p90']}")
        results["monte_carlo"] = {
            "predicted_final": dist3["median"],
            "ci_low": dist3["p10"],
            "ci_high": dist3["p90"],
            "distribution": dist3,
            "fitted": True,
        }
        ensemble.add_result("monte_carlo", dist3["median"], dist3["p10"], dist3["p90"], fitted=True)
    else:
        if verbose: print("     FAILED")
        results["monte_carlo"] = {"fitted": False}
        ensemble.add_result("monte_carlo", 0, 0, 0, fitted=False)

    # ── Model 4: Bayesian MCMC ────────────────────────────────────────────────
    if verbose: print("\n[4/5] Bayesian MCMC (Metropolis-Hastings)...")
    m4 = BayesianMCMCModel(n_iter=4000, priors=priors)
    ok4 = m4.fit(cumulative_data)
    if ok4:
        dist4 = m4.predict_distribution(t_final)
        param_post = m4.get_param_posteriors()
        if verbose:
            print(f"     Acceptance rate: {dist4.get('acceptance_rate', 0):.1%}")
            print(f"     Posterior L: mean={param_post.get('L', {}).get('mean', '?')}")
            print(f"     p10={dist4['p10']}  median={dist4['median']}  p90={dist4['p90']}")
        results["bayesian_mcmc"] = {
            "predicted_final": dist4["median"],
            "ci_low": dist4["p10"],
            "ci_high": dist4["p90"],
            "distribution": dist4,
            "param_posteriors": param_post,
            "acceptance_rate": dist4.get("acceptance_rate"),
            "fitted": True,
        }
        ensemble.add_result("bayesian_mcmc", dist4["median"], dist4["p10"], dist4["p90"], fitted=True)
    else:
        if verbose: print("     FAILED")
        results["bayesian_mcmc"] = {"fitted": False}
        ensemble.add_result("bayesian_mcmc", 0, 0, 0, fitted=False)

    # ── Model 5: Thompson Sampling ────────────────────────────────────────────
    if verbose: print("\n[5/5] Thompson Sampling...")
    prior_mean = expected_total or priors.get("L_mean", 500)
    prior_std = priors.get("L_sd", prior_mean * 0.40)
    m5 = ThompsonSamplingModel(prior_mean=prior_mean, prior_std=prior_std)

    # Set sigmoid shape from best-fitted model for fraction calculation
    if ok1 and m1.fitted:
        m5.set_sigmoid_shape(m1.params)
    elif tournament_cfg and tournament_cfg.get("prior_params"):
        pp = tournament_cfg["prior_params"]
        if "L" in pp:
            m5.set_sigmoid_shape(pp)
        elif "L2" in pp:
            # Double sigmoid prior — use single approximation
            total_L = pp.get("L1", 0) + pp.get("L2", 0)
            m5.set_sigmoid_shape({"L": total_L, "k": pp.get("k2", 0.1), "m": pp.get("m2", t_final * 0.8)})

    # Update with observed data points — use only the latest count at each t
    # to avoid overcounting when cumulative values plateau between days.
    prev_count = -1.0
    for t, count in cumulative_data:
        if float(count) > prev_count:
            m5.update(float(t), float(count), float(t_final))
            prev_count = float(count)

    ts_dist = m5.sample(n_samples=2000)
    if verbose:
        print(f"     Prior: N({prior_mean:.0f}, {prior_std:.0f})")
        print(f"     Posterior: N({ts_dist['posterior_mean']:.1f}, {ts_dist['posterior_std']:.1f})")
        print(f"     p10={ts_dist['p10']}  median={ts_dist['median']}  p90={ts_dist['p90']}")

    results["thompson_sampling"] = {
        "predicted_final": ts_dist["median"],
        "ci_low": ts_dist["p10"],
        "ci_high": ts_dist["p90"],
        "distribution": ts_dist,
        "prior": {"mean": prior_mean, "std": prior_std},
        "fitted": True,
    }
    ensemble.add_result("thompson", ts_dist["median"], ts_dist["p10"], ts_dist["p90"], fitted=True)

    # ── Model 6: Historical Ratio ─────────────────────────────────────────────
    if verbose: print("\n[6/6] Historical Ratio...")
    historical_data = load_historical_data(tournament_cfg)
    m6 = HistoricalRatioModel(historical_data=historical_data)
    ok6 = m6.fit(t_today, t_final, count_now)
    if ok6:
        pred6 = m6.predict_with_ci()
        if verbose:
            print(f"     {pred6['n_years']} historical years at {pred6['pct_elapsed']:.1f}% of cycle")
            for yd in pred6.get("year_details", []):
                print(f"       {yd['year']}: {yd['count_at_equiv']:.0f}/{yd['final']} "
                      f"= {yd['fraction']:.3f} → implied {yd['implied']:.0f}")
            print(f"     Predicted final: {pred6['predicted']} "
                  f"[{pred6['ci_low']} – {pred6['ci_high']}]")
        results["historical_ratio"] = {
            "predicted_final": pred6["predicted"],
            "ci_low": pred6["ci_low"],
            "ci_high": pred6["ci_high"],
            "pct_elapsed": pred6["pct_elapsed"],
            "year_details": pred6.get("year_details", []),
            "fitted": True,
        }
        ensemble.add_result("historical_ratio", pred6["predicted"],
                            pred6["ci_low"], pred6["ci_high"], fitted=True)
    else:
        if verbose: print("     SKIPPED (no historical family data in .tmp/)")
        results["historical_ratio"] = {"fitted": False, "reason": "no historical data"}
        ensemble.add_result("historical_ratio", 0, 0, 0, fitted=False)

    # ── Ensemble combination ──────────────────────────────────────────────────
    if verbose: print("\n── Ensemble ─────────────────────────────────────")
    ensemble_pred = ensemble.predict()

    if verbose and ensemble_pred:
        print(f"  Predicted final: {ensemble_pred['predicted_final']}")
        print(f"  CI: [{ensemble_pred['ci_low']} – {ensemble_pred['ci_high']}]")
        print(f"  Weights: {ensemble_pred['weights']}")
        print(f"  Models contributing: {ensemble_pred['n_models']}/6")

    # ── Assemble output ───────────────────────────────────────────────────────
    output = {
        "tournament_id": tid,
        "as_of": date.today().isoformat(),
        "t_today": t_today,
        "t_final": t_final,
        "actual_count": count_now,
        "n_data_points": n_points,
        "has_early_bird": has_early_bird,
        "models": results,
        "ensemble": ensemble_pred,
        "expected_total": expected_total,
    }

    # Save to .tmp/
    out_path = TMP / f"ensemble_{tid}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    if verbose:
        print(f"\nSaved → {out_path}")

    return output


# ── Backtest: evaluate models on historical full datasets ──────────────────────

def run_backtest(verbose: bool = True):
    """
    Simulate predictions at different stages of registration for historical
    tournaments (Chicago 2022, NAO 2024) and measure prediction accuracy.
    """
    datasets = [
        ("CHI2022", TMP / "cumulative_CHI2022.json", True),
        ("NAO2024", TMP / "cumulative_NAO2024.json", False),
    ]

    print("\n" + "=" * 70)
    print("BACKTEST: Simulating predictions at 25%, 50%, 75% of registration")
    print("=" * 70)

    for tid, cum_path, has_eb in datasets:
        if not cum_path.exists():
            print(f"\n{tid}: No cumulative data at {cum_path}. "
                  "Run prepare_training_data.py first.")
            continue

        with open(cum_path) as f:
            full_data = json.load(f)

        actual_final = full_data[-1][1]
        t_final = full_data[-1][0]

        print(f"\n{tid}: {len(full_data)} data points, actual final = {actual_final}")

        for frac in [0.25, 0.50, 0.75]:
            cutoff_t = t_final * frac
            partial = [[t, c] for t, c in full_data if t <= cutoff_t]
            if len(partial) < 3:
                continue

            result = run_ensemble(tid, partial, verbose=False)
            ens = result.get("ensemble", {})
            predicted = ens.get("predicted_final", 0)
            error_pct = (predicted - actual_final) / actual_final * 100 if actual_final else 0

            print(f"  At {frac:.0%} ({len(partial)} pts, count={partial[-1][1]}): "
                  f"predicted={predicted:.0f}  actual={actual_final}  "
                  f"error={error_pct:+.1f}%")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-model ensemble predictor for chess tournament entries"
    )
    parser.add_argument("--id", help="Tournament ID (e.g. WO2026, CHI2022, NAO2024)")
    parser.add_argument("--data", help="Path to cumulative JSON (overrides auto-discovery)")
    parser.add_argument("--compare", action="store_true",
                        help="Run on all available .tmp/cumulative_*.json files")
    parser.add_argument("--backtest", action="store_true",
                        help="Backtest models on historical full datasets")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    verbose = not args.quiet

    if args.backtest:
        run_backtest(verbose=True)
        return

    if args.compare:
        pattern = list(TMP.glob("cumulative_*.json"))
        if not pattern:
            print("No cumulative data files found in .tmp/. Run prepare_training_data.py first.")
            return
        for p in pattern:
            tid = p.stem.replace("cumulative_", "")
            with open(p) as f:
                data = json.load(f)
            cfg = load_tournament_config(tid)
            run_ensemble(tid, data, tournament_cfg=cfg, verbose=verbose)
        return

    if args.id:
        tid = args.id
        if args.data:
            data_path = Path(args.data)
        else:
            data_path = TMP / f"cumulative_{tid}.json"

        if not data_path.exists():
            print(f"ERROR: No data at {data_path}")
            print(f"Run: python tools/prepare_training_data.py --tournament {tid.lower()}")
            sys.exit(1)

        with open(data_path) as f:
            data = json.load(f)

        cfg = load_tournament_config(tid)
        result = run_ensemble(tid, data, tournament_cfg=cfg, verbose=verbose)

        if verbose:
            ens = result.get("ensemble", {})
            print(f"\n{'─'*50}")
            print(f"FINAL PREDICTION for {tid}:")
            print(f"  Ensemble: {ens.get('predicted_final', 'N/A')} "
                  f"[{ens.get('ci_low', '?')} – {ens.get('ci_high', '?')}]")
            print(f"  Current:  {result['actual_count']} registered")
            print(f"{'─'*50}")
        return

    # Default: run on WO2026 if available
    wo_path = TMP / "cumulative_WO2026.json"
    if wo_path.exists():
        with open(wo_path) as f:
            data = json.load(f)
        cfg = load_tournament_config("WO2026")
        run_ensemble("WO2026", data, tournament_cfg=cfg, verbose=verbose)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
