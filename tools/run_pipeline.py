"""
run_pipeline.py
---------------
Main orchestrator for the chess tournament entry prediction pipeline.

Reads config.json, then for each active tournament:
  1. Scrapes current registration entries (or loads from cache)
  2. Aggregates to cumulative daily counts
  3. Fits (or uses prior) double-sigmoid parameters
  4. Computes delta vs. model and final prediction
  5. Appends to history log
  6. Triggers alerts if delta exceeds threshold

Usage:
    python tools/run_pipeline.py                        # run all active tournaments
    python tools/run_pipeline.py --id WO2026            # run one tournament
    python tools/run_pipeline.py --dry-run              # skip scraping, use cached data
    python tools/run_pipeline.py --no-scrape --id WO2026  # fit + predict only
"""

import argparse
import json
import math
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path


# ── Load config ───────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config.json"
TMP = ROOT / ".tmp"
TMP.mkdir(exist_ok=True)

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


# ── Double sigmoid (inline so pipeline has no inter-tool imports) ─────────────

def double_sigmoid(t, L1, k1, m1, L2, k2, m2):
    def safe_sig(k, t, m):
        x = -k * (t - m)
        if x > 500: return 0.0
        if x < -500: return 1.0
        return 1.0 / (1.0 + math.exp(x))
    return L1 * safe_sig(k1, t, m1) + L2 * safe_sig(k2, t, m2)


# ── Step 1: Scrape ────────────────────────────────────────────────────────────

def _scrape_one_url(url: str, out_path: Path) -> dict | None:
    """Scrape a single registration URL. Returns parsed JSON or None on failure."""
    import subprocess
    scraper = Path(__file__).parent / "scrape_entries.py"
    result = subprocess.run(
        [sys.executable, str(scraper),
         "--url", url,
         "--out", str(out_path),
         "--verbose"],
        capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        print(f"  [scrape] ERROR scraping {url}:\n{result.stderr}", file=sys.stderr)
        return None
    with open(out_path) as f:
        return json.load(f)


def run_scrape(tournament: dict, dry_run: bool = False) -> dict | None:
    """
    Calls scrape_entries.py as a subprocess and returns parsed JSON result.
    Falls back to cached data if dry_run or if scraping fails.

    If the tournament has a 'sections' array, scrapes each section URL and
    merges all entries into a single result (entry counts are summed).
    """
    tid = tournament["id"]
    cache_path = TMP / f"entries_{tid}.json"
    sections = tournament.get("sections", [])
    url = tournament.get("registration_url")

    # Determine all URLs to scrape
    if sections:
        urls = [(s["label"], s["registration_url"]) for s in sections if s.get("registration_url")]
    elif url:
        urls = [("", url)]
    else:
        urls = []

    if dry_run or not urls:
        if cache_path.exists():
            print(f"  [scrape] Using cached data: {cache_path}")
            with open(cache_path) as f:
                return json.load(f)
        else:
            print(f"  [scrape] No URL and no cache for {tid} — skipping")
            return None

    # Scrape each URL (sections or single)
    all_entries = []
    section_counts = []
    any_success = False

    for label, section_url in urls:
        tag = f" [{label}]" if label else ""
        tmp_path = TMP / f"entries_{tid}_section_{len(section_counts)}.json"
        data = _scrape_one_url(section_url, tmp_path)
        if data:
            entries = data.get("entries", [])
            all_entries.extend(entries)
            section_counts.append({"label": label or "main", "url": section_url, "count": len(entries)})
            print(f"  [scrape]{tag} {len(entries)} entries")
            any_success = True
        else:
            # Fall back to per-section cache if available
            if tmp_path.exists():
                with open(tmp_path) as f:
                    data = json.load(f)
                entries = data.get("entries", [])
                all_entries.extend(entries)
                section_counts.append({"label": label or "main", "url": section_url, "count": len(entries), "from_cache": True})
                print(f"  [scrape]{tag} {len(entries)} entries (from cache)")
                any_success = True

    if not any_success:
        if cache_path.exists():
            print(f"  [scrape] All sections failed — falling back to merged cache")
            with open(cache_path) as f:
                return json.load(f)
        return None

    # Build merged result
    merged = {
        "scraped_at": datetime.utcnow().isoformat() + "Z",
        "url": url or sections[0]["registration_url"] if sections else url,
        "count": len(all_entries),
        "entries": all_entries,
    }
    if len(section_counts) > 1:
        merged["sections"] = section_counts

    with open(cache_path, "w") as f:
        json.dump(merged, f, indent=2)

    return merged


# ── Step 2: Aggregate to cumulative daily counts ──────────────────────────────

def aggregate_cumulative(entries_result: dict, first_reg_date: date) -> list[list]:
    """
    Convert raw entry list to [[t, cumulative_count], ...] sorted by t.
    Entries with null registered_time (walk-ins) are excluded from time-series.
    """
    entries = entries_result.get("entries", [])

    daily = {}
    for e in entries:
        raw_time = e.get("registered_time")
        if not raw_time:
            continue
        try:
            if "T" in raw_time:
                reg_date = datetime.fromisoformat(raw_time).date()
            else:
                reg_date = date.fromisoformat(raw_time[:10])
        except (ValueError, TypeError):
            continue
        t = (reg_date - first_reg_date).days
        daily[t] = daily.get(t, 0) + 1

    if not daily:
        return []

    cumulative = []
    running = 0
    for t in sorted(daily.keys()):
        running += daily[t]
        cumulative.append([t, running])
    return cumulative


# ── Step 3: Fit model ─────────────────────────────────────────────────────────

def fit_model(data: list[list], prior_params: dict | None, min_points: int = 10) -> tuple[dict, str]:
    """
    Returns (params_dict, source_string).
    source: 'fitted' | 'prior' | 'prior_fallback'
    """
    if len(data) >= min_points:
        try:
            import numpy as np
            from scipy.optimize import curve_fit

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

            p0 = [y_max * 0.12, 0.05, t_at_frac(0.25),
                  y_max * 0.88, 0.25, t_at_frac(0.75)]
            bounds = (
                [0, 0.001, 0, 0, 0.001, 0],
                [y_max * 2, 2, t_max * 2, y_max * 5, 2, t_max * 3]
            )

            def model(t, L1, k1, m1, L2, k2, m2):
                return np.array([double_sigmoid(ti, L1, k1, m1, L2, k2, m2) for ti in t])

            popt, _ = curve_fit(model, ts, ys, p0=p0, bounds=bounds, maxfev=50000, method="trf")
            L1, k1, m1, L2, k2, m2 = popt

            preds = np.array([double_sigmoid(t, L1, k1, m1, L2, k2, m2) for t in ts])
            ss_res = float(np.sum((ys - preds) ** 2))
            ss_tot = float(np.sum((ys - ys.mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            rmse = float(np.sqrt(ss_res / len(ys)))

            return ({
                "L1": round(L1, 4), "k1": round(k1, 6), "m1": round(m1, 2),
                "L2": round(L2, 4), "k2": round(k2, 6), "m2": round(m2, 2),
                "r2": round(r2, 4), "rmse": round(rmse, 3),
                "n_points": len(data), "converged": True,
            }, "fitted")

        except Exception as e:
            print(f"  [fit] scipy fit failed ({e}); using prior params", file=sys.stderr)

    if prior_params:
        source = "prior" if len(data) < min_points else "prior_fallback"
        return (dict(prior_params), source)

    return ({}, "none")


# ── Step 4: Compute delta and prediction ──────────────────────────────────────

def compute_prediction(tournament: dict, params: dict, actual_count: int, today: date) -> dict:
    first_reg = date.fromisoformat(tournament["first_reg_date"])
    tourney_date = date.fromisoformat(tournament["tournament_date"])
    t_today = (today - first_reg).days
    t_final = (tourney_date - first_reg).days

    L1, k1, m1 = params["L1"], params["k1"], params["m1"]
    L2, k2, m2 = params["L2"], params["k2"], params["m2"]
    rmse = params.get("rmse", 0)

    predicted_now = double_sigmoid(t_today, L1, k1, m1, L2, k2, m2)
    predicted_final = double_sigmoid(t_final, L1, k1, m1, L2, k2, m2)

    if predicted_now > 0:
        delta_pct = (actual_count - predicted_now) / predicted_now * 100
    else:
        delta_pct = 0.0

    threshold = 10.0
    if delta_pct > threshold:
        status = "GREEN"
    elif delta_pct < -threshold:
        status = "RED"
    else:
        status = "YELLOW"

    # Confidence interval
    margin = max(predicted_final * 0.10, 2 * rmse) if rmse else predicted_final * 0.15
    ci_low = round(predicted_final - margin, 1)
    ci_high = round(predicted_final + margin, 1)

    return {
        "tournament": tournament["name"],
        "tournament_id": tournament["id"],
        "as_of": today.isoformat(),
        "t_today": round(t_today, 1),
        "t_final": round(t_final, 1),
        "actual_count": actual_count,
        "predicted_now": round(predicted_now, 1),
        "delta_pct": round(delta_pct, 1),
        "status": status,
        "predicted_final": round(predicted_final, 1),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "params": {k: params[k] for k in ["L1","k1","m1","L2","k2","m2"] if k in params},
        "r2": params.get("r2"),
        "rmse": rmse,
    }


# ── Step 5: Append to history ─────────────────────────────────────────────────

def append_history(tid: str, prediction: dict):
    history_path = TMP / f"history_{tid}.jsonl"
    with open(history_path, "a") as f:
        f.write(json.dumps(prediction) + "\n")
    print(f"  [history] Appended to {history_path}")


# ── Step 6: Alert check ───────────────────────────────────────────────────────

def check_alerts(prediction: dict, settings: dict) -> bool:
    """Returns True if alert was triggered. Suppressed if > 100 days to event."""
    threshold = settings.get("alert_delta_threshold_pct", 10)
    delta = abs(prediction["delta_pct"])

    # Suppress alerts until within 100 days of the event
    # t_final and t_today are both in calendar days from first_reg_date
    t_remaining = prediction.get("t_final", 0) - prediction.get("t_today", 0)
    if t_remaining > 100:
        return False

    if delta > threshold:
        direction = "ABOVE" if prediction["delta_pct"] > 0 else "BELOW"
        msg = (
            f"ALERT [{prediction['tournament']}] "
            f"Tracking {prediction['delta_pct']:+.1f}% {direction} model "
            f"({prediction['actual_count']} actual vs {prediction['predicted_now']:.0f} expected) "
            f"— Status: {prediction['status']}"
        )
        print(f"\n  *** {msg} ***\n")

        # Write alert to file for record-keeping
        alert_path = TMP / f"alert_{prediction['tournament_id']}_{prediction['as_of']}.txt"
        alert_path.write_text(msg)

        # Dispatch email + SMS via send_alert.py (fires if .env credentials are set)
        import subprocess as _sp
        pred_path = TMP / f"prediction_{prediction['tournament_id']}.json"
        if pred_path.exists():
            _sp.Popen(
                [sys.executable, str(Path(__file__).parent / "send_alert.py"),
                 "--file", str(pred_path)],
                cwd=str(ROOT),
            )
        return True
    return False


# ── Per-tournament pipeline ───────────────────────────────────────────────────

def run_tournament(tournament: dict, settings: dict, args) -> dict | None:
    tid = tournament["id"]
    name = tournament["name"]
    print(f"\n{'='*60}")
    print(f"  {name} ({tid})")
    print(f"{'='*60}")

    today = date.today()
    first_reg = date.fromisoformat(tournament["first_reg_date"])

    # Step 1: Scrape
    if args.no_scrape:
        cache_path = TMP / f"entries_{tid}.json"
        if cache_path.exists():
            print(f"  [scrape] --no-scrape: loading cache {cache_path}")
            with open(cache_path) as f:
                entries_result = json.load(f)
        else:
            print(f"  [scrape] --no-scrape: no cache found, using count=0")
            entries_result = {"count": 0, "entries": []}
    else:
        entries_result = run_scrape(tournament, dry_run=args.dry_run)
        if entries_result is None:
            print(f"  [skip] No data available for {tid}")
            return None

    actual_count = entries_result.get("count", len(entries_result.get("entries", [])))
    print(f"  [scrape] {actual_count} entries")

    # Step 2: Aggregate
    cumulative = aggregate_cumulative(entries_result, first_reg)
    print(f"  [aggregate] {len(cumulative)} daily data points")

    # Save cumulative for downstream use
    cum_path = TMP / f"cumulative_{tid}.json"
    with open(cum_path, "w") as f:
        json.dump(cumulative, f)

    # Step 3: Fit
    prior = tournament.get("prior_params")
    params, source = fit_model(cumulative, prior)
    if not params:
        print(f"  [fit] No params available for {tid} — skipping")
        return None
    print(f"  [fit] source={source}, R²={params.get('r2','N/A')}, RMSE={params.get('rmse','N/A')}")

    # Save params
    param_path = TMP / f"params_{tid}.json"
    with open(param_path, "w") as f:
        json.dump(params, f, indent=2)

    # Step 4: Predict
    prediction = compute_prediction(tournament, params, actual_count, today)

    # Save prediction
    pred_path = TMP / f"prediction_{tid}.json"
    with open(pred_path, "w") as f:
        json.dump(prediction, f, indent=2)

    # Step 5: History
    append_history(tid, prediction)

    # Print summary
    print(f"\n  Actual:    {actual_count:,}")
    print(f"  Expected:  {prediction['predicted_now']:.1f}")
    print(f"  Delta:     {prediction['delta_pct']:+.1f}%  [{prediction['status']}]")
    print(f"  Final est: {prediction['predicted_final']:.0f}  ({prediction['ci_low']:.0f}–{prediction['ci_high']:.0f})")

    # Step 6: Alerts
    if not args.no_alert:
        alerted = check_alerts(prediction, settings)

    return prediction


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Run chess tournament prediction pipeline")
    parser.add_argument("--id", help="Run only this tournament ID (e.g. WO2026)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip live scraping; use cached data")
    parser.add_argument("--no-scrape", action="store_true",
                        help="Skip scraping entirely; load cached entries or use count=0")
    parser.add_argument("--no-alert", action="store_true",
                        help="Skip alert checks")
    parser.add_argument("--config", default=str(CONFIG_PATH),
                        help="Path to config.json")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    settings = cfg.get("settings", {})
    tournaments = cfg.get("tournaments", [])

    # Filter by --id
    if args.id:
        tournaments = [t for t in tournaments if t["id"] == args.id]
        if not tournaments:
            print(f"ERROR: No tournament with id='{args.id}' found in config.json", file=sys.stderr)
            sys.exit(1)

    # Only active tournaments (unless specific --id given)
    if not args.id:
        tournaments = [t for t in tournaments if t.get("active", True)]

    if not tournaments:
        print("No active tournaments to process.")
        sys.exit(0)

    print(f"Chess Tournament Prediction Pipeline — {date.today()}")
    print(f"Processing {len(tournaments)} tournament(s)")

    results = []
    for t in tournaments:
        result = run_tournament(t, settings, args)
        if result:
            results.append(result)

    # Summary JSON — useful for API / dashboard to load
    summary_path = TMP / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "tournaments": results
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Pipeline complete. Summary → {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
