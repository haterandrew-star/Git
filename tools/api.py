"""
api.py
------
FastAPI backend for the Chess Tournament Entry Prediction dashboard.

Endpoints:
    GET  /                          Health check
    GET  /tournaments               List all configured tournaments
    GET  /tournaments/{id}          Tournament config + latest prediction
    GET  /tournaments/{id}/predict  Run prediction now (reads cached data)
    GET  /tournaments/{id}/history  Historical prediction log (JSONL)
    GET  /tournaments/{id}/entries  Current scraped entries
    POST /tournaments/{id}/scrape   Trigger a scrape now (async)
    GET  /summary                   All tournaments, latest predictions

Usage:
    uvicorn tools.api:app --reload --host 0.0.0.0 --port 8000

    Or directly:
    python tools/api.py
"""

import json
import math
import subprocess
import sys
import threading
import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config.json"
TMP = ROOT / ".tmp"
TMP.mkdir(exist_ok=True)

# ── Upcoming events cache (60-min TTL) ────────────────────────────────────────
_upcoming_cache: dict = {"data": None, "expires": 0.0}
_upcoming_lock = threading.Lock()

# ── Ensemble background compute tracker ───────────────────────────────────────
_ensemble_computing: dict[str, bool] = {}
_ensemble_lock = threading.Lock()

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Chess Tournament Entry Predictor",
    description="Double-sigmoid model for predicting chess tournament registration counts",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def get_tournament(tid: str) -> dict:
    cfg = load_config()
    for t in cfg["tournaments"]:
        if t["id"] == tid:
            return t
    raise HTTPException(status_code=404, detail=f"Tournament '{tid}' not found")


def load_json_file(path: Path) -> Any:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def double_sigmoid(t, L1, k1, m1, L2, k2, m2):
    def safe_sig(k, t, m):
        x = -k * (t - m)
        if x > 500: return 0.0
        if x < -500: return 1.0
        return 1.0 / (1.0 + math.exp(x))
    return L1 * safe_sig(k1, t, m1) + L2 * safe_sig(k2, t, m2)


def compute_live_prediction(tournament: dict) -> dict | None:
    """
    Compute prediction on-the-fly from cached entries + params.
    Returns prediction dict or None if not enough data.
    """
    tid = tournament["id"]
    params = load_json_file(TMP / f"params_{tid}.json")
    entries_data = load_json_file(TMP / f"entries_{tid}.json")

    if not params:
        params = tournament.get("prior_params")
    if not params:
        return None

    actual_count = 0
    if entries_data:
        actual_count = entries_data.get("count", len(entries_data.get("entries", [])))

    today = date.today()
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
    if delta_pct > threshold: status = "GREEN"
    elif delta_pct < -threshold: status = "RED"
    else: status = "YELLOW"

    margin = max(predicted_final * 0.10, 2 * rmse) if rmse else predicted_final * 0.15

    return {
        "tournament": tournament["name"],
        "tournament_id": tid,
        "as_of": today.isoformat(),
        "t_today": round(t_today, 1),
        "t_final": round(t_final, 1),
        "actual_count": actual_count,
        "predicted_now": round(predicted_now, 1),
        "delta_pct": round(delta_pct, 1),
        "status": status,
        "predicted_final": round(predicted_final, 1),
        "ci_low": round(predicted_final - margin, 1),
        "ci_high": round(predicted_final + margin, 1),
        "params": {k: params[k] for k in ["L1","k1","m1","L2","k2","m2"] if k in params},
        "r2": params.get("r2"),
        "rmse": rmse,
    }


def generate_curve_points(params: dict, t_start: int, t_end: int, steps: int = 200) -> list[dict]:
    """Generate sigmoid curve points for charting."""
    L1, k1, m1 = params["L1"], params["k1"], params["m1"]
    L2, k2, m2 = params["L2"], params["k2"], params["m2"]
    step = (t_end - t_start) / steps
    points = []
    for i in range(steps + 1):
        t = t_start + i * step
        y = double_sigmoid(t, L1, k1, m1, L2, k2, m2)
        points.append({"t": round(t, 1), "y": round(y, 2)})
    return points


def _live_entry_count_one(url: str) -> int:
    """
    Fetch current entry count from a single onlineregistration.cc URL.

    Follows the two-step pattern:
      1. advance_entry_list.php loads JS that references a static advlists/ HTML file
      2. The static file contains the actual player table

    Returns -1 on any error.
    """
    import re as _re
    try:
        import requests as _req
        from bs4 import BeautifulSoup as _BS
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        base_url = "https://onlineregistration.cc/tournaments"

        resp = _req.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        match = _re.search(r"advlists/([^'\"]+_alp_n\.html)", resp.text)
        if not match:
            return 0

        advlist_url = f"{base_url}/{match.group(0)}"
        resp2 = _req.get(advlist_url, headers=headers, timeout=15)
        resp2.raise_for_status()
        soup = _BS(resp2.text, "html.parser")
        tables = soup.find_all("table")
        if not tables:
            return 0

        main_table = max(tables, key=lambda t: len(t.find_all("tr")))
        rows = main_table.find_all("tr")
        count = 0
        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            if not cells:
                continue
            text = row.get_text(strip=True).lower()
            if "player name" in text or text in ("name", "#"):
                continue
            if any(c.get_text(strip=True) for c in cells):
                count += 1
        return count
    except Exception:
        return -1


def _live_entry_count(url: str, tournament: dict | None = None) -> int:
    """
    Fetch current total entry count for a tournament.

    If the tournament config has a 'sections' array, sums counts across all
    section URLs. Otherwise falls back to the single provided URL.
    Returns -1 if all fetches fail.
    """
    sections = (tournament or {}).get("sections", [])
    if sections:
        urls = [s["registration_url"] for s in sections if s.get("registration_url")]
    else:
        urls = [url] if url else []

    if not urls:
        return -1

    total = 0
    any_success = False
    for u in urls:
        n = _live_entry_count_one(u)
        if n >= 0:
            total += n
            any_success = True
    return total if any_success else -1


def run_scrape_background(tournament: dict):
    """Background task: run the full pipeline for one tournament."""
    pipeline = ROOT / "tools" / "run_pipeline.py"
    subprocess.Popen(
        [sys.executable, str(pipeline), "--id", tournament["id"], "--no-alert"],
        cwd=str(ROOT),
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "Chess Tournament Entry Predictor",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/tournaments")
def list_tournaments():
    cfg = load_config()
    result = []
    for t in cfg["tournaments"]:
        result.append({
            "id": t["id"],
            "name": t["name"],
            "city": t.get("city"),
            "tournament_date": t["tournament_date"],
            "active": t.get("active", True),
            "expected_total": t.get("expected_total"),
        })
    return {"tournaments": result}


@app.get("/tournaments/{tid}")
def get_tournament_detail(tid: str):
    t = get_tournament(tid)
    prediction = compute_live_prediction(t)
    saved_prediction = load_json_file(TMP / f"prediction_{tid}.json")

    return {
        "tournament": t,
        "prediction": prediction or saved_prediction,
        "data_available": (TMP / f"entries_{tid}.json").exists(),
        "last_scraped": (
            load_json_file(TMP / f"entries_{tid}.json") or {}
        ).get("scraped_at"),
    }


@app.get("/tournaments/{tid}/predict")
def predict(tid: str):
    t = get_tournament(tid)
    prediction = compute_live_prediction(t)
    if not prediction:
        raise HTTPException(
            status_code=422,
            detail="Cannot compute prediction: no parameters available. Run pipeline first."
        )

    # Also include curve points for charting
    params = prediction.get("params", {})
    if params and all(k in params for k in ["L1","k1","m1","L2","k2","m2"]):
        first_reg = date.fromisoformat(t["first_reg_date"])
        tourney_date = date.fromisoformat(t["tournament_date"])
        t_final = (tourney_date - first_reg).days
        curve = generate_curve_points(params, 0, t_final)
        prediction["curve"] = curve

    return prediction


@app.get("/tournaments/{tid}/history")
def history(tid: str, limit: int = 100):
    get_tournament(tid)  # Validate exists
    history_path = TMP / f"history_{tid}.jsonl"
    if not history_path.exists():
        return {"history": []}

    lines = history_path.read_text().strip().split("\n")
    entries = []
    for line in lines:
        if line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Return most recent `limit` entries, newest first
    entries.reverse()
    return {"history": entries[:limit], "total": len(entries)}


@app.get("/tournaments/{tid}/entries")
def entries(tid: str):
    get_tournament(tid)
    data = load_json_file(TMP / f"entries_{tid}.json")
    if not data:
        raise HTTPException(
            status_code=404,
            detail="No entry data found. Run pipeline or trigger scrape."
        )
    return data


@app.get("/tournaments/{tid}/cumulative")
def cumulative(tid: str):
    """Return [[t, cumulative_count], ...] data points for charting."""
    get_tournament(tid)
    data = load_json_file(TMP / f"cumulative_{tid}.json")
    if not data:
        return {"data": [], "count": 0}
    return {"data": data, "count": len(data)}


def _run_ensemble_background(tid: str, cumulative_data: list, tournament_cfg: dict):
    """Compute ensemble in a background thread and save to cache file."""
    try:
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "tools"))
        from model_ensemble import run_ensemble
        run_ensemble(tid, cumulative_data, tournament_cfg=tournament_cfg, verbose=False)
    except Exception:
        pass
    finally:
        with _ensemble_lock:
            _ensemble_computing.pop(tid, None)


@app.get("/tournaments/{tid}/ensemble")
def ensemble(tid: str, refresh: bool = False, background_tasks: BackgroundTasks = None):
    """
    Multi-model ensemble prediction: Single Sigmoid, Double Sigmoid,
    Monte Carlo, Bayesian MCMC, and Thompson Sampling — combined.

    Returns cached result immediately. If no cache exists (or refresh=true),
    triggers background computation and returns {"status": "computing"}.
    Poll again in ~15s once computation completes.
    """
    t = get_tournament(tid)
    cache_path = TMP / f"ensemble_{tid}.json"

    if not refresh:
        cached = load_json_file(cache_path)
        if cached:
            cached["_cached"] = True
            return cached

    cumulative_data = load_json_file(TMP / f"cumulative_{tid}.json")
    if not cumulative_data:
        raise HTTPException(
            status_code=422,
            detail="No cumulative data. Run pipeline first: python tools/run_pipeline.py"
        )

    with _ensemble_lock:
        already_computing = _ensemble_computing.get(tid, False)
        if not already_computing:
            _ensemble_computing[tid] = True

    if not already_computing:
        if background_tasks is not None:
            background_tasks.add_task(_run_ensemble_background, tid, cumulative_data, t)
        else:
            thread = threading.Thread(
                target=_run_ensemble_background, args=(tid, cumulative_data, t), daemon=True
            )
            thread.start()

    return {"status": "computing", "tournament_id": tid, "check_in_seconds": 15, "_cached": False}


@app.post("/tournaments/{tid}/scrape")
def trigger_scrape(tid: str, background_tasks: BackgroundTasks):
    t = get_tournament(tid)
    if not t.get("registration_url"):
        raise HTTPException(
            status_code=422,
            detail="No registration_url configured for this tournament."
        )
    background_tasks.add_task(run_scrape_background, t)
    return {
        "status": "queued",
        "message": f"Scrape triggered for {t['name']}. Check /tournaments/{tid} in ~30s.",
    }


@app.get("/summary")
def summary():
    """Latest prediction for all active tournaments + summary stats."""
    cfg = load_config()
    results = []
    for t in cfg["tournaments"]:
        if not t.get("active", True):
            continue
        prediction = compute_live_prediction(t)
        if not prediction:
            prediction = load_json_file(TMP / f"prediction_{t['id']}.json")
        results.append({
            "id": t["id"],
            "name": t["name"],
            "tournament_date": t["tournament_date"],
            "prediction": prediction,
        })

    # Load from summary.json if pipeline has been run
    pipeline_summary = load_json_file(TMP / "summary.json")

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "tournaments": results,
        "last_pipeline_run": pipeline_summary.get("generated_at") if pipeline_summary else None,
    }


@app.get("/chart/{tid}")
def chart_data(tid: str):
    """
    Returns everything the dashboard needs to render the chart:
    - actual cumulative data points
    - model curve
    - CI band
    - fee deadline markers
    - current prediction
    """
    t = get_tournament(tid)
    params_data = load_json_file(TMP / f"params_{tid}.json") or t.get("prior_params", {})
    cumulative_data = load_json_file(TMP / f"cumulative_{tid}.json") or []
    prediction = compute_live_prediction(t)

    first_reg = date.fromisoformat(t["first_reg_date"])
    tourney_date = date.fromisoformat(t["tournament_date"])
    t_final = (tourney_date - first_reg).days

    curve = []
    ci_band = []
    if params_data and all(k in params_data for k in ["L1","k1","m1","L2","k2","m2"]):
        curve = generate_curve_points(params_data, 0, t_final)
        rmse = params_data.get("rmse", 0)
        for pt in curve:
            margin = max(pt["y"] * 0.08, 2 * rmse) if rmse else pt["y"] * 0.12
            ci_band.append({
                "t": pt["t"],
                "low": round(max(0, pt["y"] - margin), 2),
                "high": round(pt["y"] + margin, 2),
            })

    # Fee deadlines as t-values
    deadlines = []
    for dl in t.get("fee_deadlines", []):
        dl_date = date.fromisoformat(dl["date"])
        t_dl = (dl_date - first_reg).days
        deadlines.append({**dl, "t": t_dl})

    return {
        "tournament_id": tid,
        "tournament_name": t["name"],
        "t_final": t_final,
        "actual_data": [{"t": d[0], "y": d[1]} for d in cumulative_data],
        "curve": curve,
        "ci_band": ci_band,
        "fee_deadlines": deadlines,
        "prediction": prediction,
        "params": params_data,
    }


@app.get("/upcoming")
def upcoming():
    """
    All upcoming CCA tournaments with live entry counts and predictions.

    Priority order:
    1. Return .tmp/upcoming_tournaments.json if it exists (populated by scrape_upcoming.py)
    2. Fall back to config.json active tournaments with live-fetched counts
    Results from config fallback are cached 15 minutes.
    """
    # Prefer scraped upcoming data (run: python tools/scrape_upcoming.py)
    scraped_path = TMP / "upcoming_tournaments.json"
    if scraped_path.exists():
        data = load_json_file(scraped_path)
        if data and data.get("tournaments"):
            today = date.today().isoformat()
            # Filter to future tournaments only
            future = [
                t for t in data["tournaments"]
                if not t.get("tournament_date") or t["tournament_date"] >= today
            ]
            return {
                "generated_at": data.get("generated_at"),
                "scraped_at": data.get("generated_at"),
                "source": "scraped",
                "cached": True,
                "count": len(future),
                "tournaments": future,
            }

    # Fallback: config.json active tournaments
    now = _time.time()

    with _upcoming_lock:
        if _upcoming_cache["data"] is not None and now < _upcoming_cache["expires"]:
            return dict(_upcoming_cache["data"], cached=True)

    cfg = load_config()
    today_iso = date.today().isoformat()

    active = [
        t for t in cfg["tournaments"]
        if t.get("active", True) and t.get("tournament_date", "9999") >= today_iso
    ]

    # Fetch all live entry counts in parallel (max 5 concurrent)
    # Key by tournament id — multi-section tournaments sum across all section URLs
    tid_to_count: dict[str, int] = {}
    countable = [t for t in active if t.get("registration_url") or t.get("sections")]
    if countable:
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_map = {
                executor.submit(_live_entry_count, t.get("registration_url"), t): t["id"]
                for t in countable
            }
            try:
                for future in as_completed(future_map, timeout=60):
                    try:
                        tid_to_count[future_map[future]] = future.result()
                    except Exception as e:
                        import logging
                        logging.warning("Entry count fetch failed for %s: %s", future_map[future], e)
            except TimeoutError:
                pass  # partial results are acceptable on timeout

    results = []
    for t in active:
        entry_count = tid_to_count.get(t["id"])
        prediction = compute_live_prediction(t)
        results.append({
            "id": t["id"],
            "name": t["name"],
            "location": t.get("city"),
            "tournament_date": t["tournament_date"],
            "first_reg_date": t.get("first_reg_date"),
            "registration_url": t.get("registration_url"),
            "entry_count": entry_count,
            "expected_total": t.get("expected_total"),
            "fee_deadlines": t.get("fee_deadlines", []),
            "prior_params": t.get("prior_params"),
            "prediction": prediction,
        })

    response = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source": "config",
        "cached": False,
        "count": len(results),
        "tournaments": results,
    }

    with _upcoming_lock:
        _upcoming_cache["data"] = response
        _upcoming_cache["expires"] = now + 60 * 60  # 60 minutes

    return response


@app.post("/upcoming/refresh")
def refresh_upcoming():
    """
    Force-clear the upcoming events cache so next /upcoming call re-scrapes.
    Also triggers a background scrape via scrape_upcoming.py.
    """
    with _upcoming_lock:
        _upcoming_cache["data"] = None
        _upcoming_cache["expires"] = 0.0

    # Trigger background scrape
    scraper = ROOT / "tools" / "scrape_upcoming.py"
    if scraper.exists():
        import subprocess as _sp
        _sp.Popen(
            [sys.executable, str(scraper), "--no-counts"],
            cwd=str(ROOT),
        )
        return {"status": "cache cleared, background scrape triggered"}

    return {"status": "cache cleared"}


@app.get("/families/{family}/overlay")
def family_overlay(family: str):
    """
    Returns same-family historical tournament data normalized to registration-cycle %.

    Each tournament in the family is returned as a trace with:
      - x: fraction of registration window elapsed (0.0 = first_reg_date, 1.0 = tournament_date)
      - y: cumulative entry count (absolute)
      - y_pct: cumulative entry count as fraction of that tournament's final count (0.0–1.0)

    This drives the same-family overlay chart: plot thin historical traces behind the current
    tournament's prediction curve so the reader can see typical vs. anomalous trajectories.

    Only same-family tournaments are returned — cross-family comparisons belong at the
    archetype level, not here.
    """
    cfg = load_config()
    members = [t for t in cfg["tournaments"] if t.get("tournament_family") == family]
    if not members:
        raise HTTPException(status_code=404, detail=f"No tournaments found in family '{family}'")

    traces = []
    for t in members:
        tid = t["id"]
        cumulative_data = load_json_file(TMP / f"cumulative_{tid}.json")
        if not cumulative_data:
            continue

        first_reg = date.fromisoformat(t["first_reg_date"])
        tourney_date = date.fromisoformat(t["tournament_date"])
        total_days = (tourney_date - first_reg).days
        if total_days <= 0:
            continue

        final_count = cumulative_data[-1][1] if cumulative_data else t.get("expected_total", 0)
        if not final_count:
            final_count = t.get("expected_total", 1) or 1

        points = []
        for day_offset, count in cumulative_data:
            x_pct = round(day_offset / total_days, 4)
            y_pct = round(count / final_count, 4)
            points.append({"x_pct": x_pct, "y": count, "y_pct": y_pct})

        traces.append({
            "tournament_id": tid,
            "tournament_name": t["name"],
            "tournament_family": family,
            "tournament_date": t["tournament_date"],
            "first_reg_date": t["first_reg_date"],
            "total_reg_days": total_days,
            "final_count": final_count,
            "expected_total": t.get("expected_total"),
            "active": t.get("active", True),
            "points": points,
        })

    traces.sort(key=lambda x: x["tournament_date"])
    return {
        "family": family,
        "member_count": len(traces),
        "traces": traces,
        "_note": (
            "Use x_pct as the normalized x-axis (0=open, 1=tournament day). "
            "Scale y by target tournament's projected final to overlay traces on the same chart."
        ),
    }


@app.post("/upcoming/scrape")
def scrape_upcoming_now(background_tasks: BackgroundTasks, counts: bool = True):
    """
    Trigger a full upcoming tournament scrape (including live entry counts).
    Use counts=false to skip live entry fetching for a faster run.
    """
    def _run_scrape():
        scraper = ROOT / "tools" / "scrape_upcoming.py"
        cmd = [sys.executable, str(scraper)]
        if not counts:
            cmd.append("--no-counts")
        subprocess.Popen(cmd, cwd=str(ROOT))

    background_tasks.add_task(_run_scrape)
    return {
        "status": "queued",
        "message": "Scrape triggered. Check /upcoming in ~60s.",
        "counts": counts,
    }


# ── Dev runner ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    cfg = load_config()
    settings = cfg.get("settings", {})
    uvicorn.run(
        "tools.api:app",
        host=settings.get("api_host", "0.0.0.0"),
        port=settings.get("api_port", 8000),
        reload=True,
    )
