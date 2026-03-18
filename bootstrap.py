#!/usr/bin/env python3
"""
bootstrap.py
------------
One-click setup script for the CCA Entry Predictor.
Seeds .tmp/ with all data and verifies the full pipeline end-to-end.

Usage:
    python bootstrap.py                  # full setup
    python bootstrap.py --skip-excel     # skip Excel import (workbook not found)
    python bootstrap.py --skip-upcoming  # skip upcoming tournament scrape
    python bootstrap.py --skip-deps      # skip pip install (already installed)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
TMP = ROOT / ".tmp"
TMP.mkdir(exist_ok=True)

EXCEL_FILENAME = "CCA Attendance predictor working-IK-3 revised.xlsx"
EXCEL_SEARCH_PATHS = [
    Path.home() / "Downloads" / EXCEL_FILENAME,
    Path.home() / "Downloads" / EXCEL_FILENAME.replace(".xlsx", " (1).xlsx"),
    Path.home() / "OneDrive" / "Downloads" / EXCEL_FILENAME,
    Path.home() / "Desktop" / EXCEL_FILENAME,
    ROOT / EXCEL_FILENAME,
]


# ── Output helpers ────────────────────────────────────────────────────────────

def ok(msg):   print(f"  [OK]  {msg}")
def warn(msg): print(f"  [!!]  {msg}")
def err(msg):  print(f"  [ERR] {msg}", file=sys.stderr)
def step(msg): print(f"\n>> {msg}")


def run_cmd(cmd: list, label: str) -> bool:
    """Run a subprocess, stream output, return True on success."""
    print(f"     $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        err(f"{label} exited with code {result.returncode}")
        return False
    ok(label)
    return True


def find_excel() -> Path | None:
    for p in EXCEL_SEARCH_PATHS:
        if p.exists():
            return p
    return None


# ── Step 1: Install dependencies ──────────────────────────────────────────────

def install_deps() -> bool:
    step("Installing Python dependencies")
    req = ROOT / "requirements.api.txt"
    if not req.exists():
        req = ROOT / "requirements.txt"
    return run_cmd(
        [sys.executable, "-m", "pip", "install", "-r", str(req), "-q"],
        "pip install requirements"
    )


# ── Step 2: Import Excel workbook (WO2026 entries + params) ──────────────────

def import_excel(skip: bool) -> bool:
    step("Importing World Open 2026 data from Excel workbook")
    if skip:
        warn("--skip-excel: skipping Excel import")
        return True

    excel = find_excel()
    if not excel:
        warn(f"Workbook not found. Searched:")
        for p in EXCEL_SEARCH_PATHS:
            print(f"       {p}")
        warn("Continuing without Excel data — pipeline will use prior params from config.json")
        warn("To import later:  python tools/parse_excel_workbook.py --help")
        return True

    ok(f"Found: {excel.name}")
    ok(f"  at:  {excel}")

    return run_cmd([
        sys.executable, "tools/parse_excel_workbook.py",
        "--file",           str(excel),
        "--tournament",     "World Open 2026",
        "--first-reg-date", "2025-09-01",
        "--out-entries",    str(TMP / "entries_WO2026.json"),
        "--out-data",       str(TMP / "cumulative_WO2026_excel.json"),
        "--out-params",     str(TMP / "params_WO2026_excel.json"),
    ], "parse_excel_workbook → WO2026")


# ── Step 3: Build cumulative training data ────────────────────────────────────

def prepare_training() -> bool:
    step("Building cumulative training data (CHI2022 + NAO2024)")
    print("  Note: CSVs must be in paths defined in tools/prepare_training_data.py")
    print("  Missing CSVs are skipped automatically — not a fatal error.")
    return run_cmd(
        [sys.executable, "tools/prepare_training_data.py"],
        "prepare_training_data"
    )


# ── Step 4: Run prediction pipeline for WO2026 ───────────────────────────────

def run_pipeline() -> bool:
    step("Running prediction pipeline for WO2026")
    print("  Using --no-scrape (loads Excel entries if imported, else uses prior params)")
    return run_cmd(
        [sys.executable, "tools/run_pipeline.py",
         "--id", "WO2026", "--no-scrape", "--no-alert"],
        "run_pipeline WO2026"
    )


# ── Step 5: Compute ensemble ──────────────────────────────────────────────────

def run_ensemble() -> bool:
    step("Computing multi-model ensemble for WO2026")
    cum = TMP / "cumulative_WO2026.json"
    if not cum.exists():
        warn("No cumulative data yet — ensemble will run on prior params only")
    return run_cmd(
        [sys.executable, "tools/model_ensemble.py", "--id", "WO2026"],
        "model_ensemble WO2026"
    )


# ── Step 6: Scrape upcoming tournaments ──────────────────────────────────────

def scrape_upcoming(skip: bool) -> bool:
    step("Fetching upcoming CCA tournament list")
    if skip:
        warn("--skip-upcoming: skipping")
        return True
    print("  Using --no-counts for speed (no per-tournament entry fetch)")
    print("  To fetch live counts later: python tools/scrape_upcoming.py")
    return run_cmd(
        [sys.executable, "tools/scrape_upcoming.py", "--no-counts"],
        "scrape_upcoming"
    )


# ── Step 7: Run backtest on historical data ───────────────────────────────────

def run_backtest() -> bool:
    step("Running ensemble backtest on historical data")
    chi = TMP / "cumulative_CHI2022.json"
    nao = TMP / "cumulative_NAO2024.json"
    if not chi.exists() and not nao.exists():
        warn("No historical cumulative data — skipping backtest")
        warn("Run prepare_training_data.py with the CSV files to enable backtesting")
        return True
    return run_cmd(
        [sys.executable, "tools/model_ensemble.py", "--backtest"],
        "model_ensemble --backtest"
    )


# ── Verify output ──────────────────────────────────────────────────────────────

def verify() -> bool:
    step("Verifying pipeline outputs")
    required = {
        "prediction_WO2026.json": "Live prediction",
        "params_WO2026.json":     "Model parameters",
        "cumulative_WO2026.json": "Cumulative registration curve",
        "summary.json":           "Pipeline summary",
    }
    optional = {
        "ensemble_WO2026.json":       "Ensemble model",
        "entries_WO2026.json":        "Entry list (from Excel)",
        "upcoming_tournaments.json":  "Upcoming CCA tournaments",
        "cumulative_CHI2022.json":    "CHI2022 training data",
        "cumulative_NAO2024.json":    "NAO2024 training data",
        "training_meta.json":         "Training metadata",
    }

    all_required = True
    for fname, label in required.items():
        p = TMP / fname
        if p.exists():
            ok(f"{fname}  ({label})")
        else:
            warn(f"MISSING: {fname}  ({label})")
            all_required = False

    print()
    for fname, label in optional.items():
        p = TMP / fname
        status = "ok  " if p.exists() else "skip"
        print(f"     [{status}] {fname}  ({label})")

    return all_required


# ── Print final summary ────────────────────────────────────────────────────────

def print_summary():
    pred_path = TMP / "prediction_WO2026.json"
    ensemble_path = TMP / "ensemble_WO2026.json"

    print(f"\n{'='*62}")
    print("  BOOTSTRAP COMPLETE")
    print(f"{'='*62}")

    if pred_path.exists():
        with open(pred_path) as f:
            p = json.load(f)
        status_colors = {"GREEN": "↑", "YELLOW": "→", "RED": "↓"}
        arrow = status_colors.get(p.get("status", ""), "?")

        print(f"""
  World Open 2026 — Current Prediction
  ─────────────────────────────────────
  Actual entries:    {p.get('actual_count', '?')}
  Model expects now: {p.get('predicted_now', 0):.1f}
  Delta:             {p.get('delta_pct', 0):+.1f}%  {arrow} {p.get('status', '?')}
  Final projection:  {p.get('predicted_final', 0):.0f}  (95% CI: {p.get('ci_low', 0):.0f}–{p.get('ci_high', 0):.0f})
  Model source:      {p.get('r2', 'N/A')} R²""")

    if ensemble_path.exists():
        with open(ensemble_path) as f:
            e = json.load(f)
        ens = e.get("ensemble", {})
        print(f"  Ensemble estimate: {ens.get('predicted_final', '?')}")

    print(f"""
  NEXT STEPS
  ──────────
  1. Open chess_predictor.html — dashboard works now in static mode

  2. Start API (for live data):
       python tools/api.py
     Then open http://localhost:8000 and reload chess_predictor.html

  3. Schedule scrapes every 6h:
       Double-click  schedule_scrape.bat

  4. Deploy to Railway (public URL):
       See DEPLOY.md for step-by-step instructions

  5. Configure alerts (optional):
       Edit .env  →  add SENDGRID_API_KEY and TWILIO keys
       Edit config.json  →  set alert_email and alert_sms
{'='*62}""")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bootstrap CCA Entry Predictor")
    parser.add_argument("--skip-excel",    action="store_true", help="Skip Excel import")
    parser.add_argument("--skip-upcoming", action="store_true", help="Skip upcoming tournament scrape")
    parser.add_argument("--skip-deps",     action="store_true", help="Skip pip install")
    parser.add_argument("--skip-backtest", action="store_true", help="Skip historical backtest")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════╗")
    print("║   CCA Entry Predictor — Bootstrap        ║")
    print("╚══════════════════════════════════════════╝")
    print(f"  Working directory: {ROOT}")
    print(f"  Python:            {sys.executable}")

    pipeline = [
        ("install_deps",     lambda: install_deps()              if not args.skip_deps     else (warn("--skip-deps") or True)),
        ("import_excel",     lambda: import_excel(args.skip_excel)),
        ("prepare_training", lambda: prepare_training()),
        ("run_pipeline",     lambda: run_pipeline()),
        ("run_ensemble",     lambda: run_ensemble()),
        ("scrape_upcoming",  lambda: scrape_upcoming(args.skip_upcoming)),
        ("run_backtest",     lambda: run_backtest()              if not args.skip_backtest else (warn("--skip-backtest") or True)),
        ("verify",           lambda: verify()),
    ]

    for name, fn in pipeline:
        success = fn()
        if not success:
            print(f"\n  [!] Step '{name}' failed.")
            print("      Fix the error above, then re-run bootstrap.py")
            print("      Use --skip-* flags to bypass completed steps.")
            print_summary()
            sys.exit(1)

    print_summary()


if __name__ == "__main__":
    main()
