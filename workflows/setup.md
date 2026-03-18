# Workflow: System Setup

## Objective
Get the Chess Tournament Entry Prediction system running from scratch on a new machine.

---

## Prerequisites

- Python 3.11+
- Git
- PostgreSQL 14+ (optional for Phase 2; not needed for the HTML dashboard)
- A text editor for `.env`

---

## Step 1 — Install Python dependencies

```bash
cd Initial/
pip install -r requirements.txt
```

For a lightweight start (dashboard only, no database):
```bash
pip install requests beautifulsoup4 lxml numpy scipy openpyxl python-dotenv
```

---

## Step 2 — Configure `.env`

Edit `.env` with your credentials. Only fill in what you need:

```env
# Alerting (optional — leave blank to skip)
SENDGRID_API_KEY=SG.xxxx
ALERT_EMAIL_FROM=you@example.com
ALERT_EMAIL_TO=director@chess.org

TWILIO_ACCOUNT_SID=ACxxxx
TWILIO_AUTH_TOKEN=xxxx
TWILIO_FROM_NUMBER=+15551234567
ALERT_SMS_TO=+15559876543

# Database (Phase 2+ only)
DATABASE_URL=postgresql://localhost:5432/chess_predictor
```

---

## Step 3 — Open the dashboard (no server needed)

Open `chess_predictor.html` directly in a browser. It runs entirely client-side:
- World Open 2026: pre-fitted parameters from the brief, 20 entries as of 2026-03-09 (expected final: ~1,100–1,400 across all 8 sections)
- Chicago Open 2022: 943-entry historical dataset, fitted on page load (~2s)
- All charts, sliders, and delta display work offline

---

## Step 4 — Run the prediction pipeline

```bash
# Dry run (uses cached data, no scraping)
python tools/run_pipeline.py --dry-run

# Run for World Open only (no scraping, use cached entries)
python tools/run_pipeline.py --id WO2026 --no-scrape

# Full live run (requires registration URL to be reachable)
python tools/run_pipeline.py --id WO2026
```

Output files in `.tmp/`:
- `entries_WO2026.json` — raw scraped entries
- `cumulative_WO2026.json` — `[[t, count], ...]` daily data
- `params_WO2026.json` — fitted model parameters
- `prediction_WO2026.json` — full prediction with delta + CI
- `history_WO2026.jsonl` — every prediction run (append-only)
- `summary.json` — all tournaments latest predictions

---

## Step 5 — Parse the Excel workbook

To extract World Open 2026 data from the CCA workbook:

```bash
# First, inspect sheet names
python tools/parse_excel_workbook.py \
  --file "path/to/CCA Attendance predictor working-IK-3 revised.xlsx" \
  --dump-sheets

# Then extract all data
python tools/parse_excel_workbook.py \
  --file "path/to/CCA Attendance predictor working-IK-3 revised.xlsx" \
  --tournament "World Open 2026" \
  --first-reg-date 2026-02-05 \
  --out-data .tmp/cumulative_WO2026.json \
  --out-params .tmp/params_WO2026.json \
  --out-entries .tmp/entries_WO2026.json
```

---

## Step 6 — Start the API server (optional, enables live dashboard)

```bash
uvicorn tools.api:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: `http://localhost:8000/docs`

Key endpoints:
- `GET /summary` — all tournaments, latest predictions
- `GET /tournaments/WO2026/predict` — live prediction + chart curve
- `GET /chart/WO2026` — all data needed to render the chart
- `POST /tournaments/WO2026/scrape` — trigger live scrape (async)

The HTML dashboard auto-detects the API on port 8000 and switches from static to live mode.

---

## Step 7 — Set up the database (Phase 2)

```bash
# Create the database
createdb chess_predictor

# Run schema
psql chess_predictor < tools/schema.sql
```

---

## Step 8 — Schedule recurring scrapes

**Windows Task Scheduler** (every 6 hours):
```
Action: python C:\path\to\Initial\tools\run_pipeline.py
Schedule: Every 6 hours
```

**Linux/Mac cron** (every 6 hours):
```
0 */6 * * * cd /path/to/Initial && python tools/run_pipeline.py >> .tmp/cron.log 2>&1
```

---

## Finding Tournament Info

To configure a new CCA tournament:
- **Full schedule:** https://www.chesstour.com/refs.html
- **Tournament detail pages:** https://www.chesstour.com/`<code>`.htm (e.g., `chio26.htm`, `wo26.htm`)
- **Registration portal:** https://chessaction.com — find the event, click "Advance Entry List" to get the `tid`
- **Historical results:** https://chessevents.com — search by event name for prior-year entry counts

**Key fee deadline dates for 2026 events (early bird):**
- Chicago Open 2026: March 19 (→ $207 becomes $227)
- World Open 2026: ~May 15 estimate (check chesstour.com/wo26.htm when posted)

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: scipy` | `pip install scipy numpy` |
| `ModuleNotFoundError: bs4` | `pip install beautifulsoup4` |
| Scraper returns 0 entries | Check that `registration_url` in `config.json` is correct |
| Fit fails to converge | Normal at <10 data points — using `prior_params` from config |
| API 404 on `/predict` | Run pipeline first: `python tools/run_pipeline.py --id WO2026 --no-scrape` |
| Dashboard shows static data | API not running — open `chess_predictor.html` and start `uvicorn tools.api:app` |

---

## File Reference

```
chess_predictor.html          # Single-file HTML dashboard (open in browser)
config.json                   # Tournament configuration + model priors
requirements.txt              # Python dependencies

tools/
  run_pipeline.py             # Main orchestrator (scrape → fit → predict → alert)
  scrape_entries.py           # HTML scraper for onlineregistration.cc
  fit_double_sigmoid.py       # scipy/gradient-descent model fitter
  compute_delta.py            # Delta % and GREEN/YELLOW/RED status
  send_alert.py               # Email (SendGrid) + SMS (Twilio) alerts
  parse_excel_workbook.py     # Extract data from CCA Excel workbook
  api.py                      # FastAPI backend for live dashboard
  schema.sql                  # PostgreSQL database schema

workflows/
  predict_entries.md          # SOP: full prediction pipeline
  setup.md                    # This file
```
