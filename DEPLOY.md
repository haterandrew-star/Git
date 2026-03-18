# Deployment Guide — CCA Entry Predictor

## Overview

The app has two deployable pieces:
- **API** (`tools/api.py`) — FastAPI backend, deploy to Railway (~$5/mo hobby plan)
- **Dashboard** (`chess_predictor.html`) — static single-file HTML, serve anywhere

---

## Part 1: Deploy API to Railway

### Prerequisites
- GitHub account (Railway deploys from a repo)
- Railway account at [railway.app](https://railway.app) (free trial, then ~$5/mo)

### Steps

**1. Push the project to GitHub**
```bash
git init
git add .
git commit -m "initial"
git remote add origin https://github.com/YOUR_USERNAME/chess-predictor.git
git push -u origin main
```
Make sure `.gitignore` excludes `.env` and `.tmp/` (already set).

**2. Create a Railway project**
- Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub repo
- Select your repo
- Railway auto-detects `nixpacks.toml` and runs: `uvicorn tools.api:app --host 0.0.0.0 --port $PORT`

**3. Add environment variables in Railway**
Go to your Railway project → Variables tab → Add each from your `.env`:

| Variable | Value |
|----------|-------|
| `SENDGRID_API_KEY` | your SendGrid key |
| `ALERT_EMAIL_FROM` | alerts@yourdomain.com |
| `ALERT_EMAIL_TO` | director@yourdomain.com |
| `TWILIO_ACCOUNT_SID` | your Twilio SID |
| `TWILIO_AUTH_TOKEN` | your Twilio token |
| `TWILIO_FROM_NUMBER` | your Twilio number |
| `ALERT_SMS_TO` | your cell number |

Railway automatically injects `DATABASE_URL` if you add a Postgres plugin (optional for Phase 2).

**4. Get your Railway URL**
Settings tab → Domains → copy the URL (looks like `https://chess-predictor-production.up.railway.app`)

**5. Update the dashboard HTML with your Railway URL**
Open `chess_predictor.html`, find line ~1150:
```javascript
const RAILWAY_URL = 'https://YOUR-RAILWAY-URL.up.railway.app';
```
Replace with your actual URL, then redeploy or re-share the HTML file.

**6. Verify deployment**
```
https://your-url.up.railway.app/
# → {"status":"ok","service":"Chess Tournament Entry Predictor",...}

https://your-url.up.railway.app/tournaments
# → list of tournaments

https://your-url.up.railway.app/summary
# → current predictions
```

---

## Part 2: Serve the Dashboard

The HTML file is completely self-contained. Options:

**Option A — GitHub Pages (free)**
1. Put `chess_predictor.html` at the root of a public GitHub repo (or in `docs/`)
2. Settings → Pages → deploy from main branch
3. Your dashboard is live at `https://username.github.io/repo-name/chess_predictor.html`

**Option B — Netlify Drop (30 seconds)**
1. Go to [app.netlify.com/drop](https://app.netlify.com/drop)
2. Drag and drop the `chess_predictor.html` file
3. Get a URL instantly (e.g. `https://random-name.netlify.app`)

**Option C — Share directly**
Just email the `chess_predictor.html` file. It works offline in static mode.
When API is up, open it in a browser and it auto-connects.

---

## Part 3: Set Up Scheduled Scrapes

### On Railway (recommended for production)
Railway has a built-in Cron service:
1. New Service → Cron Job
2. Schedule: `0 */6 * * *` (every 6 hours)
3. Command: `python tools/run_pipeline.py --id WO2026`
4. Uses same environment as the API service

### On your Windows machine (local backup)
Run `schedule_scrape.bat` as Administrator — registers a Windows Task Scheduler task that runs every 6 hours.

---

## Part 4: Add More Tournaments

Edit `config.json` — add an entry to the `tournaments` array:

```json
{
  "id": "CHI2026",
  "name": "Chicago Open 2026",
  "city": "Wheeling, IL",
  "tournament_family": "chicago_open",
  "tournament_date": "2026-05-22",
  "end_date": "2026-05-25",
  "first_reg_date": "2026-01-01",
  "registration_url": "https://onlineregistration.cc/tournaments/advance_entry_list.php?tid=<TID_FROM_CHESSACTION>",
  "expected_total": 950,
  "active": true,
  "has_early_bird": true,
  "fee_deadlines": [
    {"date": "2026-03-19", "label": "Early Bird ($207→$227)", "price_delta": 20},
    {"date": "2026-05-19", "label": "Standard→Late ($227→$250)", "price_delta": 23}
  ],
  "prior_params": null
}

NOTE: Get the TID from chessaction.com — find Chicago Open 2026 in the upcoming list,
click Advance Entry List, copy the tid= parameter from the URL.
```

- Set `prior_params: null` initially — the system fits from live data once 10+ entries exist
- Or copy WO2026's `prior_params` as a starting point and scale `expected_total`

---

## Phase 2: PostgreSQL (optional)

The schema is ready at `tools/schema.sql`. To enable:
1. Add a Postgres plugin in Railway → it injects `DATABASE_URL` automatically
2. Run `psql $DATABASE_URL < tools/schema.sql` to create tables
3. The API will use the DB for history and entries once connected

---

## Troubleshooting

**API returns 422 on /predict or /ensemble**
→ Run the pipeline first: `python tools/run_pipeline.py --id WO2026`

**Dashboard stuck in "static mode"**
→ Check that `RAILWAY_URL` in the HTML matches your Railway domain exactly (no trailing slash)

**scrape_upcoming returns empty list**
→ chessaction.com AJAX endpoint may have changed. Check `tools/scrape_upcoming.py` AJAX_URL.
→ Run with `--limit 5` to debug a small batch.

**Twilio/SendGrid not sending**
→ Verify env vars are set: `python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.environ.get('SENDGRID_API_KEY', 'NOT SET'))"`
→ Test manually: `python tools/send_alert.py --file .tmp/prediction_WO2026.json`
→ Add `--dry-run` flag to preview the message without sending
