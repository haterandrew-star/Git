"""
send_alert.py
-------------
Sends alerts (email and/or SMS) when a tournament's delta exceeds the threshold.

Reads alert config from .env (or environment variables).
Called by run_pipeline.py or the FastAPI backend when status flips to RED.

Environment variables required (add to .env):
    ALERT_EMAIL_FROM=your@email.com
    ALERT_EMAIL_TO=director@chess.org
    SENDGRID_API_KEY=SG.xxxx         # for email
    TWILIO_ACCOUNT_SID=ACxxxx        # for SMS
    TWILIO_AUTH_TOKEN=xxxx
    TWILIO_FROM_NUMBER=+15551234567
    ALERT_SMS_TO=+15559876543

Usage:
    python tools/send_alert.py \\
        --tournament "World Open 2026" \\
        --status RED \\
        --delta -14.2 \\
        --actual 240 \\
        --predicted 279.5 \\
        --predicted-final 277

    # Or pipe prediction JSON from run_pipeline.py:
    cat .tmp/prediction_WO2026.json | python tools/send_alert.py --stdin
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Load .env if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass  # Fall through to os.environ directly


# ── Message formatting ────────────────────────────────────────────────────────

def format_subject(prediction: dict) -> str:
    status = prediction["status"]
    name = prediction["tournament"]
    delta = prediction["delta_pct"]
    direction = "above" if delta > 0 else "below"
    emoji = {"GREEN": "✅", "YELLOW": "⚠️", "RED": "🔴"}.get(status, "")
    return f"{emoji} [{status}] {name} tracking {abs(delta):.1f}% {direction} model"


def format_body(prediction: dict) -> str:
    delta = prediction["delta_pct"]
    direction = "above" if delta > 0 else "below"
    status_msg = {
        "GREEN": "Registration is tracking ABOVE expectations — momentum is strong.",
        "YELLOW": "Registration is on track with model expectations.",
        "RED": "Registration is tracking BELOW expectations — consider action.",
    }.get(prediction["status"], "")

    return f"""
Chess Tournament Entry Prediction Alert
========================================

Tournament:  {prediction['tournament']}
As of:       {prediction['as_of']}

REGISTRATION STATUS: {prediction['status']}
{status_msg}

Actual registrations:    {prediction['actual_count']:,}
Model expected (today):  {prediction['predicted_now']:.0f}
Delta:                   {delta:+.1f}% {direction} model

Final count forecast:    {prediction['predicted_final']:.0f}
  (95% CI: {prediction['ci_low']:.0f} – {prediction['ci_high']:.0f})

Model parameters:
  L1={prediction['params'].get('L1')}, k1={prediction['params'].get('k1')}, m1={prediction['params'].get('m1')}
  L2={prediction['params'].get('L2')}, k2={prediction['params'].get('k2')}, m2={prediction['params'].get('m2')}
  R²={prediction.get('r2')}, RMSE={prediction.get('rmse')}

---
Chess Tournament Prediction System
Generated at {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
""".strip()


# ── Email via SendGrid ────────────────────────────────────────────────────────

def send_email(subject: str, body: str) -> bool:
    api_key = os.environ.get("SENDGRID_API_KEY")
    from_email = os.environ.get("ALERT_EMAIL_FROM")
    to_email = os.environ.get("ALERT_EMAIL_TO")

    if not all([api_key, from_email, to_email]):
        print("  [email] Skipped: SENDGRID_API_KEY / ALERT_EMAIL_FROM / ALERT_EMAIL_TO not set",
              file=sys.stderr)
        return False

    try:
        import sendgrid
        from sendgrid.helpers.mail import Mail

        sg = sendgrid.SendGridAPIClient(api_key)
        message = Mail(
            from_email=from_email,
            to_emails=to_email,
            subject=subject,
            plain_text_content=body,
        )
        response = sg.send(message)
        if response.status_code in (200, 202):
            print(f"  [email] Sent to {to_email} (status {response.status_code})")
            return True
        else:
            print(f"  [email] Unexpected status {response.status_code}", file=sys.stderr)
            return False

    except ImportError:
        print("  [email] sendgrid not installed. Run: pip install sendgrid", file=sys.stderr)
        return False
    except Exception as e:
        print(f"  [email] Failed: {e}", file=sys.stderr)
        return False


# ── SMS via Twilio ────────────────────────────────────────────────────────────

def send_sms(body: str) -> bool:
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
    from_number = os.environ.get("TWILIO_FROM_NUMBER")
    to_number = os.environ.get("ALERT_SMS_TO")

    if not all([account_sid, auth_token, from_number, to_number]):
        print("  [sms] Skipped: Twilio credentials not set", file=sys.stderr)
        return False

    try:
        from twilio.rest import Client
        client = Client(account_sid, auth_token)

        # SMS is limited to ~160 chars — send a condensed version
        sms_body = body[:300]

        message = client.messages.create(
            body=sms_body,
            from_=from_number,
            to=to_number,
        )
        print(f"  [sms] Sent to {to_number} (SID: {message.sid})")
        return True

    except ImportError:
        print("  [sms] twilio not installed. Run: pip install twilio", file=sys.stderr)
        return False
    except Exception as e:
        print(f"  [sms] Failed: {e}", file=sys.stderr)
        return False


# ── Log to file ───────────────────────────────────────────────────────────────

def log_alert(prediction: dict, channels_used: list[str]):
    log_path = Path(__file__).parent.parent / ".tmp" / "alerts.jsonl"
    log_path.parent.mkdir(exist_ok=True)
    record = {
        "sent_at": datetime.utcnow().isoformat() + "Z",
        "tournament": prediction["tournament"],
        "tournament_id": prediction["tournament_id"],
        "status": prediction["status"],
        "delta_pct": prediction["delta_pct"],
        "actual_count": prediction["actual_count"],
        "channels": channels_used,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"  [alert] Logged to {log_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Send tournament prediction alert")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stdin", action="store_true",
                       help="Read prediction JSON from stdin")
    group.add_argument("--file", help="Path to prediction JSON file")
    group.add_argument("--tournament", help="Tournament name (use with --status etc.)")

    # Manual fields (used when --tournament is specified instead of JSON)
    parser.add_argument("--status", choices=["GREEN", "YELLOW", "RED"])
    parser.add_argument("--delta", type=float, help="Delta percentage")
    parser.add_argument("--actual", type=int, help="Actual registration count")
    parser.add_argument("--predicted", type=float, help="Model prediction for today")
    parser.add_argument("--predicted-final", type=float, help="Final count forecast")
    parser.add_argument("--id", help="Tournament ID")

    parser.add_argument("--email-only", action="store_true")
    parser.add_argument("--sms-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print message but don't send")
    return parser.parse_args()


def build_prediction_from_args(args) -> dict:
    return {
        "tournament": args.tournament,
        "tournament_id": args.id or args.tournament.replace(" ", "_"),
        "as_of": datetime.utcnow().date().isoformat(),
        "status": args.status or "YELLOW",
        "delta_pct": args.delta or 0.0,
        "actual_count": args.actual or 0,
        "predicted_now": args.predicted or 0.0,
        "predicted_final": args.predicted_final or 0.0,
        "ci_low": (args.predicted_final or 0) * 0.85,
        "ci_high": (args.predicted_final or 0) * 1.15,
        "params": {},
        "r2": None,
        "rmse": None,
    }


def main():
    args = parse_args()

    # Load prediction data
    if args.stdin:
        prediction = json.load(sys.stdin)
    elif args.file:
        with open(args.file) as f:
            prediction = json.load(f)
    else:
        prediction = build_prediction_from_args(args)

    subject = format_subject(prediction)
    body = format_body(prediction)

    if args.dry_run:
        print("--- DRY RUN --- (not sending)")
        print(f"Subject: {subject}")
        print(f"\n{body}")
        return

    channels_used = []

    if not args.sms_only:
        if send_email(subject, body):
            channels_used.append("email")

    if not args.email_only:
        # SMS gets a short version
        sms_msg = (
            f"[{prediction['status']}] {prediction['tournament']}: "
            f"{prediction['actual_count']} actual, {prediction['predicted_now']:.0f} expected "
            f"({prediction['delta_pct']:+.1f}%). "
            f"Final forecast: {prediction['predicted_final']:.0f}"
        )
        if send_sms(sms_msg):
            channels_used.append("sms")

    log_alert(prediction, channels_used)

    if channels_used:
        print(f"Alert sent via: {', '.join(channels_used)}")
    else:
        print("No alerts sent (credentials not configured). Message logged only.")


if __name__ == "__main__":
    main()
