# Website + Private Login — Roadmap

> Chess Tournament Entry Predictor — public deployment plan

---

## Phase 1 — Auth Layer on the Backend (1–2 hrs)
Add JWT-based login to existing `tools/api.py`:
- `POST /auth/login` — accepts username/password, returns a signed JWT token
- All other endpoints require `Authorization: Bearer <token>` header
- Store users in a simple table in `schema.sql` (username, hashed password)
- Use `python-jose` for JWT, `passlib` for bcrypt hashing

This is the security foundation. Nothing works without this.

---

## Phase 2 — Login Page in the Frontend (1 hr)
Add a login screen to `chess_predictor.html`:
- On load, check `localStorage` for a valid token
- If missing/expired → show login overlay (full-screen, blocks dashboard)
- On successful `/auth/login` → store token, show dashboard
- All `fetch()` calls to the API add the `Authorization` header
- "Logout" clears token and reloads

---

## Phase 3 — Deployment (2–4 hrs, one-time)
Pick a hosting stack:

| Layer | Recommended | Cost |
|-------|-------------|------|
| Backend (FastAPI) | Railway or Render | Free tier / ~$5/mo |
| Frontend (HTML) | Serve from FastAPI itself, or Netlify | Free |
| Database (Postgres) | Railway or Supabase | Free tier |
| Domain | Namecheap / Cloudflare | ~$10/yr |

Simplest setup: FastAPI serves the HTML file as a static route. One deployment, one URL.

---

## Phase 4 — HTTPS + Hardening (30 min)
- Railway/Render give you HTTPS automatically
- Add `SECURE_COOKIES`, `CORS` origin whitelist, token expiry (e.g., 7 days)
- Rotate `SECRET_KEY` via environment variable (never hardcode)

---

## Phase 5 — Multi-user / Invite System (optional, later)
- Admin-only user creation endpoint
- Invite link that creates a new account
- Role flags (`admin` vs. `viewer`) if you want read-only access for some users

---

## Implementation Order

When ready to build:
1. **Phase 1 + 2 first** — adds working auth locally (no deployment needed yet)
2. **Phase 3** — deploy when ready to go live
3. **Phase 4** — handled mostly automatically by Railway/Render
4. **Phase 5** — only if multiple users needed

## Dependencies to Add
```
python-jose[cryptography]
passlib[bcrypt]
python-multipart
```
