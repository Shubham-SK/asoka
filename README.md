# Salesforce Sysadmin Slackbot (Phase 0-1)

FastAPI + Slack Bolt backend scaffold for the trial spec.

Current scope:

- Phase 0: app skeleton, config, health endpoint, Slack event ingress.
- Phase 1: DM intent routing (read/write/context), read-only Salesforce handlers for initial use cases.
- DB foundation includes future-proof identity mapping for later per-user Salesforce OAuth.

## Quick Start

### 1) Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -e .
```

### 3) Configure environment

```bash
cp .env.example .env
```

Fill required values:

- Slack: `SLACK_BOT_TOKEN`, `SLACK_SIGNING_SECRET`, `SLACK_COWORKER_USER_ID`
- LLM (Claude): `LLM_PROVIDER=anthropic`, `LLM_MODEL`, `ANTHROPIC_API_KEY`
- Salesforce (for read path): `SALESFORCE_USERNAME`, `SALESFORCE_PASSWORD`, `SALESFORCE_SECURITY_TOKEN`
- `SALESFORCE_DOMAIN` should be `login` or `test` (not a full URL)
- For per-user OAuth (recommended): `SALESFORCE_OAUTH_CLIENT_ID`, `SALESFORCE_OAUTH_CLIENT_SECRET`, `SALESFORCE_OAUTH_REDIRECT_URI`
- OAuth state/token secrets: `OAUTH_STATE_SECRET`, `TOKEN_ENCRYPTION_KEY`
- Public base URL for links: `APP_BASE_URL` (for local dev use your tunnel URL)
- Database: `DATABASE_URL` (defaults to local SQLite if omitted)

### 4) Run app

```bash
uvicorn app.main:app --reload --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

## Slack Configuration

In your Slack app config:

- Enable Event Subscriptions
- Request URL: `https://<public-url>/slack/events` (use ngrok/cloudflared locally)
- Subscribe bot event: `message.im`
- OAuth scopes: `chat:write`, `im:history`, `im:read`
- Install app to workspace

## Supported DM Behavior (Phase 1)

- Read request:
  - Claude-driven agent loop can choose read-only tools:
    - Salesforce object describe
    - Read-only SOQL query (`SELECT ...` only)
  - Auth path priority:
    - Per-user Salesforce OAuth token (if connected)
    - Fallback integration credentials from env
  - If Claude or Salesforce config is missing/invalid, bot returns a setup error.
- Write request:
  - Bot detects write intent and responds with Phase 2 approval-gate placeholder.
- Context edit:
  - Coworker-only check wired; persistence comes in later phases.

## Notes

- Slack channel messages are ignored; DM-only behavior is enforced.
- If a user has not connected OAuth and integration creds are missing, bot returns a connect link.
- If `ANTHROPIC_API_KEY` is missing, read requests return a setup prompt.
- No write operations are executed in Phase 1.
- DM conversation context is now persisted per workspace/user/channel in DB, so multi-user chats remain stateful.

## Postgres Quick Start (optional but recommended)

Run Postgres locally with Docker:

```bash
docker run --name asoka-postgres \
  -e POSTGRES_USER=asoka \
  -e POSTGRES_PASSWORD=asoka \
  -e POSTGRES_DB=asoka \
  -p 5432:5432 -d postgres:16
```

Set `DATABASE_URL`:

```bash
DATABASE_URL=postgresql+psycopg://asoka:asoka@localhost:5432/asoka
```

Restart app after updating env.

## Salesforce OAuth Setup (Dashboard Steps)

1. In Salesforce, go to Setup -> App Manager -> New Connected App.
2. Set a name and contact email.
3. Enable OAuth Settings.
4. Callback URL: set to `<APP_BASE_URL>/oauth/salesforce/callback`.
5. OAuth scopes: include `api` and `refresh_token` (or equivalent offline scope).
6. Save, then copy Consumer Key and Consumer Secret.
7. Set env vars:
   - `SALESFORCE_OAUTH_CLIENT_ID=<Consumer Key>`
   - `SALESFORCE_OAUTH_CLIENT_SECRET=<Consumer Secret>`
   - `SALESFORCE_OAUTH_REDIRECT_URI=<APP_BASE_URL>/oauth/salesforce/callback`
   - `OAUTH_STATE_SECRET=<random long secret>`
   - `TOKEN_ENCRYPTION_KEY=<fernet key>`
8. Generate a Fernet key with:
   - `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`
9. Restart the app.
10. In Slack DM with the bot, send `connect salesforce` and complete the OAuth flow.
