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
- Future OAuth placeholders (not required yet): `SALESFORCE_OAUTH_CLIENT_ID`, `SALESFORCE_OAUTH_CLIENT_SECRET`, `SALESFORCE_OAUTH_REDIRECT_URI`

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
  - If Claude or Salesforce config is missing/invalid, bot returns a setup error.
- Write request:
  - Bot detects write intent and responds with Phase 2 approval-gate placeholder.
- Context edit:
  - Coworker-only check wired; persistence comes in later phases.

## Notes

- Slack channel messages are ignored; DM-only behavior is enforced.
- If Salesforce credentials are missing, read requests return a setup prompt.
- If `ANTHROPIC_API_KEY` is missing, read requests return a setup prompt.
- No write operations are executed in Phase 1.
