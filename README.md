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
- Optional MCP backend: `READ_BACKEND=salesforce_mcp`, `SALESFORCE_MCP_COMMAND`, `SALESFORCE_MCP_ARGS`
- Plan execution backend: `PLAN_BACKEND`, `PLAN_EXECUTE_ON_APPROVE`
- Knowledge ingestion controls: `KNOWLEDGE_INGESTION_ENABLED`, `KNOWLEDGE_INGESTION_MAX_ITEMS`
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
  - Optional Salesforce DX MCP backend:
    - Set `READ_BACKEND=salesforce_mcp`
    - The MCP agent auto-filters mutating tools by name and blocks disallowed calls at runtime
  - Auth path priority:
    - Per-user Salesforce OAuth token (if connected)
    - Fallback integration credentials from env
  - If Claude or Salesforce config is missing/invalid, bot returns a setup error.
- Write request:
  - Bot detects write intent and stores a DB-backed execution plan intent.
  - End-user write plans are `pending_approval` for coworker review.
  - Coworker write plans are created and immediately marked `approved`.
  - If `PLAN_BACKEND=salesforce_api` and `PLAN_EXECUTE_ON_APPROVE=true`, approved plans execute deterministically from `operations_json` step-by-step.
  - Coworker approval commands:
    - `approve plan <plan_id>`
    - `reject plan <plan_id> because <reason>`
    - `request changes plan <plan_id> because <reason>`
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

## Salesforce DX MCP Setup (Read-Only Path)

Use this when you want read coverage from Salesforce DX MCP tools instead of handwritten read tools.

1. Install prerequisites:
   - Node.js 18+
   - Salesforce CLI
   - Python dependency `mcp` (included in this project dependency list)
2. Authorize at least one org locally with Salesforce CLI:
   - `sf org login web -a <your_alias>`
3. Verify your default org, or decide explicit `--orgs` value:
   - `sf org list`
4. Configure env vars:
   - `READ_BACKEND=salesforce_mcp`
   - `SALESFORCE_MCP_COMMAND=npx`
   - `SALESFORCE_MCP_ARGS=-y @salesforce/mcp@latest --orgs DEFAULT_TARGET_ORG --toolsets orgs,data --tools list_all_orgs,get_username,run_soql_query`
5. Optional hardening:
   - Keep `--tools` limited to read-oriented tools.
   - Avoid `--toolsets all` unless you explicitly need it.
   - Avoid `--allow-non-ga-tools` unless required.
6. Restart the app.
7. Test in DM with a simple read prompt, for example:
   - "List 5 accounts from Salesforce."

### Recommended MCP Permission Scope

- `--orgs`: Prefer `DEFAULT_TARGET_ORG` or specific alias; avoid `ALLOW_ALL_ORGS` unless necessary.
- `--toolsets`: Prefer `orgs,data` for read scenarios.
- `--tools`: Start with `list_all_orgs,get_username,run_soql_query`.
- Keep your bot in `READ_BACKEND=salesforce_mcp` only after local CLI auth is working.

## Deterministic Plan Execution Backend

Use this to execute approved plans as concrete Salesforce API transactions.

- `PLAN_BACKEND=manual` (default): approval updates status only.
- `PLAN_BACKEND=salesforce_api`: execute approved plans via deterministic operation list.
- `PLAN_EXECUTE_ON_APPROVE=true`: run executor immediately after approval.

Supported deterministic operation schema (inside `operations_json`):

- `sobject_create`: `{ "op":"sobject_create","object":"Account","fields":{...} }`
- `sobject_update`: `{ "op":"sobject_update","object":"Opportunity","record_id":"006...","fields":{...} }`
- `sobject_upsert`: `{ "op":"sobject_upsert","object":"Account","external_id_field":"External_Id__c","external_id":"abc","fields":{...} }`
- `sobject_delete`: `{ "op":"sobject_delete","object":"Account","record_id":"001..." }`

Invalid or unsupported operations are marked `failed` with an audit reason.

## Initial Knowledge Ingestion Pipeline

After each successful read response, ingestion can extract structured:

- facts
- hypotheses
- sysadmin questions

into `knowledge_items` with confidence tiering and provenance.

- `KNOWLEDGE_INGESTION_ENABLED=true`
- `KNOWLEDGE_INGESTION_MAX_ITEMS=30`
