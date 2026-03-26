# System Architecture and Implementation Status

This document describes the current system architecture for the Salesforce Slackbot, what is implemented today, known limitations, and recommended next improvements.

## High-Level System Diagram

```text
                                +----------------------+
                                |   Slack Workspace    |
                                | (DMs + button clicks)|
                                +----------+-----------+
                                           |
                                           v
                                 POST /slack/events
                                           |
                                           v
+-------------------+            +-----------------------+            +----------------------+
| FastAPI App       |----------->| Slack Bolt Handlers   |----------->| Orchestrator         |
| app/main.py       |            | app/slack/handlers.py |            | app/orchestrator/*   |
+-------------------+            +-----------+-----------+            +----+------------+----+
                                              |                            |            |
                                              |                            |            |
                                              v                            v            v
                                      +---------------+           +----------------+  +----------------+
                                      | SQLite/Postgres|          | Read Agent     |  | Plan Agent     |
                                      | app/db/*       |          | app/agent/*    |  | plan_agent.py  |
                                      +-------+-------+           +--------+-------+  +--------+-------+
                                              |                            |                  |
                                              v                            v                  v
                                      +----------------+           +----------------+  +----------------+
                                      | Conversation   |           | Salesforce API |  | Plan Backend   |
                                      | history/context|           | + Tooling API  |  | execute plan   |
                                      +----------------+           +----------------+  +----------------+
                                                                                               |
                                                                                               v
                                                                                      +--------------------+
                                                                                      | Salesforce Writes  |
                                                                                      | (approved plans)   |
                                                                                      +--------------------+
```

## Request Flow (Current)

1. Slack DM arrives at `POST /slack/events`.
2. Handler loads last 25 messages for that DM from DB.
3. Handler persists inbound message + context snapshots.
4. Orchestrator classifies intent using Claude (`read_request`, `write_request`, `approval_response`, etc).
5. Route by intent:
   - Read path -> `run_read_agent` or MCP read agent (config driven).
   - Write/approval/plan management -> `run_plan_agent`.
   - Knowledge ingestion/management -> ingestion agent + KB flows (coworker only).
6. Handler posts placeholder (`Thinking...`) and updates final result, with chunked followups if needed.
7. Messages and plan status changes are persisted and optionally notified.

## Components Implemented

### 1) API + Slack Integration
- FastAPI app and Slack events endpoint in `app/main.py`.
- OAuth endpoints for per-user Salesforce connect:
  - `/oauth/salesforce/start`
  - `/oauth/salesforce/callback`
- Slack DM-only handling and approval button actions in `app/slack/handlers.py`.

### 2) Orchestration and Intent Routing
- Central router in `app/orchestrator/service.py`.
- LLM-based classifier in `app/orchestrator/classifier.py`.
- Rule-based coworker shortcuts for knowledge ingestion/management still exist (in classifier).

### 3) Read Agent
- Read-agent loop in `app/agent/service.py`.
- Supports tool actions:
  - `query`
  - `tooling_query`
  - `describe`
  - artifact exploration (`artifact_list_keys`, `artifact_get_tree`, `artifact_search_text`, `artifact_extract_path`)
- Includes:
  - step-by-step observability trace
  - progress callback updates
  - forced finalization when step budget is exhausted
  - inline vs artifact materialization for tool outputs

### 4) Artifact System
- In-memory artifact store in `app/agent/artifacts.py`.
- Artifacts currently process-local and non-durable.
- Conditional artifacting based on payload size/record counts.

### 5) Write Plan and Approval Flow
- Plan creation/orchestration in `app/orchestrator/plan_agent.py`.
- Plan persistence and statuses in DB.
- Approval via Slack button and/or command flow.
- Post-approval execution pipeline in `app/orchestrator/plan_backend.py`.
- Execution includes operation validation, deterministic op execution, and execution observability.

### 6) Knowledge Ingestion/Management
- Ingestion pipeline in `app/evidence/ingestion.py`:
  - discovery context gathering
  - staged extraction to structured facts/hypotheses/questions
  - persistence into `knowledge_items`
- Knowledge management agent path exists for coworker workflows.

### 7) Persistence Layer
- SQLAlchemy models in `app/db/models.py`.
- Key persisted entities:
  - workspace/user/identity
  - conversation/messages/context entries
  - execution plans
  - knowledge items

## Current Strengths

- End-to-end Slack DM workflow is operational.
- Intent routing is centralized and observable.
- Read path has flexible tooling + artifact-based large JSON exploration.
- Plan approval model with explicit coworker authority exists.
- Deterministic plan execution pipeline is separated from planning.
- Conversation windows are persisted and reused for context.

## Known Limitations

1. **Artifacts are in-memory only**
   - Lost on restart/reload.
   - Not shared across instances.

2. **Manual JSON action protocol**
   - Both read and plan agents depend on model emitting valid JSON.
   - Forced-finalization mitigates but does not eliminate protocol fragility.

3. **Pseudo-streaming only**
   - Uses placeholder + `chat_update` and chunked followups.
   - Does not currently use Slack `chat.startStream/appendStream/stopStream`.

4. **Tool schema awareness gaps**
   - Tooling/standard SOQL field mismatches still occur and cause retries.
   - Needs stronger object-field capability constraints to reduce API errors.

5. **Context + policy semantics still maturing**
   - Context-edit and KB workflows exist, but policy precedence and conflict resolution need more hardening.

6. **Runtime safeguards**
   - Long-running broad queries can still be expensive despite step limits.
   - Additional wall-clock cutoffs and error-budget caps are advisable.

## Recommended Next Improvements (Prioritized)

### Priority 1 (Reliability)
1. Persist artifacts to Redis/Postgres with TTL and cleanup policies.
2. Add wall-clock timeout + max consecutive tool error threshold.
3. Add schema-aware query planner helpers (valid fields per object/tool endpoint).
4. Expose artifact thresholds via config/env instead of constants.

### Priority 2 (Protocol Robustness)
1. Migrate manual JSON action loops to Anthropic native tool calls.
2. Keep deterministic tool wrappers and observability unchanged.
3. Retain forced-finalization fallback for resilience.

### Priority 3 (Product UX)
1. Upgrade to Slack stream APIs for richer progressive updates.
2. Add user-facing status stages (classify -> discover -> query -> summarize).
3. Improve clarification UX with concrete selectable suggestions.

### Priority 4 (Governance and Security)
1. Encrypt sensitive identity tokens at rest.
2. Add audit trail table for tool calls and plan execution transitions.
3. Add per-workspace concurrency controls and idempotency keys.

## Definition of "Stable v1"

The current implementation should be considered stable v1 when:
- artifact persistence is durable (not in-memory only),
- read and plan agents use native tool-call protocol,
- API schema mismatch rate is reduced by pre-validation,
- and Slack streaming UX is consistent across all primary user paths.
