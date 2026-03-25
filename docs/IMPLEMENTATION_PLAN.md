# Salesforce Sysadmin Slackbot - Implementation Plan

## Purpose

This document captures the initial implementation plan for a Slack DM-based Salesforce sysadmin bot with strict human-in-the-loop controls. The design optimizes for fast delivery, safety, and auditability for the trial scope.

## Spec-Aligned Constraints (Non-Negotiable)

1. Slack DMs only (no channels).
2. Read-only requests execute immediately.
3. Any Salesforce write (record or metadata create/update/delete) must be approved by exactly one designated human coworker before execution.
4. Only the designated human coworker can edit bot context.
5. Bot starts with minimal hardcoded org knowledge and learns from Salesforce discovery + coworker-provided context.

## Architecture Summary

Single Python backend service using FastAPI + Slack Bolt, modularized by responsibility.

- **Ingress**: Slack events/interactions received and verified.
- **Orchestrator**: Routes each message to read/write/context/approval paths.
- **Planner**: Produces structured plans with assumptions, ordered operations, and safety rationale.
- **Policy + Evidence**: Evaluates decisions against discovered org facts and coworker-authored rules.
- **Approvals**: Persists and presents immutable plans to coworker for explicit approval.
- **Executor**: Runs only approved frozen plans.
- **Context**: Stores and versions coworker-authored and observed context.
- **Discovery**: Bootstraps schema/rules/patterns from Salesforce.
- **Audit**: Tracks all critical decision and execution events.

## Recommended Stack

- Python 3.11+
- FastAPI + Uvicorn
- Slack Bolt for Python (`slack_bolt`, `slack_sdk`)
- Salesforce client adapter (`simple-salesforce` + targeted REST/Tooling/Metadata requests)
- PostgreSQL + SQLAlchemy + Alembic
- Pydantic for strict request/plan schemas
- Optional async jobs: lightweight queue after MVP (RQ/Celery later)

## Proposed Repository Layout (MVP-Oriented)

```text
app/
  main.py
  config.py
  lifecycle.py
  logging.py

  api/
    routes/
      slack_events.py
      slack_interactions.py
      health.py

  slack/
    verifier.py
    parser.py
    responder.py
    blocks.py
    role_resolution.py

  orchestrator/
    service.py
    classifier.py
    intent_models.py
    clarification.py

  planning/
    service.py
    schemas.py
    normalizer.py
    safety_checks.py

  policy/
    service.py
    evaluator.py
    retrieval.py
    question_generator.py
    schemas.py

  evidence/
    service.py
    ingestion.py
    provenance.py
    redaction.py

  discovery/
    bootstrap.py
    metadata_scan.py
    validation_rules.py
    field_descriptions.py
    data_patterns.py

  salesforce/
    client_factory.py
    records.py
    metadata.py
    queries.py
    describe.py
    trace.py

  approvals/
    service.py
    presenter.py
    parser.py
    freeze.py

  executions/
    service.py
    preflight.py
    runner.py
    postflight.py

  context/
    service.py
    mutations.py
    coworker_updates.py
    schemas.py

  db/
    base.py
    session.py
    enums.py
    models/
      workspace.py
      user.py
      conversation.py
      message.py
      execution_plan.py
      approval.py
      execution.py
      context_entry.py
      policy_evidence.py
      audit_log.py

  llm/
    client.py
    schemas.py
    parser.py
```

## Message Routing Model

Every inbound DM is classified into one intent:

- `read_request`
- `write_request`
- `context_edit` (coworker-only)
- `approval_response` (coworker-only)
- `clarification`

Routing outcome:

- **Read path**: execute now, return result.
- **Write path**: draft plan, persist, request coworker approval, wait.
- **Context edit path**: validate coworker identity, store update, acknowledge with summary.
- **Approval path**: parse approval/denial, transition status, optionally execute.

## Structured Plan Contract

Each actionable request produces a strict plan object:

- User intent summary
- Request type (`read` or `write`)
- Assumptions
- Missing information / clarifications needed
- Ordered Salesforce operations
- Safety checks and rationale
- User-facing draft response

For write plans, the persisted plan is immutable once approval is requested.

## Approval Invariants

1. No write executes without explicit coworker approval.
2. Approval applies to one exact immutable plan snapshot.
3. If material context changes occur before execution, previous approval is invalidated and re-approval is required.
4. All approval decisions and execution outcomes are audit logged.

## Context and Policy Model

Context entries are versioned and source-tagged:

- `observation` (from discovery/evidence)
- `inference` (heuristics with confidence)
- `coworker` (highest precedence)

Precedence at decision time:

1. Coworker-authored active context
2. Verified Salesforce observations
3. Inferred heuristics

Only coworker-authored messages can create or modify authoritative context.

## Salesforce Discovery Strategy

Bootstrap job for each workspace:

1. **Schema discovery**: objects, fields, picklists, requiredness, descriptions/help text.
2. **Constraint discovery**: validation rules and relevant automation metadata where feasible.
3. **Pattern sampling**: naming conventions and common data shapes on key objects.
4. **Gap detection**: generate targeted coworker questions for unresolved policy ambiguity.

MVP focus objects:

- Account
- Opportunity
- Case
- User

## Initial Database Entities

Core entities for MVP:

- `workspace`, `user`, `conversation`, `message`
- `execution_plan`, `approval`, `execution`
- `context_entry`
- `policy_evidence`
- `audit_log`
- `slack_user_salesforce_user_map` (or equivalent mapping table)

## Iterative Delivery Phases

### Phase 0 - Skeleton
- FastAPI app, Slack endpoint verification, config, DB, migrations, logging, stub orchestrator.

### Phase 1 - Read-Only MVP
- Classifier, Salesforce describe/query path, formatted Slack responses, clarifications for ambiguous read requests.

### Phase 2 - Write Planning
- Structured plan generation and persistence, write-intent detection, coworker approval message rendering.

### Phase 3 - Approval + Exact Execution
- Approval parser/buttons, immutable plan freeze, exact ordered execution, completion summary with IDs/links.

### Phase 4 - Coworker Context Updates
- Coworker-only context mutation, acknowledgement + change summary, supersession/versioning.

### Phase 5 - Discovery + Evidence
- Bootstrap scan, ingest metadata and validation rules, pattern extraction, unresolved question queue.

### Phase 6 - Policy Evaluator
- Allowed/disallowed/ambiguous decisions tied to evidence and prior precedent.

## MVP Acceptance Criteria

MVP is complete when all are true:

1. Read-only requests execute immediately without approval.
2. Any write request produces a structured plan and requests coworker approval.
3. Approved write executes exactly as planned and reports completion with Salesforce IDs/links.
4. Denied write does not execute and requester is informed.
5. Coworker can update context and receives a concise confirmation of what changed.
6. Discovery bootstrap stores evidence and at least a basic queue of unresolved policy questions.
7. Plan/approval/execution lifecycle is auditable from persisted records.

## Risks and Mitigations

- **Approval bypass risk** -> enforce write-tool access only in executor after approval status check.
- **Identity mismatch risk** -> explicit Slack-to-role mapping + Slack-to-Salesforce user mapping with coworker correction flow.
- **Ambiguous policy risk** -> force coworker clarification when evidence confidence is low.
- **Secrets leakage risk** -> env vars only, redact logs, never store raw credentials in DB or prompts.

## Immediate Next Implementation Steps

1. Initialize project scaffold (`app/`, `alembic/`, `tests/`, `docs/`).
2. Implement Slack ingress routes and signature verification.
3. Set up SQLAlchemy models + first Alembic migration for core entities.
4. Build read-only orchestration path for 2-3 representative queries.
5. Add plan schema and approval packet generation for write intents.

