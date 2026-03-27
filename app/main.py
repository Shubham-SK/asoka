from __future__ import annotations

import math
from html import escape

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy import String, cast, func, or_, select

from app.config import get_settings
from app.db.bootstrap import init_db
from app.db.enums import KnowledgeKind, KnowledgeLifecycleStatus
from app.db.models import KnowledgeItem, Workspace
from app.db.session import SessionLocal
from app.logging import setup_logging
from app.salesforce.oauth import build_salesforce_authorize_url, handle_oauth_callback
from app.slack.bolt_app import build_slack_app

settings = get_settings()
setup_logging(settings.log_level)

app = FastAPI(title="Salesforce Slackbot", version="0.1.0")

slack_handler = None
if settings.slack_enabled:
    _, slack_handler = build_slack_app(settings)


@app.on_event("startup")
async def startup() -> None:
    init_db()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "env": settings.app_env}


@app.post("/slack/events")
async def slack_events(req: Request):
    if not settings.slack_enabled or slack_handler is None:
        raise HTTPException(
            status_code=503, detail="Slack is not configured. Set SLACK_BOT_TOKEN and secret."
        )
    return await slack_handler.handle(req)


@app.get("/oauth/salesforce/start")
async def oauth_salesforce_start(
    slack_user_id: str = Query(...),
    workspace_id: str = Query(...),
):
    try:
        url = build_salesforce_authorize_url(
            slack_user_id=slack_user_id,
            workspace_id=workspace_id,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(url=url, status_code=302)


@app.get("/oauth/salesforce/callback")
async def oauth_salesforce_callback(
    code: str = Query(...),
    state: str = Query(...),
):
    try:
        payload = handle_oauth_callback(code=code, state=state)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"OAuth callback failed: {exc}") from exc
    return JSONResponse(
        {
            "status": "connected",
            "message": "Salesforce OAuth connected. You can return to Slack and retry your request.",
            "slack_user_id": payload["slack_user_id"],
            "workspace_id": payload["workspace_id"],
            "salesforce_org_key": payload["salesforce_org_key"],
        }
    )


@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse(
        {
            "service": "sf-slackbot",
            "phase": "0-1",
            "slack_enabled": settings.slack_enabled,
            "salesforce_enabled": settings.salesforce_enabled,
        }
    )


@app.get("/knowledge", response_class=HTMLResponse)
async def knowledge_review(
    workspace_id: str = Query("default"),
    q: str = Query(""),
    kind: str = Query(""),
    lifecycle: str = Query(""),
    sf_object: str = Query(""),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=5, le=100),
) -> HTMLResponse:
    with SessionLocal() as db:
        workspace = db.scalar(select(Workspace).where(Workspace.slack_team_id == workspace_id))
        if workspace is None:
            return HTMLResponse(
                _render_knowledge_page(
                    workspace_id=workspace_id,
                    items=[],
                    total_items=0,
                    page=1,
                    page_size=page_size,
                    q=q,
                    kind=kind,
                    lifecycle=lifecycle,
                    sf_object=sf_object,
                    object_options=[],
                )
            )

        filters = [KnowledgeItem.workspace_id == workspace.id]
        if kind in {item.value for item in KnowledgeKind}:
            filters.append(KnowledgeItem.kind == KnowledgeKind(kind))
        if lifecycle in {item.value for item in KnowledgeLifecycleStatus}:
            filters.append(KnowledgeItem.lifecycle_status == KnowledgeLifecycleStatus(lifecycle))
        if sf_object.strip():
            filters.append(KnowledgeItem.sf_object_api_name == sf_object.strip())
        query_text = q.strip().lower()
        if query_text:
            filters.append(
                or_(
                    func.lower(KnowledgeItem.title).like(f"%{query_text}%"),
                    func.lower(cast(KnowledgeItem.content_json, String)).like(f"%{query_text}%"),
                )
            )

        count_stmt = select(func.count()).select_from(KnowledgeItem).where(*filters)
        total_items = int(db.scalar(count_stmt) or 0)
        total_pages = max(1, math.ceil(total_items / page_size)) if total_items else 1
        current_page = min(page, total_pages)
        offset = (current_page - 1) * page_size

        stmt = (
            select(KnowledgeItem)
            .where(*filters)
            .order_by(KnowledgeItem.updated_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        items = list(db.scalars(stmt).all())

        object_stmt = (
            select(KnowledgeItem.sf_object_api_name)
            .where(KnowledgeItem.workspace_id == workspace.id, KnowledgeItem.sf_object_api_name.is_not(None))
            .distinct()
            .order_by(KnowledgeItem.sf_object_api_name.asc())
        )
        object_options = [str(v).strip() for v in db.scalars(object_stmt).all() if str(v).strip()]

    return HTMLResponse(
        _render_knowledge_page(
            workspace_id=workspace_id,
            items=items,
            total_items=total_items,
            page=current_page,
            page_size=page_size,
            q=q,
            kind=kind,
            lifecycle=lifecycle,
            sf_object=sf_object,
            object_options=object_options,
        )
    )


def _render_knowledge_page(
    workspace_id: str,
    items: list[KnowledgeItem],
    total_items: int,
    page: int,
    page_size: int,
    q: str,
    kind: str,
    lifecycle: str,
    sf_object: str,
    object_options: list[str],
) -> str:
    total_pages = max(1, math.ceil(total_items / page_size)) if total_items else 1

    kind_options = ['<option value="">All kinds</option>']
    for item in KnowledgeKind:
        selected = ' selected="selected"' if kind == item.value else ""
        kind_options.append(f'<option value="{escape(item.value)}"{selected}>{escape(item.value)}</option>')

    lifecycle_options = ['<option value="">All lifecycle statuses</option>']
    for item in KnowledgeLifecycleStatus:
        selected = ' selected="selected"' if lifecycle == item.value else ""
        lifecycle_options.append(
            f'<option value="{escape(item.value)}"{selected}>{escape(item.value)}</option>'
        )

    object_select_options = ['<option value="">All objects</option>']
    for obj in object_options:
        selected = ' selected="selected"' if sf_object == obj else ""
        object_select_options.append(f'<option value="{escape(obj)}"{selected}>{escape(obj)}</option>')

    rows: list[str] = []
    for item in items:
        statement = str((item.content_json or {}).get("statement", "")).strip()
        if len(statement) > 220:
            statement = statement[:217] + "..."
        rows.append(
            "<tr>"
            f"<td><code>{escape(item.id)}</code></td>"
            f"<td>{escape(item.kind.value)}</td>"
            f"<td>{escape(item.lifecycle_status.value)}</td>"
            f"<td>{escape(item.title)}</td>"
            f"<td>{escape(statement)}</td>"
            f"<td>{escape(item.sf_object_api_name or '')}</td>"
            f"<td>{int(item.usage_count or 0)}</td>"
            f"<td>{escape(item.updated_at.isoformat())}</td>"
            "</tr>"
        )
    if not rows:
        rows.append('<tr><td colspan="8">No knowledge items found.</td></tr>')

    prev_page = max(1, page - 1)
    next_page = min(total_pages, page + 1)
    qs_base = (
        f"workspace_id={escape(workspace_id)}&q={escape(q)}&kind={escape(kind)}"
        f"&lifecycle={escape(lifecycle)}&sf_object={escape(sf_object)}&page_size={page_size}"
    )
    prev_href = f"/knowledge?{qs_base}&page={prev_page}"
    next_href = f"/knowledge?{qs_base}&page={next_page}"

    return f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Knowledge Review</title>
    <style>
      body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; }}
      h1 {{ margin-bottom: 8px; }}
      form {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }}
      input, select, button {{ padding: 6px 8px; font-size: 14px; }}
      table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
      th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
      th {{ background: #f7f7f7; text-align: left; }}
      .meta {{ margin-bottom: 12px; color: #555; }}
      .pager {{ margin-top: 12px; display: flex; gap: 10px; align-items: center; }}
    </style>
  </head>
  <body>
    <h1>Knowledge Items</h1>
    <div class="meta">Workspace: <code>{escape(workspace_id)}</code> | Total: {total_items}</div>
    <form method="get" action="/knowledge">
      <input type="hidden" name="workspace_id" value="{escape(workspace_id)}" />
      <input type="text" name="q" placeholder="Search title/statement" value="{escape(q)}" />
      <select name="kind">{''.join(kind_options)}</select>
      <select name="lifecycle">{''.join(lifecycle_options)}</select>
      <select name="sf_object">{''.join(object_select_options)}</select>
      <select name="page_size">
        <option value="10"{" selected=\"selected\"" if page_size == 10 else ""}>10 / page</option>
        <option value="25"{" selected=\"selected\"" if page_size == 25 else ""}>25 / page</option>
        <option value="50"{" selected=\"selected\"" if page_size == 50 else ""}>50 / page</option>
        <option value="100"{" selected=\"selected\"" if page_size == 100 else ""}>100 / page</option>
      </select>
      <button type="submit">Apply</button>
    </form>
    <table>
      <thead>
        <tr>
          <th>ID</th><th>Kind</th><th>Lifecycle</th><th>Title</th><th>Statement</th>
          <th>Object</th><th>Usage</th><th>Updated</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
    <div class="pager">
      <a href="{prev_href}">Prev</a>
      <span>Page {page} / {total_pages}</span>
      <a href="{next_href}">Next</a>
    </div>
  </body>
</html>
"""
