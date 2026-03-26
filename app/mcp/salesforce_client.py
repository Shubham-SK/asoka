from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from app.config import Settings


@dataclass
class SalesforceMcpTool:
    name: str
    description: str
    input_schema: dict[str, Any]


class SalesforceMcpClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if not settings.salesforce_mcp_enabled:
            raise RuntimeError(
                "Salesforce MCP is not configured. Set SALESFORCE_MCP_COMMAND and "
                "SALESFORCE_MCP_ARGS."
            )

    def list_tools(self) -> list[SalesforceMcpTool]:
        return asyncio.run(self._list_tools_async())

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return asyncio.run(self._call_tool_async(tool_name=tool_name, arguments=arguments))

    async def _list_tools_async(self) -> list[SalesforceMcpTool]:
        session = await self._open_session()
        try:
            result = await asyncio.wait_for(
                session["session"].list_tools(),
                timeout=max(5, self.settings.salesforce_mcp_tool_timeout_seconds),
            )
            raw_tools = list(getattr(result, "tools", []) or [])
            tools: list[SalesforceMcpTool] = []
            for tool in raw_tools:
                tools.append(
                    SalesforceMcpTool(
                        name=str(getattr(tool, "name", "") or ""),
                        description=str(getattr(tool, "description", "") or ""),
                        input_schema=self._to_dict(getattr(tool, "inputSchema", {}) or {}),
                    )
                )
            return tools
        finally:
            await self._close_session(session)

    async def _call_tool_async(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        session = await self._open_session()
        try:
            result = await asyncio.wait_for(
                session["session"].call_tool(tool_name, arguments),
                timeout=max(5, self.settings.salesforce_mcp_tool_timeout_seconds),
            )
            payload: dict[str, Any] = {
                "is_error": bool(getattr(result, "isError", False)),
                "content": [],
            }
            for item in list(getattr(result, "content", []) or []):
                payload["content"].append(self._content_item_to_json(item))
            return payload
        finally:
            await self._close_session(session)

    async def _open_session(self) -> dict[str, Any]:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except Exception as exc:  # pragma: no cover - import path varies by runtime
            raise RuntimeError(
                "Python package `mcp` is required for MCP backend. Install dependencies and retry."
            ) from exc

        server_params = StdioServerParameters(
            command=self.settings.salesforce_mcp_command,
            args=self.settings.salesforce_mcp_args_list,
            env=self.settings.salesforce_mcp_env,
        )
        stdio_cm = stdio_client(server_params)
        read_stream, write_stream = await stdio_cm.__aenter__()
        session_cm = ClientSession(read_stream, write_stream)
        session = await session_cm.__aenter__()
        await asyncio.wait_for(
            session.initialize(),
            timeout=max(5, self.settings.salesforce_mcp_init_timeout_seconds),
        )
        return {"stdio_cm": stdio_cm, "session_cm": session_cm, "session": session}

    async def _close_session(self, state: dict[str, Any]) -> None:
        session_cm = state.get("session_cm")
        stdio_cm = state.get("stdio_cm")
        if session_cm is not None:
            await session_cm.__aexit__(None, None, None)
        if stdio_cm is not None:
            await stdio_cm.__aexit__(None, None, None)

    def _content_item_to_json(self, item: Any) -> dict[str, Any]:
        item_type = str(getattr(item, "type", "") or "")
        if item_type == "text":
            return {"type": "text", "text": str(getattr(item, "text", "") or "")}
        if item_type == "image":
            return {
                "type": "image",
                "mimeType": str(getattr(item, "mimeType", "") or ""),
                "data": str(getattr(item, "data", "") or ""),
            }
        if item_type == "resource":
            resource = getattr(item, "resource", None)
            return {
                "type": "resource",
                "resource": self._to_dict(resource) if resource is not None else {},
            }
        return {"type": item_type or "unknown", "value": self._to_dict(item)}

    def _to_dict(self, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if hasattr(value, "model_dump"):
            dumped = value.model_dump()
            return dumped if isinstance(dumped, dict) else {"value": dumped}
        if hasattr(value, "__dict__"):
            return {k: v for k, v in value.__dict__.items() if not k.startswith("_")}
        try:
            raw = json.loads(str(value))
            return raw if isinstance(raw, dict) else {"value": raw}
        except Exception:
            return {"value": str(value)}

