from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from grading_agent.schemas import (
    KnowledgeApprove,
)
from grading_agent.storage.repository import (
    create_knowledge_asset,
    search_knowledge,
)
from grading_agent.web.http_errors import fail

router = APIRouter()


@router.post("/api/knowledge/approve")
def knowledge_approve(payload: KnowledgeApprove) -> dict[str, Any]:
    try:
        return create_knowledge_asset(
            payload.response_id,
            payload.kind,
            payload.redacted_text,
            payload.approved_by,
            payload.metadata,
        )
    except Exception as exc:
        raise fail(exc) from exc


@router.get("/api/knowledge/search")
def knowledge_search(
    q: str = Query(default="", max_length=200), limit: int = Query(default=20, ge=1, le=100)
) -> list[dict[str, Any]]:
    return search_knowledge(q, limit)
