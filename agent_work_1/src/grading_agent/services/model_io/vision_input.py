"""Provider-aware document input for vision requests."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from grading_agent.config import Settings
from grading_agent.model_catalog import provider_for
from grading_agent.services.documents.ingestion import NormalizedPage, pdf_visual_pages


def add_visual_document(
    content: list[dict[str, Any]],
    path: Path,
    media_type: str,
    settings: Settings,
    pages: list[NormalizedPage] | None = None,
) -> None:
    # Render PDFs for known vision presets and Qwen 3.8 (which silently ignores
    # PDF input via Responses). Leave other custom providers' old file path intact.
    visual_pdf = (
        settings.openai_model.lower().startswith("qwen3.8-")
        or provider_for(settings.openai_base_url, settings.openai_model) != "custom"
    )
    if media_type == "application/pdf" and visual_pdf:
        for number, blob in pdf_visual_pages(path, pages):
            content.append({"type": "input_text", "text": f"PDF 第 {number} 页："})
            content.append(
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64," + base64.b64encode(blob).decode("ascii"),
                    "detail": "high",
                }
            )
        return
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    data_url = f"data:{media_type};base64,{encoded}"
    if media_type == "application/pdf":
        content.append({"type": "input_file", "filename": path.name, "file_data": data_url})
    else:
        content.append({"type": "input_image", "image_url": data_url, "detail": "high"})
