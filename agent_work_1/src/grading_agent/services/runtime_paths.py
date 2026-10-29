"""Reuse installed OCR caches and non-secret options; never share business DBs."""

import json
from pathlib import Path


def project_roots(root=None):
    root = root or Path(__file__).resolve().parents[3]
    return [
        p
        for p in [root, *list(root.parents)[:4]]
        if (p / "src" / "grading_agent" / "config.py").is_file()
    ]


def inherited_ocr_defaults(root):
    for candidate in project_roots(root)[1:]:
        try:
            data = json.loads((candidate / "data" / "settings.json").read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                continue
            if data.get("ocr_provider") == "paddleocr":
                return {
                    k: data[k]
                    for k in ("ocr_provider", "ocr_device", "local_ocr_first")
                    if k in data
                }
        except (OSError, ValueError, TypeError):
            continue
    return {}


def model_cache_root():
    roots = project_roots()
    return next(
        (p / ".cache" for p in roots if (p / ".cache" / "paddlex" / "official_models").is_dir()),
        (roots[0] if roots else Path(__file__).resolve().parents[3]) / ".cache",
    )
