"""Serve content-addressed UI assets and revalidate unversioned requests."""

import hashlib
import mimetypes
import re
from urllib.parse import parse_qs

from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from grading_agent.web.paths import STATIC_DIR

ASSETS = {
    name: (STATIC_DIR / name).read_bytes()
    for name in ("style.css", "app.js", "review-evidence.js", "chart.umd.min.js")
}
VERSIONS = {name: hashlib.sha256(content).hexdigest()[:20] for name, content in ASSETS.items()}


def render_index() -> str:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    for name, version in VERSIONS.items():
        html = re.sub(
            rf'/static/{re.escape(name)}(?:\?[^"]*)?(?=")',
            f"/static/{name}?v={version}",
            html,
        )
    return html


class VersionedStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        version = parse_qs(scope.get("query_string", b"").decode("ascii", errors="ignore")).get("v")
        if scope["method"] in {"GET", "HEAD"} and path in ASSETS and version == [VERSIONS[path]]:
            content = ASSETS[path]
            return Response(
                content if scope["method"] == "GET" else b"",
                media_type=mimetypes.guess_type(path)[0],
                headers={
                    "Cache-Control": "public, max-age=31536000, immutable",
                    "Content-Length": str(len(content)),
                },
            )
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response
