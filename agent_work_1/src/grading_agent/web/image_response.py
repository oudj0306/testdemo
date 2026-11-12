import io

from fastapi.responses import Response


def png_response(image):
    out = io.BytesIO()
    image.save(out, format="PNG")
    return Response(out.getvalue(), media_type="image/png", headers={"Cache-Control": "no-store"})
