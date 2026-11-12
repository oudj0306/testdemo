from fastapi import HTTPException

from grading_agent.storage.db import INTEGRITY_ERRORS


def fail(exc: Exception) -> HTTPException:
    if isinstance(exc, LookupError):
        return HTTPException(404, str(exc))
    if isinstance(exc, (ValueError, *INTEGRITY_ERRORS)):
        return HTTPException(400, str(exc))
    return HTTPException(500, str(exc))
