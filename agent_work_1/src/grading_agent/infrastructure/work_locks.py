"""Process-local locks for work on the same saved submission.

The application runs with one server process (as required by the durable queue).
Reference counting keeps the lock registry bounded by in-flight work.
"""

from contextlib import contextmanager
from threading import Lock, RLock

_guard = Lock()
_entries = {}


class SubmissionBusyError(ValueError):
    pass


@contextmanager
def submission_work(database_path, submission_id, *, blocking=True):
    key = (str(database_path), submission_id)
    with _guard:
        entry = _entries.setdefault(key, [RLock(), 0])
        entry[1] += 1
    acquired = False
    try:
        acquired = entry[0].acquire(blocking=blocking)
        if not acquired:
            raise SubmissionBusyError("这份答卷正在识别或批改，请等待当前任务完成")
        yield
    finally:
        if acquired:
            entry[0].release()
        with _guard:
            entry[1] -= 1
            if not entry[1]:
                del _entries[key]
