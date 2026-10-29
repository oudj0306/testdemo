"""One bounded question pool shared by all batch and interactive requests."""

import os
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

_lock = Lock()
_pool = None


def question_executor():
    global _pool
    with _lock:
        if _pool is None:
            try:
                workers = max(1, min(8, int(os.getenv("GRADING_QUESTION_WORKERS", "4"))))
            except ValueError:
                workers = 4
            _pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="grading-question")
        return _pool
