from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from grading_agent.services.workflows.batch_queue import BatchQueue
from grading_agent.services.workflows.submission_processing import (
    persist_submission as persist_submission,
)
from grading_agent.services.workflows.submission_processing import (
    prepare_submission as prepare_submission,
)
from grading_agent.services.workflows.submission_processing import (
    store_submission as store_submission,
)
from grading_agent.storage.db import initialize_database
from grading_agent.web.paths import STATIC_DIR
from grading_agent.web.routes import (
    assessments,
    batches,
    demo,
    knowledge,
    layouts,
    recognition,
    reviews,
    runtime,
    submissions,
    training,
)
from grading_agent.web.static_assets import VersionedStaticFiles
from grading_agent.web.workspace_views import router as workspace_router


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_database()
    queue = BatchQueue(persist_submission, prepare_submission)
    queue.start()
    try:
        yield
    finally:
        await run_in_threadpool(queue.close)


app = FastAPI(title="作业/试卷批改 Agent", version="0.1.0", lifespan=lifespan)
app.include_router(workspace_router)
app.mount("/static", VersionedStaticFiles(directory=STATIC_DIR), name="static")


for feature in (
    runtime,
    assessments,
    demo,
    submissions,
    batches,
    reviews,
    knowledge,
    layouts,
    recognition,
    training,
):
    app.include_router(feature.router)
