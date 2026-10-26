"""Cooperative batch controls; finish and save in-flight work before suspending."""
from contextlib import contextmanager
from contextvars import ContextVar

from ..db import transaction

_batch=ContextVar('controlled_batch',default=None)
_finishing=ContextVar('batch_finishing_result',default=False)


class BatchSuspended(Exception):
    pass


@contextmanager
def batch_scope(batch_id,item_id=None):
    token=_batch.set((batch_id,item_id))
    try:yield
    finally:_batch.reset(token)


@contextmanager
def finishing_result():
    token=_finishing.set(True)
    try:yield
    finally:_finishing.reset(token)


def controlled():
    return _batch.get() is not None


def checkpoint():
    context=_batch.get()
    if not context or _finishing.get():return
    with transaction() as db:
        row=db.execute('SELECT status FROM grading_batches WHERE id=?',(context[0],)).fetchone()
    if row and row['status'] in {'paused','stopped'}:raise BatchSuspended(row['status'])


def bind_submission(submission_id):
    context=_batch.get()
    if context and context[1]:
        with transaction() as db:
            db.execute('UPDATE grading_batch_items SET submission_id=? WHERE id=?',(submission_id,context[1]))


def control_batch(batch_id,action):
    from ..repository import get_batch
    if action not in {'pause','stop','resume'}:raise ValueError('未知批阅控制操作')
    with transaction() as db:
        db.execute('BEGIN IMMEDIATE')
        batch=db.execute('SELECT * FROM grading_batches WHERE id=?',(batch_id,)).fetchone()
        if not batch:raise LookupError('批次不存在')
        active=db.execute("SELECT COUNT(*) FROM grading_batch_items WHERE batch_id=? AND status IN ('extracting','grading')",(batch_id,)).fetchone()[0]
        if action=='resume':
            if batch['status'] not in {'paused','stopped'}:raise ValueError('只有已暂停或停止的批次可以继续')
            if active:raise ValueError('当前任务正在保存进度，请稍后继续')
            state='processing'
        else:
            if batch['status'] not in {'processing','paused','stopped'}:raise ValueError('已结束批次不能暂停或停止')
            state='paused' if action=='pause' else 'stopped'
        db.execute('UPDATE grading_batches SET status=?,completed_at=NULL WHERE id=?',(state,batch_id))
    return get_batch(batch_id)
