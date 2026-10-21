"""Durable local queue. Run the application with one server process."""
from __future__ import annotations

import logging
from contextlib import nullcontext
from threading import Event, Thread
from typing import Callable

from ..config import get_settings
from ..db import transaction, utc_now
from ..repository import complete_batch, get_assessment, get_submission, new_id
from .grading import grade_submission
from .work_locks import SubmissionBusyError, submission_work

logger = logging.getLogger(__name__)


def enqueue(batch_id, filename, student_ref, content, media_type, auto_grade, answer_layout='auto'):
    item_id = new_id()
    with transaction() as connection:
        connection.execute(
            "INSERT INTO grading_batch_items(id,batch_id,filename,student_ref,status,created_at) VALUES (?,?,?,?,?,?)",
            (item_id, batch_id, filename, student_ref, "queued", utc_now()),
        )
        connection.execute("INSERT INTO batch_jobs(item_id,content,media_type,auto_grade,answer_layout) VALUES (?,?,?,?,?)",
                           (item_id, content, media_type, auto_grade, answer_layout))


class BatchQueue:
    def __init__(self, persist: Callable, prepare: Callable | None = None):
        self.persist = persist
        self.prepare = prepare
        self.stop = Event()
        self.threads = []

    def start(self):
        with transaction() as connection:
            # Don't silently repeat potentially billed model calls after a crash.
            connection.execute("UPDATE grading_batch_items SET status='failed',error='服务中断，请检查已保存答卷后重试' WHERE status IN ('extracting','grading')")
            # An interrupted upload can leave an incomplete batch count.
            connection.execute("UPDATE grading_batches SET total_count=(SELECT COUNT(*) FROM grading_batch_items WHERE batch_id=grading_batches.id) WHERE status='processing'")
            batches = [r[0] for r in connection.execute("SELECT id FROM grading_batches WHERE status='processing'")]
        for batch_id in batches:
            complete_batch(batch_id)
        for index in range(2):
            thread = Thread(target=self.run, name=f"grading-{index}", daemon=True)
            thread.start()
            self.threads.append(thread)

    def close(self):
        self.stop.set()
        for thread in self.threads:
            thread.join()

    def claim(self):
        with transaction() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("""SELECT i.*,j.content,j.media_type,j.auto_grade,j.answer_layout,b.assessment_id
                FROM grading_batch_items i JOIN batch_jobs j ON j.item_id=i.id
                JOIN grading_batches b ON b.id=i.batch_id
                WHERE i.status='queued' ORDER BY i.created_at LIMIT 1""").fetchone()
            if row:
                connection.execute("UPDATE grading_batch_items SET status='extracting' WHERE id=?", (row['id'],))
                return dict(row)
        return None

    def run(self):
        while not self.stop.is_set():
            try:
                job = self.claim()
                if job:
                    self.process(job)
                else:
                    self.stop.wait(0.25)
            except Exception:
                logger.exception("后台批改队列异常")
                self.stop.wait(1)

    def process(self, job):
        try:
            guard = (submission_work(get_settings().database_path, job['submission_id'], blocking=False)
                     if job['submission_id'] else nullcontext())
            with guard:
                self.process_submission(job)
            with transaction() as connection:
                connection.execute("UPDATE grading_batch_items SET status='completed',error=NULL WHERE id=?", (job['id'],))
                # Original has been preserved by ingestion; discard the queue-only copy.
                connection.execute("DELETE FROM batch_jobs WHERE item_id=?", (job['id'],))
        except SubmissionBusyError:
            # An interactive request or another worker already owns this
            # submission. Keep the durable job queued instead of failing it or
            # running the same billed work again.
            with transaction() as connection:
                connection.execute("UPDATE grading_batch_items SET status='queued',error=NULL WHERE id=?", (job['id'],))
            self.stop.wait(0.25)
        except Exception as exc:
            with transaction() as connection:
                connection.execute("UPDATE grading_batch_items SET status='failed',error=? WHERE id=?", (str(exc)[:500], job['id']))
        finally:
            complete_batch(job['batch_id'])

    def process_submission(self, job):
        assessment = get_assessment(job['assessment_id'])
        submission = get_submission(job['submission_id']) if job['submission_id'] else None
        if submission and submission['status'] in {'uploaded', 'extraction_failed'} and self.prepare:
            submission = self.prepare(submission['id'])
        if submission is None:
            submission = self.persist(assessment, job['student_ref'], job['content'], job['media_type'],
                                      None, job['answer_layout'])
            with transaction() as connection:
                connection.execute("UPDATE grading_batch_items SET submission_id=? WHERE id=?", (submission['id'], job['id']))
        if submission.get('extraction_warning'):
            raise ValueError("识别失败；可打开答卷校对后重批：" + submission['extraction_warning'])
        if job['auto_grade']:
            with transaction() as connection:
                connection.execute("UPDATE grading_batch_items SET status='grading' WHERE id=?", (job['id'],))
            grade_submission(submission['id'], get_settings(), pending_only=True)


def enqueue_saved(assessment_id: str, submission_ids: list[str]) -> dict:
    from ..repository import get_batch
    batch_id = new_id()
    with transaction() as connection:
        connection.execute('BEGIN IMMEDIATE')
        selected = set(submission_ids)
        rows = connection.execute("SELECT * FROM submissions WHERE assessment_id=? AND (status IN ('uploaded','extraction_failed','extracted') OR (status='review_required' AND ?))", (assessment_id,bool(selected))).fetchall()
        rows = [r for r in rows if (not selected or r['id'] in selected) and not connection.execute("SELECT 1 FROM grading_batch_items WHERE submission_id=? AND status IN ('queued','extracting','grading')", (r['id'],)).fetchone()]
        if not rows:
            raise ValueError('没有待批改答卷（已排队、已批改的答卷不会重复提交）')
        connection.execute('INSERT INTO grading_batches(id,assessment_id,status,total_count,created_at) VALUES (?,?,?,?,?)', (batch_id, assessment_id, 'processing', len(rows), utc_now()))
        for row in rows:
            item_id = new_id()
            connection.execute('INSERT INTO grading_batch_items(id,batch_id,submission_id,filename,student_ref,status,created_at) VALUES (?,?,?,?,?,?,?)', (item_id,batch_id,row['id'],row['student_ref'],row['student_ref'],'queued',utc_now()))
            connection.execute("INSERT INTO batch_jobs(item_id,content,media_type,auto_grade,answer_layout) VALUES (?,?,?,?,?)",
                               (item_id,b'',row['media_type'],True,'auto'))
    return get_batch(batch_id)
