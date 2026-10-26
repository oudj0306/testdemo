"""Durable local queue. Run the application with one server process."""
from __future__ import annotations

import logging
import hashlib
import json
from contextlib import nullcontext, ExitStack
from datetime import datetime, timezone
from uuid import uuid4
from threading import Event, Thread
from typing import Callable

from ..config import get_settings
from ..db import transaction, utc_now
from ..repository import complete_batch, get_assessment, get_submission, new_id
from .grading import grade_submission
from .work_locks import SubmissionBusyError, submission_work
from .telemetry import scope, timed, event
from .batch_control import BatchSuspended,batch_scope,checkpoint,finishing_result

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
            # Recover durable progress, but require an explicit resume after a service interruption.
            connection.execute("UPDATE grading_batches SET status='paused' WHERE id IN (SELECT batch_id FROM grading_batch_items WHERE status IN ('extracting','grading')) AND status='processing'")
            connection.execute("UPDATE grading_batch_items SET status='queued',error=NULL WHERE status IN ('extracting','grading')")
            connection.execute("UPDATE grading_batches SET total_count=(SELECT COUNT(*) FROM grading_batch_items WHERE batch_id=grading_batches.id) WHERE status IN ('processing','paused','stopped')")
            batches = [r[0] for r in connection.execute("SELECT id FROM grading_batches WHERE status IN ('processing','paused','stopped')")]
        for batch_id in batches:
            complete_batch(batch_id)
        for index in range(2):
            thread = Thread(target=self.run, name=f"grading-{index}", daemon=True)
            thread.start()
            self.threads.append(thread)

    def close(self):
        self.stop.set()
        with transaction() as db:
            db.execute("UPDATE grading_batches SET status='paused' WHERE status='processing'")
        for thread in self.threads:
            thread.join()

    def claim(self):
        with transaction() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("""SELECT i.*,j.content,j.media_type,j.auto_grade,j.answer_layout,b.assessment_id
                FROM grading_batch_items i JOIN batch_jobs j ON j.item_id=i.id
                JOIN grading_batches b ON b.id=i.batch_id
                WHERE i.status='queued' AND b.status='processing' ORDER BY i.created_at LIMIT 1""").fetchone()
            if row:
                if get_settings().grading_profile == 'daily' and row['auto_grade']:
                    rows = connection.execute('''SELECT i.*,j.content,j.media_type,j.auto_grade,j.answer_layout,b.assessment_id
                        FROM grading_batch_items i JOIN batch_jobs j ON j.item_id=i.id
                        JOIN grading_batches b ON b.id=i.batch_id
                        WHERE i.status='queued' AND b.status='processing' AND i.batch_id=? AND j.auto_grade=1
                        ORDER BY i.created_at,i.id LIMIT 8''', (row['batch_id'],)).fetchall()
                    connection.executemany("UPDATE grading_batch_items SET status='extracting' WHERE id=?",
                                           [(r['id'],) for r in rows])
                    return {'daily_group': [dict(r) for r in rows]}
                connection.execute("UPDATE grading_batch_items SET status='extracting' WHERE id=?", (row['id'],))
                return dict(row)
        return None

    def run(self):
        while not self.stop.is_set():
            try:
                job = self.claim()
                if job:
                    if 'daily_group' in job:
                        self.process_daily(job['daily_group'])
                    else:
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
            with guard,batch_scope(job['batch_id'],job['id']):
                checkpoint()
                settings = get_settings()
                with scope(settings, batch_id=job['batch_id'], assessment_id=job['assessment_id'],
                           submission_id=job.get('submission_id')):
                    self.queue_wait(job)
                    self.process_submission(job)
            with transaction() as connection:
                connection.execute("UPDATE grading_batch_items SET status='completed',error=NULL WHERE id=?", (job['id'],))
                # Original has been preserved by ingestion; discard the queue-only copy.
                connection.execute("DELETE FROM batch_jobs WHERE item_id=?", (job['id'],))
        except BatchSuspended:
            self.finish(job,requeue=True)
        except SubmissionBusyError:
            # An interactive request or another worker already owns this
            # submission. Keep the durable job queued instead of failing it or
            # running the same billed work again.
            with transaction() as connection:
                connection.execute("UPDATE grading_batch_items SET status='queued',error=NULL WHERE id=?", (job['id'],))
            self.stop.wait(0.25)
        except Exception as exc:
            self.finish(job,exc)
        finally:
            complete_batch(job['batch_id'])

    def process_submission(self, job):
        checkpoint()
        assessment = get_assessment(job['assessment_id'])
        submission = get_submission(job['submission_id']) if job['submission_id'] else None
        from .hybrid_recognition import needs_recognition
        if submission and self.prepare and needs_recognition(submission,assessment['questions'],get_settings()):
            with timed('extraction'):
                submission = self.prepare(submission['id'])
        if submission is None:
            with timed('extraction'):
                submission = self.persist(assessment, job['student_ref'], job['content'], job['media_type'],
                                          None, job['answer_layout'])
            with transaction() as connection:
                connection.execute("UPDATE grading_batch_items SET submission_id=? WHERE id=?", (submission['id'], job['id']))
        if submission.get('extraction_warning') and submission.get('normalization_report', {}).get('local_recognition', {}).get('mode') not in {'local_only','hybrid'}:
            raise ValueError("识别失败；可打开答卷校对后重批：" + submission['extraction_warning'])
        checkpoint()
        if job['auto_grade']:
            with transaction() as connection:
                connection.execute("UPDATE grading_batch_items SET status='grading' WHERE id=?", (job['id'],))
            grade_submission(submission['id'], get_settings(), pending_only=True)
        return submission

    @staticmethod
    def queue_wait(job):
        wait = max(0, (datetime.now(timezone.utc) - datetime.fromisoformat(job['created_at'])).total_seconds() * 1000)
        event('queue', wait)

    @staticmethod
    def finish(job, error=None, requeue=False):
        with transaction() as db:
            state=db.execute('SELECT b.status FROM grading_batches b JOIN grading_batch_items i ON i.batch_id=b.id WHERE i.id=?',(job['id'],)).fetchone()
            if error and state and state['status'] in {'paused','stopped'}:requeue=True
            db.execute('UPDATE grading_batch_items SET status=?,error=? WHERE id=?',
                       ('queued' if requeue else 'failed' if error else 'completed',
                        str(error)[:500] if error else None, job['id']))
            if not error and not requeue:
                db.execute('DELETE FROM batch_jobs WHERE item_id=?', (job['id'],))

    def process_daily(self, jobs):
        """Extract a bounded cohort first, then grade each shared question."""
        from ..repository import questions_for_submission
        from ..db import utc_now
        from .grading import requires_model,grading_question
        from .bulk_grader import grade_daily_items
        settings = get_settings()
        ready, groups = [], {}
        failed_jobs=set()
        batch_id = jobs[0]['batch_id']
        with ExitStack() as guards,batch_scope(batch_id), scope(settings, batch_id=batch_id,
                                         assessment_id=jobs[0]['assessment_id'], submission_id=None):
            try:
                for job in jobs:
                    try:
                        checkpoint()
                        had_submission = bool(job.get('submission_id'))
                        if job.get('submission_id'):
                            guards.enter_context(submission_work(settings.database_path, job['submission_id'], blocking=False))
                        with batch_scope(batch_id,job['id']),scope(settings, submission_id=job.get('submission_id')):
                            self.queue_wait(job)
                            submission = self.process_submission({**job, 'auto_grade': False})
                        job['submission_id'] = submission['id']
                        if not had_submission:
                            guards.enter_context(submission_work(settings.database_path, submission['id'], blocking=False))
                        if not any(j['submission_id'] == submission['id'] for j, *_ in ready):
                            _, questions, responses = questions_for_submission(submission['id'])
                            current = get_submission(submission['id'])
                            scored = {g['question_id'] for g in current['grade_items'] if g['score'] is not None or g.get('final_score') is not None}
                            pending = [q for q in questions if q['id'] not in scored]
                            expected = {q['id']: (q, responses.get(q['id'])) for q in pending}
                            prepared = {}
                            ready.append((job, prepared, expected, utc_now()))
                            for q in pending:
                                response = responses.get(q['id'])
                                if requires_model(q, response,settings):
                                    # A teacher may edit standards between two extractions.
                                    # Never mix the old and new snapshots in one request.
                                    key = (q['id'], hashlib.sha256(json.dumps(q, ensure_ascii=False,
                                           sort_keys=True).encode()).hexdigest())
                                    groups.setdefault(key, (grading_question(q), []))[1].append((prepared, response))
                        else:
                            self.finish(job)
                    except BatchSuspended:
                        raise
                    except SubmissionBusyError:
                        self.finish(job, requeue=True)
                    except Exception as exc:
                        self.finish(job, exc)
                with transaction() as db:
                    db.executemany("UPDATE grading_batch_items SET status='grading' WHERE id=?", [(j['id'],) for j, *_ in ready])
                for question, members in groups.values():
                    checkpoint()
                    failed_prepared={id(prepared) for job,prepared,*_ in ready if job['id'] in failed_jobs}
                    members=[(prepared,response) for prepared,response in members if id(prepared) not in failed_prepared]
                    if not members:continue
                    items, by_id = [], {}
                    for prepared, response in members:
                        aid = uuid4().hex
                        items.append({'answer_id': aid, 'answer': response['raw_text']})
                        by_id[aid] = prepared
                    decisions = grade_daily_items(question, items, settings)
                    for aid, decision in decisions.items():
                        by_id[aid][question['id']] = decision
                    # Persist every returned answer before honoring pause/stop on the next request.
                    with finishing_result():
                        for job,prepared,expected,started_at in ready:
                            if job['id'] in failed_jobs:continue
                            qid=question['id']
                            if qid in prepared:
                                try:
                                    grade_submission(job['submission_id'],settings,pending_only=True,prepared=prepared,
                                                     expected={qid:expected[qid]},started_at=started_at,question_ids={qid})
                                except Exception as exc:
                                    self.finish(job,exc)
                                    failed_jobs.add(job['id'])
                for job, prepared, expected, started_at in ready:
                    if job['id'] in failed_jobs:continue
                    try:
                        checkpoint()
                        grade_submission(job['submission_id'], settings, pending_only=True,
                                         prepared=prepared, expected=expected, started_at=started_at)
                        self.finish(job)
                    except BatchSuspended:
                        raise
                    except Exception as exc:
                        self.finish(job, exc)
            except BatchSuspended:
                for job in jobs:
                    with transaction() as db:
                        db.execute("UPDATE grading_batch_items SET status='queued',error=NULL WHERE id=? AND status IN ('extracting','grading')",(job['id'],))
            except Exception as exc:
                for job, *_ in ready:
                    self.finish(job, exc)
            finally:
                complete_batch(batch_id)


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
