CREATE TABLE IF NOT EXISTS subjects (
    code VARCHAR(64) PRIMARY KEY,
    name LONGTEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 0
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS assessments (
    id VARCHAR(64) PRIMARY KEY,
    title LONGTEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    status VARCHAR(128) NOT NULL DEFAULT 'draft',
    subject_codes LONGTEXT NOT NULL,
    created_at VARCHAR(128) NOT NULL,
    published_at VARCHAR(128)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS questions (
    id VARCHAR(64) PRIMARY KEY,
    assessment_id VARCHAR(64) NOT NULL,
    question_no VARCHAR(128) NOT NULL,
    prompt LONGTEXT NOT NULL,
    question_type LONGTEXT NOT NULL,
    capture_mode LONGTEXT NOT NULL,
    answer_form LONGTEXT NOT NULL,
    scoring_mode LONGTEXT NOT NULL,
    max_score DOUBLE NOT NULL,
    answer_key LONGTEXT NOT NULL,
    answer_key_state LONGTEXT NOT NULL DEFAULT ('verified'),
    rule_config LONGTEXT NOT NULL,
    rubric LONGTEXT NOT NULL,
    review_policy LONGTEXT NOT NULL,
    created_at VARCHAR(128) NOT NULL,
    UNIQUE(assessment_id, question_no),
    FOREIGN KEY (assessment_id) REFERENCES assessments(id),
    KEY idx_questions_assessment (assessment_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS submissions (
    id VARCHAR(64) PRIMARY KEY,
    assessment_id VARCHAR(64) NOT NULL,
    student_ref LONGTEXT NOT NULL,
    status VARCHAR(128) NOT NULL,
    source_hash VARCHAR(128) NOT NULL,
    source_path LONGTEXT NOT NULL,
    media_type LONGTEXT NOT NULL,
    normalization_report LONGTEXT NOT NULL,
    created_at VARCHAR(128) NOT NULL,
    finalized_at VARCHAR(128),
    FOREIGN KEY (assessment_id) REFERENCES assessments(id),
    KEY idx_submissions_assessment (assessment_id),
    KEY idx_submissions_hash (assessment_id,source_hash)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS submission_pages (
    id VARCHAR(64) PRIMARY KEY,
    submission_id VARCHAR(64) NOT NULL,
    page_no INTEGER NOT NULL,
    image_path LONGTEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    quality_state LONGTEXT NOT NULL,
    quality_report LONGTEXT NOT NULL,
    UNIQUE(submission_id, page_no),
    FOREIGN KEY (submission_id) REFERENCES submissions(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS responses (
    id VARCHAR(64) PRIMARY KEY,
    submission_id VARCHAR(64) NOT NULL,
    question_id VARCHAR(64) NOT NULL,
    raw_text LONGTEXT NOT NULL DEFAULT (''),
    crop_path LONGTEXT,
    ocr_confidence DOUBLE,
    blank_confirmed INTEGER NOT NULL DEFAULT 0,
    extraction_source LONGTEXT NOT NULL DEFAULT ('manual'),
    updated_at VARCHAR(128) NOT NULL,
    UNIQUE(submission_id, question_id),
    FOREIGN KEY (submission_id) REFERENCES submissions(id),
    FOREIGN KEY (question_id) REFERENCES questions(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS grade_items (
    id VARCHAR(64) PRIMARY KEY,
    submission_id VARCHAR(64) NOT NULL,
    question_id VARCHAR(64) NOT NULL,
    response_id VARCHAR(64),
    score DOUBLE,
    max_score DOUBLE NOT NULL,
    score_state LONGTEXT NOT NULL,
    needs_review INTEGER NOT NULL,
    reason_codes LONGTEXT NOT NULL,
    evidence LONGTEXT NOT NULL,
    grader_kind LONGTEXT NOT NULL,
    grader_version LONGTEXT NOT NULL,
    created_at VARCHAR(128) NOT NULL,
    superseded_at VARCHAR(128),
    FOREIGN KEY (submission_id) REFERENCES submissions(id),
    FOREIGN KEY (question_id) REFERENCES questions(id),
    FOREIGN KEY (response_id) REFERENCES responses(id),
    KEY idx_grade_items_submission (submission_id),
    snapshot LONGTEXT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS review_events (
    id VARCHAR(64) PRIMARY KEY,
    grade_item_id VARCHAR(64) NOT NULL,
    reviewer_ref LONGTEXT NOT NULL,
    final_score DOUBLE NOT NULL,
    reason LONGTEXT NOT NULL,
    created_at VARCHAR(128) NOT NULL,
    FOREIGN KEY (grade_item_id) REFERENCES grade_items(id),
    KEY idx_reviews_grade (grade_item_id,created_at),
    rowid BIGINT NOT NULL AUTO_INCREMENT UNIQUE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS knowledge_assets (
    id VARCHAR(64) PRIMARY KEY,
    question_id VARCHAR(64) NOT NULL,
    response_id VARCHAR(64),
    kind VARCHAR(128) NOT NULL,
    status VARCHAR(128) NOT NULL,
    redacted_text LONGTEXT NOT NULL,
    metadata LONGTEXT NOT NULL,
    approved_by LONGTEXT,
    created_at VARCHAR(128) NOT NULL,
    FOREIGN KEY (question_id) REFERENCES questions(id),
    FOREIGN KEY (response_id) REFERENCES responses(id),
    KEY idx_knowledge_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS assessment_documents (
    id VARCHAR(64) PRIMARY KEY,
    assessment_id VARCHAR(64) NOT NULL,
    kind VARCHAR(128) NOT NULL,
    source_hash VARCHAR(128) NOT NULL,
    source_path LONGTEXT NOT NULL,
    media_type LONGTEXT NOT NULL,
    extracted_text LONGTEXT NOT NULL DEFAULT (''),
    extraction_state LONGTEXT NOT NULL,
    page_count INTEGER NOT NULL,
    created_at VARCHAR(128) NOT NULL,
    UNIQUE(assessment_id, kind, source_hash),
    FOREIGN KEY (assessment_id) REFERENCES assessments(id),
    KEY idx_documents_assessment (assessment_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS grading_batches (
    id VARCHAR(64) PRIMARY KEY,
    assessment_id VARCHAR(64) NOT NULL,
    status VARCHAR(128) NOT NULL,
    total_count INTEGER NOT NULL,
    success_count INTEGER NOT NULL DEFAULT 0,
    failed_count INTEGER NOT NULL DEFAULT 0,
    created_at VARCHAR(128) NOT NULL,
    completed_at VARCHAR(128),
    FOREIGN KEY (assessment_id) REFERENCES assessments(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS grading_batch_items (
    id VARCHAR(64) PRIMARY KEY,
    batch_id VARCHAR(64) NOT NULL,
    submission_id VARCHAR(64),
    filename LONGTEXT NOT NULL,
    student_ref LONGTEXT NOT NULL,
    status VARCHAR(128) NOT NULL,
    error LONGTEXT,
    created_at VARCHAR(128) NOT NULL,
    FOREIGN KEY (batch_id) REFERENCES grading_batches(id),
    FOREIGN KEY (submission_id) REFERENCES submissions(id),
    KEY idx_batch_items_submission (submission_id,status),
    KEY idx_batch_items_batch (batch_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS token_usage (
    id VARCHAR(64) PRIMARY KEY,
    assessment_id VARCHAR(64),
    submission_id VARCHAR(64),
    stage VARCHAR(128) NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cached_tokens INTEGER NOT NULL DEFAULT 0,
    created_at VARCHAR(128) NOT NULL,
    FOREIGN KEY (assessment_id) REFERENCES assessments(id),
    FOREIGN KEY (submission_id) REFERENCES submissions(id),
    KEY idx_token_usage_assessment (assessment_id),
    KEY idx_token_usage_submission (submission_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;














CREATE TABLE IF NOT EXISTS batch_jobs (
    item_id VARCHAR(64) PRIMARY KEY,
    content LONGBLOB NOT NULL,
    media_type LONGTEXT,
    auto_grade INTEGER NOT NULL,
    answer_layout LONGTEXT NOT NULL DEFAULT ('auto'),
    FOREIGN KEY (item_id) REFERENCES grading_batch_items(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS performance_events (
    id VARCHAR(64) PRIMARY KEY, batch_id VARCHAR(128), assessment_id VARCHAR(128), submission_id VARCHAR(128),
    stage VARCHAR(128) NOT NULL, elapsed_ms DOUBLE NOT NULL, success INTEGER NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0,
    cached_tokens INTEGER NOT NULL DEFAULT 0, details LONGTEXT NOT NULL, created_at VARCHAR(128) NOT NULL,
    KEY idx_performance_batch (batch_id,stage)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS assessment_layouts (
    assessment_id VARCHAR(64) PRIMARY KEY,
    document_id VARCHAR(64) NOT NULL,
    source_hash VARCHAR(128) NOT NULL, question_fingerprint LONGTEXT NOT NULL, version LONGTEXT NOT NULL,
    payload LONGTEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 1, created_at VARCHAR(128) NOT NULL,
    FOREIGN KEY (assessment_id) REFERENCES assessments(id) ON DELETE CASCADE,
    FOREIGN KEY (document_id) REFERENCES assessment_documents(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;
CREATE TABLE IF NOT EXISTS training_approvals (
    id VARCHAR(64) PRIMARY KEY, grade_item_id VARCHAR(64) NOT NULL,
    review_id VARCHAR(64), redacted_text LONGTEXT NOT NULL,
    approved_by LONGTEXT NOT NULL, created_at VARCHAR(128) NOT NULL,
    FOREIGN KEY (grade_item_id) REFERENCES grade_items(id) ON DELETE CASCADE,
    FOREIGN KEY (review_id) REFERENCES review_events(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS stable_grades (
 cache_key VARCHAR(128) PRIMARY KEY, decision LONGTEXT NOT NULL,
 independent_passes INTEGER NOT NULL, created_at VARCHAR(64) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;
CREATE TABLE IF NOT EXISTS grade_leases (
 cache_key VARCHAR(128) PRIMARY KEY, owner VARCHAR(64) NOT NULL, expires_at DOUBLE NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;
