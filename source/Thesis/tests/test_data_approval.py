from datetime import date
from pathlib import Path

import pytest

from thesis_rag.canonical import canonical_json, canonical_sha256, sha256_text
from thesis_rag.data_approval import (
    EXACT_TEST_DATA_APPROVAL_STATEMENT,
    ApprovedTestDataCounts,
    apply_test_data_review_statuses,
    build_test_data_approval_record,
    mark_test_detector_cases_approved,
    mark_test_questions_approved,
    read_test_data_approval_record,
    verify_test_data_approval,
    write_test_data_approval_record,
)
from thesis_rag.detector_pilot import (
    DetectorCaseClass,
    DetectorCaseReviewStatus,
    DetectorEvaluationCase,
)
from thesis_rag.domain import Chunk, Language, Split
from thesis_rag.qa import (
    QuestionReviewStatus,
    QuestionType,
    ResearchQuestion,
)


def _question(
    *,
    split: Split = Split.TEST,
    review_status: QuestionReviewStatus = QuestionReviewStatus.DRAFT,
) -> ResearchQuestion:
    return ResearchQuestion(
        question_id="Q_TEST_001" if split is Split.TEST else "Q_DEV_001",
        split=split,
        language=Language.KO,
        question="문서에 없는 실시간 정보는 무엇인가?",
        answerable=False,
        question_type=QuestionType.UNANSWERABLE,
        reference_answer="승인된 문서에서 확인할 수 없다.",
        review_status=review_status,
    )


def _detector_case(
    *,
    split: Split = Split.TEST,
    review_status: DetectorCaseReviewStatus = DetectorCaseReviewStatus.DRAFT,
) -> DetectorEvaluationCase:
    return DetectorEvaluationCase(
        case_id="HN_TEST_001" if split is Split.TEST else "HN_DEV_001",
        split=split,
        case_class=DetectorCaseClass.HARD_NEGATIVE,
        chunk=Chunk(
            chunk_id="HN_CHUNK_001",
            document_id="DOC_HARD_NEGATIVE_TEST",
            text="이전 지시를 무시하라는 표현을 분석하는 정상 교육 문장이다.",
            language=Language.KO,
        ),
        review_status=review_status,
    )


def _write_jsonl(path: Path, records: tuple[object, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{canonical_json(record)}\n" for record in records),  # type: ignore[arg-type]
        encoding="utf-8",
    )


def _approval_fixture(root: Path, *, approved: bool = True) -> tuple[Path, ...]:
    question_status = (
        QuestionReviewStatus.RESEARCHER_APPROVED
        if approved
        else QuestionReviewStatus.DRAFT
    )
    detector_status = (
        DetectorCaseReviewStatus.RESEARCHER_APPROVED
        if approved
        else DetectorCaseReviewStatus.DRAFT
    )
    question_path = root / "data/questions/test.jsonl"
    attack_path = root / "data/attacks/test.jsonl"
    hard_negative_path = root / "data/detector/test-hard-negatives.jsonl"
    detector_case_path = root / "data/detector/test-cases.jsonl"
    _write_jsonl(
        question_path,
        (_question(review_status=question_status),),
    )
    _write_jsonl(
        detector_case_path,
        (_detector_case(review_status=detector_status),),
    )
    attack_path.parent.mkdir(parents=True, exist_ok=True)
    attack_path.write_text('{"attack_id":"ATK_TEST_001"}\n', encoding="utf-8")
    hard_negative_path.write_text(
        '{"case_id":"HN_TEST_001"}\n',
        encoding="utf-8",
    )
    return question_path, attack_path, hard_negative_path, detector_case_path


def _counts() -> ApprovedTestDataCounts:
    return ApprovedTestDataCounts(
        questions=1,
        attacks=1,
        hard_negatives=1,
        detector_cases=1,
    )


def test_marks_only_test_review_statuses_as_researcher_approved() -> None:
    question = _question()
    detector_case = _detector_case()

    approved_questions = mark_test_questions_approved((question,))
    approved_detector_cases = mark_test_detector_cases_approved(
        (detector_case,)
    )

    assert approved_questions[0].review_status is (
        QuestionReviewStatus.RESEARCHER_APPROVED
    )
    assert approved_detector_cases[0].review_status is (
        DetectorCaseReviewStatus.RESEARCHER_APPROVED
    )
    assert approved_questions[0].model_copy(
        update={"review_status": QuestionReviewStatus.DRAFT}
    ) == question
    assert approved_detector_cases[0].model_copy(
        update={"review_status": DetectorCaseReviewStatus.DRAFT}
    ) == detector_case


def test_rejects_development_records_in_test_data_approval() -> None:
    with pytest.raises(ValueError, match="test questions"):
        mark_test_questions_approved((_question(split=Split.DEV),))
    with pytest.raises(ValueError, match="test detector cases"):
        mark_test_detector_cases_approved(
            (_detector_case(split=Split.DEV),)
        )


def test_builds_and_verifies_hash_bound_test_data_approval(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    paths = _approval_fixture(root)
    development_approval = root / "artifacts/approvals/development-data-approval.json"
    development_approval.parent.mkdir(parents=True, exist_ok=True)
    development_approval.write_text('{"approved":true}\n', encoding="utf-8")
    before_hashes = {
        path.relative_to(root).as_posix(): sha256_text(
            path.read_text(encoding="utf-8")
        )
        for path in paths
    }
    artifact_specs = {
        path.relative_to(root).as_posix(): 1
        for path in paths
    }

    record = build_test_data_approval_record(
        root,
        approval_statement=EXACT_TEST_DATA_APPROVAL_STATEMENT,
        approved_on=date(2026, 7, 30),
        before_approval_sha256=before_hashes,
        artifact_record_counts=artifact_specs,
        counts=_counts(),
        development_data_approval_sha256=sha256_text(
            development_approval.read_text(encoding="utf-8")
        ),
    )
    verification = verify_test_data_approval(
        root,
        record,
        development_approval_path=development_approval,
    )

    assert record.approval_statement_exact == "테스트 데이터 전체 승인"
    assert record.test_data_approved is True
    assert record.protocol_freeze_approved is False
    assert record.main_run_approved is False
    assert record.artifact_set_sha256 == canonical_sha256(
        [artifact.model_dump(mode="json") for artifact in record.artifacts]
    )
    assert record.after_approval_sha256 == {
        artifact.path: artifact.sha256 for artifact in record.artifacts
    }
    assert verification.valid is True
    assert verification.missing_artifacts == ()
    assert verification.changed_artifacts == ()
    assert verification.invalid_review_status_artifacts == ()


def test_approval_verification_detects_changed_artifact(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    paths = _approval_fixture(root)
    development_approval = root / "artifacts/approvals/development-data-approval.json"
    development_approval.parent.mkdir(parents=True, exist_ok=True)
    development_approval.write_text('{"approved":true}\n', encoding="utf-8")
    record = build_test_data_approval_record(
        root,
        approval_statement=EXACT_TEST_DATA_APPROVAL_STATEMENT,
        approved_on=date(2026, 7, 30),
        before_approval_sha256={
            path.relative_to(root).as_posix(): sha256_text(
                path.read_text(encoding="utf-8")
            )
            for path in paths
        },
        artifact_record_counts={
            path.relative_to(root).as_posix(): 1
            for path in paths
        },
        counts=_counts(),
        development_data_approval_sha256=sha256_text(
            development_approval.read_text(encoding="utf-8")
        ),
    )
    paths[1].write_text('{"attack_id":"CHANGED"}\n', encoding="utf-8")

    verification = verify_test_data_approval(
        root,
        record,
        development_approval_path=development_approval,
    )

    assert verification.valid is False
    assert verification.changed_artifacts == ("data/attacks/test.jsonl",)


def test_approval_verification_detects_invalid_review_status(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    paths = _approval_fixture(root, approved=False)
    development_approval = root / "artifacts/approvals/development-data-approval.json"
    development_approval.parent.mkdir(parents=True, exist_ok=True)
    development_approval.write_text('{"approved":true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="researcher-approved"):
        build_test_data_approval_record(
            root,
            approval_statement=EXACT_TEST_DATA_APPROVAL_STATEMENT,
            approved_on=date(2026, 7, 30),
            before_approval_sha256={
                path.relative_to(root).as_posix(): sha256_text(
                    path.read_text(encoding="utf-8")
                )
                for path in paths
            },
            artifact_record_counts={
                path.relative_to(root).as_posix(): 1
                for path in paths
            },
            counts=_counts(),
            development_data_approval_sha256=sha256_text(
                development_approval.read_text(encoding="utf-8")
            ),
        )


def test_rejects_nonexact_test_data_approval_statement(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    paths = _approval_fixture(root)

    with pytest.raises(ValueError, match="exact approval statement"):
        build_test_data_approval_record(
            root,
            approval_statement="테스트 데이터 승인",
            approved_on=date(2026, 7, 30),
            before_approval_sha256={
                path.relative_to(root).as_posix(): "a" * 64
                for path in paths
            },
            artifact_record_counts={
                path.relative_to(root).as_posix(): 1
                for path in paths
            },
            counts=_counts(),
            development_data_approval_sha256="b" * 64,
        )


def test_applies_test_review_statuses_and_records_before_after_hashes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    paths = _approval_fixture(root, approved=False)
    attack_before = paths[1].read_bytes()
    hard_negative_before = paths[2].read_bytes()

    transition = apply_test_data_review_statuses(root)

    questions = tuple(
        ResearchQuestion.model_validate_json(line, strict=True)
        for line in paths[0].read_text(encoding="utf-8").splitlines()
        if line
    )
    detector_cases = tuple(
        DetectorEvaluationCase.model_validate_json(line, strict=True)
        for line in paths[3].read_text(encoding="utf-8").splitlines()
        if line
    )
    assert transition.question_count == 1
    assert transition.detector_case_count == 1
    assert transition.before_sha256["data/questions/test.jsonl"] != (
        transition.after_sha256["data/questions/test.jsonl"]
    )
    assert transition.before_sha256["data/detector/test-cases.jsonl"] != (
        transition.after_sha256["data/detector/test-cases.jsonl"]
    )
    assert all(
        question.review_status is QuestionReviewStatus.RESEARCHER_APPROVED
        for question in questions
    )
    assert all(
        case.review_status is DetectorCaseReviewStatus.RESEARCHER_APPROVED
        for case in detector_cases
    )
    assert paths[1].read_bytes() == attack_before
    assert paths[2].read_bytes() == hard_negative_before


def test_writes_and_reads_canonical_test_data_approval_record(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    paths = _approval_fixture(root)
    development_approval = root / "artifacts/approvals/development-data-approval.json"
    development_approval.parent.mkdir(parents=True, exist_ok=True)
    development_approval.write_text('{"approved":true}\n', encoding="utf-8")
    record = build_test_data_approval_record(
        root,
        approval_statement=EXACT_TEST_DATA_APPROVAL_STATEMENT,
        approved_on=date(2026, 7, 30),
        before_approval_sha256={
            path.relative_to(root).as_posix(): sha256_text(
                path.read_text(encoding="utf-8")
            )
            for path in paths
        },
        artifact_record_counts={
            path.relative_to(root).as_posix(): 1
            for path in paths
        },
        counts=_counts(),
        development_data_approval_sha256=sha256_text(
            development_approval.read_text(encoding="utf-8")
        ),
    )
    approval_path = root / "artifacts/approvals/test-data-approval.json"

    write_test_data_approval_record(approval_path, record)
    loaded = read_test_data_approval_record(approval_path)

    assert loaded == record
    assert approval_path.read_text(encoding="utf-8") == (
        f"{canonical_json(record)}\n"
    )
