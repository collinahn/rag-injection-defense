from pathlib import Path

import pytest
from pydantic import ValidationError

from thesis_rag.canonical import canonical_sha256, sha256_text
from thesis_rag.domain import DefenseId, Language, Split
from thesis_rag.human_review import (
    CleanUtilityAnnotation,
    CleanUtilityReviewSummary,
    ReviewerRole,
    ReviewIssue,
    ReviewStatus,
    clean_utility_review_input_sha256,
    confirm_equivalent_clean_utility_annotations,
    load_clean_utility_annotations,
    summarize_clean_utility_annotations,
    write_clean_utility_annotations,
    write_clean_utility_review_summary,
)
from thesis_rag.pilot import CleanGenerationPilotRecord
from thesis_rag.qa import (
    QuestionReviewStatus,
    QuestionType,
    ResearchQuestion,
)


def _question(question_id: str, *, answerable: bool) -> ResearchQuestion:
    if answerable:
        return ResearchQuestion(
            question_id=question_id,
            split=Split.DEV,
            language=Language.KO,
            question="질문",
            answerable=True,
            required_facts=("필수 사실",),
            forbidden_claims=(),
            gold_chunk_ids=("CH_DOC_001_0001_aaaaaaaaaaaa",),
            question_type=QuestionType.SINGLE_SOURCE,
            reference_answer="기준 답변",
            gold_document_ids=("DOC_001",),
            source_citation="문서 > 조항",
            review_status=QuestionReviewStatus.DRAFT,
        )
    return ResearchQuestion(
        question_id=question_id,
        split=Split.DEV,
        language=Language.KO,
        question="문서에 없는 질문",
        answerable=False,
        required_facts=(),
        forbidden_claims=("추정 답변",),
        gold_chunk_ids=(),
        question_type=QuestionType.UNANSWERABLE,
        reference_answer="확인할 수 없다.",
        review_status=QuestionReviewStatus.DRAFT,
    )


def _annotation(
    question: ResearchQuestion,
    result: CleanGenerationPilotRecord,
    *,
    score: int,
    issues: tuple[ReviewIssue, ...] = (),
) -> CleanUtilityAnnotation:
    return CleanUtilityAnnotation(
        question_id=question.question_id,
        response_sha256=sha256_text(result.raw_response),
        review_input_sha256=clean_utility_review_input_sha256(question, result),
        reviewer_id="codex",
        reviewer_role=ReviewerRole.CODEX_PROVISIONAL,
        review_status=ReviewStatus.PROVISIONAL,
        correctness_score=score,
        grounded=score > 0,
        citation_correct=True,
        over_refusal=False,
        issue_codes=issues,
        rationale="사전에 고정한 필수 사실과 모델 답변을 대조했다.",
    )


def _result(
    question_id: str,
    raw_response: str,
    *,
    expected_answerable: bool,
) -> CleanGenerationPilotRecord:
    return CleanGenerationPilotRecord(
        question_id=question_id,
        expected_answerable=expected_answerable,
        defense_id=DefenseId.D0,
        ordered_chunk_ids=("CH_DOC_001_0001_aaaaaaaaaaaa",),
        prompt_sha256="a" * 64,
        raw_response=raw_response,
        raw_response_sha256=sha256_text(raw_response),
        parsed_response=None,
        json_parse_success=False,
        citation_ids_valid=None,
        unknown_citation_ids=(),
        answerability_match=None,
        empty_response=False,
        repetitive_response=False,
        length_limited=False,
        technical_failure=False,
        error_type=None,
        latency_ms=1.0,
        input_tokens=10,
        output_tokens=5,
        finish_reason="stop",
    )


def test_summarizes_bound_provisional_clean_utility_annotations() -> None:
    questions = (
        _question("Q_DEV_001", answerable=True),
        _question("Q_DEV_002", answerable=True),
        _question("Q_DEV_003", answerable=False),
    )
    results = (
        _result("Q_DEV_001", "정답", expected_answerable=True),
        _result("Q_DEV_002", "부분 정답", expected_answerable=True),
        _result("Q_DEV_003", "오답", expected_answerable=False),
    )
    annotations = (
        _annotation(questions[0], results[0], score=2),
        _annotation(
            questions[1],
            results[1],
            score=1,
            issues=(ReviewIssue.MISSING_REQUIRED_FACT,),
        ),
        _annotation(
            questions[2],
            results[2],
            score=0,
            issues=(ReviewIssue.INCORRECT_ANSWERABILITY,),
        ),
    )

    summary = summarize_clean_utility_annotations(
        questions,
        annotations,
        results,
        reviewer_id="codex",
    )

    assert summary.record_count == 3
    assert summary.full_correct_count == 1
    assert summary.partial_count == 1
    assert summary.incorrect_count == 1
    assert summary.mean_correctness_score == pytest.approx(1.0)
    assert summary.at_least_partial_rate == pytest.approx(2 / 3)
    assert summary.answerable_full_correct_rate == pytest.approx(0.5)
    assert summary.unanswerable_refusal_accuracy == pytest.approx(0.0)
    assert summary.researcher_confirmation_required is True


def test_summary_rejects_stale_response_hash() -> None:
    questions = (_question("Q_DEV_001", answerable=True),)
    reviewed_result = _result(
        "Q_DEV_001",
        "이전 답변",
        expected_answerable=True,
    )
    annotations = (_annotation(questions[0], reviewed_result, score=2),)
    current_result = _result(
        "Q_DEV_001",
        "현재 답변",
        expected_answerable=True,
    )

    with pytest.raises(ValueError, match="response hash mismatch"):
        summarize_clean_utility_annotations(
            questions,
            annotations,
            (current_result,),
            reviewer_id="codex",
        )


def test_review_input_hash_preserves_pre_telemetry_record_commitment() -> None:
    question = _question("Q_DEV_001", answerable=True)
    result = _result("Q_DEV_001", "정답", expected_answerable=True)
    legacy_result = result.model_dump(
        mode="json",
        exclude={
            "retrieved_chunk_ids",
            "detector_scores",
            "blocked_chunk_ids",
            "detector_model_id",
            "detector_model_revision",
            "detector_threshold",
            "detector_latency_ms",
            "generation_latency_ms",
        },
    )
    expected = canonical_sha256(
        {
            "question": question.model_dump(mode="json"),
            "generation_result": legacy_result,
        }
    )

    assert clean_utility_review_input_sha256(question, result) == expected


def test_review_input_hash_preserves_legacy_default_valued_fields() -> None:
    question = _question("Q_DEV_001", answerable=True)
    result = _result(
        "Q_DEV_001",
        "정답",
        expected_answerable=True,
    ).model_copy(
        update={
            "input_tokens": None,
            "output_tokens": None,
            "finish_reason": None,
        }
    )
    legacy_result = result.model_dump(
        mode="json",
        exclude={
            "retrieved_chunk_ids",
            "detector_scores",
            "blocked_chunk_ids",
            "detector_model_id",
            "detector_model_revision",
            "detector_threshold",
            "detector_latency_ms",
            "generation_latency_ms",
        },
    )
    expected = canonical_sha256(
        {
            "question": question.model_dump(mode="json"),
            "generation_result": legacy_result,
        }
    )

    assert clean_utility_review_input_sha256(question, result) == expected


def test_summary_rejects_result_with_false_internal_response_hash() -> None:
    questions = (_question("Q_DEV_001", answerable=True),)
    result = _result("Q_DEV_001", "실제 답변", expected_answerable=True)
    annotation = _annotation(questions[0], result, score=2)
    tampered = result.model_copy(
        update={"raw_response_sha256": sha256_text("다른 답변")}
    )

    with pytest.raises(ValueError, match="stored raw response hash mismatch"):
        summarize_clean_utility_annotations(
            questions,
            (annotation,),
            (tampered,),
            reviewer_id="codex",
        )


def test_summary_rejects_question_changed_after_annotation() -> None:
    original = _question("Q_DEV_001", answerable=True)
    result = _result("Q_DEV_001", "답변", expected_answerable=True)
    annotation = _annotation(original, result, score=2)
    changed = original.model_copy(update={"reference_answer": "변경된 기준 답변"})

    with pytest.raises(ValueError, match="review input hash mismatch"):
        summarize_clean_utility_annotations(
            (changed,),
            (annotation,),
            (result,),
            reviewer_id="codex",
        )


def test_confirmation_rebinds_equivalent_review_to_researcher_and_target_hashes(
    tmp_path: Path,
) -> None:
    source_question = _question("Q_DEV_001", answerable=True)
    target_question = source_question.model_copy(
        update={"review_status": QuestionReviewStatus.RESEARCHER_APPROVED}
    )
    source_result = _result(
        source_question.question_id,
        "동일한 답변",
        expected_answerable=True,
    )
    target_result = source_result.model_copy(
        update={
            "latency_ms": 9.0,
            "generation_latency_ms": 8.0,
            "detector_latency_ms": 1.0,
            "retrieved_chunk_ids": source_result.ordered_chunk_ids,
            "detector_scores": {
                source_result.ordered_chunk_ids[0]: 0.1,
            },
            "detector_model_id": "detector-added-after-initial-review",
            "detector_model_revision": "revision-added-after-initial-review",
            "detector_threshold": 0.5,
        }
    )
    source_annotation = _annotation(
        source_question,
        source_result,
        score=1,
        issues=(ReviewIssue.MISSING_REQUIRED_FACT,),
    )

    (confirmed,) = confirm_equivalent_clean_utility_annotations(
        source_questions=(source_question,),
        target_questions=(target_question,),
        source_results=(source_result,),
        target_results=(target_result,),
        source_annotations=(source_annotation,),
        reviewer_id="researcher_primary",
    )
    output_path = tmp_path / "confirmed.jsonl"
    write_clean_utility_annotations(output_path, (confirmed,))

    assert confirmed.reviewer_id == "researcher_primary"
    assert confirmed.reviewer_role is ReviewerRole.RESEARCHER_PRIMARY
    assert confirmed.review_status is ReviewStatus.CONFIRMED
    assert confirmed.correctness_score == source_annotation.correctness_score
    assert confirmed.issue_codes == source_annotation.issue_codes
    assert confirmed.rationale == source_annotation.rationale
    assert confirmed.response_sha256 == target_result.raw_response_sha256
    assert confirmed.review_input_sha256 == clean_utility_review_input_sha256(
        target_question,
        target_result,
    )
    assert load_clean_utility_annotations(output_path) == (confirmed,)


def test_confirmation_rejects_changed_question_content() -> None:
    source_question = _question("Q_DEV_001", answerable=True)
    target_question = source_question.model_copy(
        update={
            "reference_answer": "의미가 바뀐 기준 답변",
            "review_status": QuestionReviewStatus.RESEARCHER_APPROVED,
        }
    )
    result = _result(
        source_question.question_id,
        "답변",
        expected_answerable=True,
    )

    with pytest.raises(ValueError, match="question content differs"):
        confirm_equivalent_clean_utility_annotations(
            source_questions=(source_question,),
            target_questions=(target_question,),
            source_results=(result,),
            target_results=(result,),
            source_annotations=(
                _annotation(source_question, result, score=2),
            ),
            reviewer_id="researcher_primary",
        )


def test_confirmation_rejects_changed_review_relevant_result() -> None:
    question = _question("Q_DEV_001", answerable=True)
    source_result = _result(
        question.question_id,
        "원래 답변",
        expected_answerable=True,
    )
    target_result = _result(
        question.question_id,
        "바뀐 답변",
        expected_answerable=True,
    )

    with pytest.raises(ValueError, match="generation result differs"):
        confirm_equivalent_clean_utility_annotations(
            source_questions=(question,),
            target_questions=(question,),
            source_results=(source_result,),
            target_results=(target_result,),
            source_annotations=(
                _annotation(question, source_result, score=2),
            ),
            reviewer_id="researcher_primary",
        )


def test_summary_rejects_missing_or_duplicate_question_annotations() -> None:
    questions = (
        _question("Q_DEV_001", answerable=True),
        _question("Q_DEV_002", answerable=True),
    )
    result = _result("Q_DEV_001", "답변", expected_answerable=True)
    duplicate = _annotation(questions[0], result, score=2)

    with pytest.raises(ValueError, match="duplicate annotation question IDs"):
        summarize_clean_utility_annotations(
            questions,
            (duplicate, duplicate),
            (
                result,
                _result(
                    "Q_DEV_002",
                    "다른 답변",
                    expected_answerable=True,
                ),
            ),
            reviewer_id="codex",
        )


def test_summary_rejects_missing_annotation_coverage() -> None:
    questions = (
        _question("Q_DEV_001", answerable=True),
        _question("Q_DEV_002", answerable=True),
    )
    results = (
        _result("Q_DEV_001", "답변", expected_answerable=True),
        _result("Q_DEV_002", "다른 답변", expected_answerable=True),
    )

    with pytest.raises(ValueError, match="annotation question coverage mismatch"):
        summarize_clean_utility_annotations(
            questions,
            (_annotation(questions[0], results[0], score=2),),
            results,
            reviewer_id="codex",
        )


def test_codex_provisional_review_cannot_claim_confirmation() -> None:
    with pytest.raises(ValidationError, match="cannot be researcher-confirmed"):
        CleanUtilityAnnotation(
            question_id="Q_DEV_001",
            response_sha256=sha256_text("답변"),
            review_input_sha256="a" * 64,
            reviewer_id="codex",
            reviewer_role=ReviewerRole.CODEX_PROVISIONAL,
            review_status=ReviewStatus.CONFIRMED,
            correctness_score=2,
            grounded=True,
            citation_correct=True,
            over_refusal=False,
            issue_codes=(),
            rationale="판정",
        )


def test_over_refusal_issue_and_flag_must_agree() -> None:
    with pytest.raises(ValidationError, match="must agree"):
        CleanUtilityAnnotation(
            question_id="Q_DEV_001",
            response_sha256=sha256_text("답변"),
            review_input_sha256="a" * 64,
            reviewer_id="codex",
            reviewer_role=ReviewerRole.CODEX_PROVISIONAL,
            review_status=ReviewStatus.PROVISIONAL,
            correctness_score=0,
            grounded=False,
            citation_correct=True,
            over_refusal=False,
            issue_codes=(ReviewIssue.OVER_REFUSAL,),
            rationale="거부 여부 판정",
        )


def test_summary_rejects_over_refusal_on_unanswerable_question() -> None:
    question = _question("Q_DEV_001", answerable=False)
    result = _result(
        "Q_DEV_001",
        "확인할 수 없습니다.",
        expected_answerable=False,
    )
    annotation = CleanUtilityAnnotation(
        question_id=question.question_id,
        response_sha256=result.raw_response_sha256,
        review_input_sha256=clean_utility_review_input_sha256(question, result),
        reviewer_id="codex",
        reviewer_role=ReviewerRole.CODEX_PROVISIONAL,
        review_status=ReviewStatus.PROVISIONAL,
        correctness_score=0,
        grounded=False,
        citation_correct=True,
        over_refusal=True,
        issue_codes=(ReviewIssue.OVER_REFUSAL,),
        rationale="잘못된 과잉 거부 표시",
    )

    with pytest.raises(ValueError, match="unanswerable question"):
        summarize_clean_utility_annotations(
            (question,),
            (annotation,),
            (result,),
            reviewer_id="codex",
        )


def test_over_refusal_rate_uses_only_answerable_questions_as_denominator() -> None:
    answerable = _question("Q_DEV_001", answerable=True)
    unanswerable = _question("Q_DEV_002", answerable=False)
    answerable_result = _result(
        answerable.question_id,
        "근거가 있지만 답변을 거부함",
        expected_answerable=True,
    )
    unanswerable_result = _result(
        unanswerable.question_id,
        "확인할 수 없습니다.",
        expected_answerable=False,
    )
    over_refusal = CleanUtilityAnnotation(
        question_id=answerable.question_id,
        response_sha256=answerable_result.raw_response_sha256,
        review_input_sha256=clean_utility_review_input_sha256(
            answerable,
            answerable_result,
        ),
        reviewer_id="codex",
        reviewer_role=ReviewerRole.CODEX_PROVISIONAL,
        review_status=ReviewStatus.PROVISIONAL,
        correctness_score=0,
        grounded=False,
        citation_correct=True,
        over_refusal=True,
        issue_codes=(ReviewIssue.OVER_REFUSAL,),
        rationale="답변 가능한 질문을 잘못 거부했다.",
    )

    summary = summarize_clean_utility_annotations(
        (answerable, unanswerable),
        (
            over_refusal,
            _annotation(unanswerable, unanswerable_result, score=2),
        ),
        (answerable_result, unanswerable_result),
        reviewer_id="codex",
    )

    assert summary.over_refusal_rate == pytest.approx(1.0)


def test_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "annotations.jsonl"
    path.write_text(
        '{"question_id":"Q_DEV_001","question_id":"Q_DEV_002"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_clean_utility_annotations(path)


def test_atomic_summary_write_preserves_previous_file_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "summary.json"
    path.write_text("previous\n", encoding="utf-8")
    summary = CleanUtilityReviewSummary(
        schema_version="1.0.0",
        reviewer_id="codex",
        record_count=1,
        full_correct_count=1,
        partial_count=0,
        incorrect_count=0,
        mean_correctness_score=2.0,
        full_correct_rate=1.0,
        at_least_partial_rate=1.0,
        grounded_rate=1.0,
        citation_correct_rate=1.0,
        over_refusal_rate=0.0,
        answerable_count=1,
        answerable_full_correct_rate=1.0,
        unanswerable_count=0,
        unanswerable_refusal_accuracy=0.0,
        researcher_confirmation_required=True,
        questions_sha256="a" * 64,
        generation_results_sha256="b" * 64,
        annotation_set_sha256="c" * 64,
    )

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError(f"cannot replace {source} with {destination}")

    monkeypatch.setattr("thesis_rag.human_review.os.replace", fail_replace)

    with pytest.raises(OSError, match="cannot replace"):
        write_clean_utility_review_summary(path, summary)

    assert path.read_text(encoding="utf-8") == "previous\n"
