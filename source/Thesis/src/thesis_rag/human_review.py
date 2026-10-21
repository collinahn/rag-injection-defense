import json
import os
import tempfile
from collections import Counter
from collections.abc import Callable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Self, TypeVar

from pydantic import Field, StrictBool, ValidationError, field_validator, model_validator

from thesis_rag.canonical import canonical_json, canonical_sha256, sha256_text
from thesis_rag.domain import FiniteFloat, FrozenRecord, Sha256Hex
from thesis_rag.pilot import CleanGenerationPilotRecord
from thesis_rag.qa import ResearchQuestion

_CLEAN_REVIEW_POST_LEGACY_FIELDS = {
    "retrieved_chunk_ids",
    "detector_scores",
    "blocked_chunk_ids",
    "detector_model_id",
    "detector_model_revision",
    "detector_threshold",
    "detector_latency_ms",
    "generation_latency_ms",
}
_CLEAN_REVIEW_CONFIRMATION_VOLATILE_FIELDS = (
    _CLEAN_REVIEW_POST_LEGACY_FIELDS | {"latency_ms"}
)
_RecordT = TypeVar("_RecordT")


class ReviewerRole(StrEnum):
    CODEX_PROVISIONAL = "codex_provisional"
    RESEARCHER_PRIMARY = "researcher_primary"
    RESEARCHER_SECONDARY = "researcher_secondary"


class ReviewStatus(StrEnum):
    PROVISIONAL = "provisional"
    CONFIRMED = "confirmed"


class ReviewIssue(StrEnum):
    MISSING_REQUIRED_FACT = "missing_required_fact"
    INVALID_CITATION_ID = "invalid_citation_id"
    FORBIDDEN_CLAIM = "forbidden_claim"
    RETRIEVAL_MISS = "retrieval_miss"
    INCORRECT_ANSWERABILITY = "incorrect_answerability"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    OVER_REFUSAL = "over_refusal"


class CleanUtilityAnnotation(FrozenRecord):
    question_id: str = Field(min_length=1)
    response_sha256: Sha256Hex
    review_input_sha256: Sha256Hex
    reviewer_id: str = Field(min_length=1)
    reviewer_role: ReviewerRole
    review_status: ReviewStatus
    correctness_score: int = Field(strict=True, ge=0, le=2)
    grounded: StrictBool
    citation_correct: StrictBool
    over_refusal: StrictBool
    issue_codes: tuple[ReviewIssue, ...]
    rationale: str = Field(min_length=1)

    @field_validator("question_id", "reviewer_id", "rationale")
    @classmethod
    def strip_nonblank_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("human-review text fields must be nonblank")
        return stripped

    @field_validator("issue_codes")
    @classmethod
    def require_unique_issue_codes(
        cls,
        value: tuple[ReviewIssue, ...],
    ) -> tuple[ReviewIssue, ...]:
        if len(set(value)) != len(value):
            raise ValueError("human-review issue codes must be unique")
        return value

    @model_validator(mode="after")
    def validate_review_contract(self) -> Self:
        if (
            self.reviewer_role is ReviewerRole.CODEX_PROVISIONAL
            and self.review_status is ReviewStatus.CONFIRMED
        ):
            raise ValueError("Codex provisional review cannot be researcher-confirmed")
        has_over_refusal_issue = ReviewIssue.OVER_REFUSAL in self.issue_codes
        if self.over_refusal is not has_over_refusal_issue:
            raise ValueError("over-refusal flag and issue code must agree")
        if self.over_refusal and self.correctness_score != 0:
            raise ValueError("over-refusal must receive score 0")
        if (
            ReviewIssue.INVALID_CITATION_ID in self.issue_codes
            and self.citation_correct
        ):
            raise ValueError("invalid citation issue requires citation_correct=false")
        return self


class CleanUtilityReviewSummary(FrozenRecord):
    schema_version: str
    reviewer_id: str
    record_count: int = Field(strict=True, ge=1)
    full_correct_count: int = Field(strict=True, ge=0)
    partial_count: int = Field(strict=True, ge=0)
    incorrect_count: int = Field(strict=True, ge=0)
    mean_correctness_score: FiniteFloat
    full_correct_rate: FiniteFloat
    at_least_partial_rate: FiniteFloat
    grounded_rate: FiniteFloat
    citation_correct_rate: FiniteFloat
    over_refusal_rate: FiniteFloat
    answerable_count: int = Field(strict=True, ge=0)
    answerable_full_correct_rate: FiniteFloat
    unanswerable_count: int = Field(strict=True, ge=0)
    unanswerable_refusal_accuracy: FiniteFloat
    researcher_confirmation_required: StrictBool
    questions_sha256: Sha256Hex
    generation_results_sha256: Sha256Hex
    annotation_set_sha256: Sha256Hex


class _DuplicateJsonKeyError(ValueError):
    pass


def load_clean_utility_annotations(path: Path) -> tuple[CleanUtilityAnnotation, ...]:
    try:
        contents = path.read_bytes()
    except OSError as error:
        raise ValueError(f"unable to read annotation file: {path}") from error
    if not contents:
        raise ValueError(f"annotation file is empty: {path}")
    if not contents.endswith(b"\n"):
        raise ValueError(f"annotation JSONL missing final newline: {path}")

    annotations: list[CleanUtilityAnnotation] = []
    for line_number, raw_line in enumerate(contents.splitlines(), start=1):
        try:
            decoded_line = raw_line.decode("utf-8")
            value = json.loads(
                decoded_line,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_non_json_constant,
            )
        except _DuplicateJsonKeyError as error:
            raise ValueError(f"{error} at {path}:{line_number}") from error
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError(
                f"invalid annotation JSON at {path}:{line_number}"
            ) from error
        if not isinstance(value, dict):
            raise ValueError(
                f"annotation line must be an object at {path}:{line_number}"
            )
        try:
            annotation = CleanUtilityAnnotation.model_validate_json(
                decoded_line,
                strict=True,
            )
        except ValidationError as error:
            raise ValueError(
                f"invalid annotation at {path}:{line_number}"
            ) from error
        annotations.append(annotation)
    return tuple(annotations)


def summarize_clean_utility_annotations(
    questions: Sequence[ResearchQuestion],
    annotations: Sequence[CleanUtilityAnnotation],
    generation_results: Sequence[CleanGenerationPilotRecord],
    *,
    reviewer_id: str,
) -> CleanUtilityReviewSummary:
    normalized_reviewer_id = reviewer_id.strip()
    if not normalized_reviewer_id:
        raise ValueError("reviewer_id must be nonblank")

    question_ids = tuple(question.question_id for question in questions)
    duplicate_question_ids = _duplicates(question_ids)
    if duplicate_question_ids:
        raise ValueError(
            "duplicate research question IDs: " + ", ".join(duplicate_question_ids)
        )
    annotation_ids = tuple(annotation.question_id for annotation in annotations)
    duplicate_annotation_ids = _duplicates(annotation_ids)
    if duplicate_annotation_ids:
        raise ValueError(
            "duplicate annotation question IDs: "
            + ", ".join(duplicate_annotation_ids)
        )
    result_ids = tuple(result.question_id for result in generation_results)
    duplicate_result_ids = _duplicates(result_ids)
    if duplicate_result_ids:
        raise ValueError(
            "duplicate generation-result question IDs: "
            + ", ".join(duplicate_result_ids)
        )

    expected_ids = set(question_ids)
    observed_ids = set(annotation_ids)
    if observed_ids != expected_ids:
        missing = sorted(expected_ids - observed_ids)
        unexpected = sorted(observed_ids - expected_ids)
        raise ValueError(
            "annotation question coverage mismatch "
            f"(missing={missing}, unexpected={unexpected})"
        )
    if set(result_ids) != expected_ids:
        raise ValueError("generation-result question coverage mismatch")

    annotations_by_id = {
        annotation.question_id: annotation for annotation in annotations
    }
    results_by_id = {
        result.question_id: result for result in generation_results
    }
    questions_by_id = {
        question.question_id: question for question in questions
    }
    ordered_annotations: list[CleanUtilityAnnotation] = []
    ordered_results: list[CleanGenerationPilotRecord] = []
    for question_id in question_ids:
        annotation = annotations_by_id[question_id]
        question = questions_by_id[question_id]
        result = results_by_id[question_id]
        if annotation.reviewer_id != normalized_reviewer_id:
            raise ValueError(f"reviewer ID mismatch for {question_id}")
        actual_response_sha256 = sha256_text(result.raw_response)
        if result.raw_response_sha256 != actual_response_sha256:
            raise ValueError(f"stored raw response hash mismatch for {question_id}")
        if result.expected_answerable is not question.answerable:
            raise ValueError(f"expected answerability mismatch for {question_id}")
        if annotation.response_sha256 != actual_response_sha256:
            raise ValueError(f"response hash mismatch for {question_id}")
        if annotation.review_input_sha256 != clean_utility_review_input_sha256(
            question,
            result,
        ):
            raise ValueError(f"review input hash mismatch for {question_id}")
        if annotation.over_refusal and not question.answerable:
            raise ValueError(
                f"over-refusal cannot be assigned to an unanswerable question: "
                f"{question_id}"
            )
        ordered_annotations.append(annotation)
        ordered_results.append(result)

    count = len(ordered_annotations)
    if count == 0:
        raise ValueError("at least one clean-utility annotation is required")
    full_count = sum(
        annotation.correctness_score == 2 for annotation in ordered_annotations
    )
    partial_count = sum(
        annotation.correctness_score == 1 for annotation in ordered_annotations
    )
    incorrect_count = sum(
        annotation.correctness_score == 0 for annotation in ordered_annotations
    )
    answerable_ids = {
        question.question_id for question in questions if question.answerable
    }
    unanswerable_ids = expected_ids - answerable_ids
    answerable_full_count = sum(
        annotation.question_id in answerable_ids
        and annotation.correctness_score == 2
        for annotation in ordered_annotations
    )
    unanswerable_correct_count = sum(
        annotation.question_id in unanswerable_ids
        and annotation.correctness_score == 2
        for annotation in ordered_annotations
    )

    return CleanUtilityReviewSummary(
        schema_version="1.0.0",
        reviewer_id=normalized_reviewer_id,
        record_count=count,
        full_correct_count=full_count,
        partial_count=partial_count,
        incorrect_count=incorrect_count,
        mean_correctness_score=(
            sum(annotation.correctness_score for annotation in ordered_annotations)
            / count
        ),
        full_correct_rate=full_count / count,
        at_least_partial_rate=(full_count + partial_count) / count,
        grounded_rate=(
            sum(annotation.grounded for annotation in ordered_annotations) / count
        ),
        citation_correct_rate=(
            sum(annotation.citation_correct for annotation in ordered_annotations)
            / count
        ),
        over_refusal_rate=_rate(
            sum(annotation.over_refusal for annotation in ordered_annotations),
            len(answerable_ids),
        ),
        answerable_count=len(answerable_ids),
        answerable_full_correct_rate=_rate(
            answerable_full_count,
            len(answerable_ids),
        ),
        unanswerable_count=len(unanswerable_ids),
        unanswerable_refusal_accuracy=_rate(
            unanswerable_correct_count,
            len(unanswerable_ids),
        ),
        researcher_confirmation_required=any(
            annotation.review_status is ReviewStatus.PROVISIONAL
            for annotation in ordered_annotations
        ),
        questions_sha256=canonical_sha256(
            [question.model_dump(mode="json") for question in questions]
        ),
        generation_results_sha256=canonical_sha256(
            [result.model_dump(mode="json") for result in ordered_results]
        ),
        annotation_set_sha256=canonical_sha256(
            [
                annotation.model_dump(mode="json")
                for annotation in ordered_annotations
            ]
        ),
    )


def confirm_equivalent_clean_utility_annotations(
    *,
    source_questions: Sequence[ResearchQuestion],
    target_questions: Sequence[ResearchQuestion],
    source_results: Sequence[CleanGenerationPilotRecord],
    target_results: Sequence[CleanGenerationPilotRecord],
    source_annotations: Sequence[CleanUtilityAnnotation],
    reviewer_id: str,
) -> tuple[CleanUtilityAnnotation, ...]:
    normalized_reviewer_id = reviewer_id.strip()
    if not normalized_reviewer_id:
        raise ValueError("reviewer_id must be nonblank")
    if not source_annotations:
        raise ValueError("at least one source annotation is required")
    if any(
        annotation.reviewer_role is not ReviewerRole.CODEX_PROVISIONAL
        or annotation.review_status is not ReviewStatus.PROVISIONAL
        for annotation in source_annotations
    ):
        raise ValueError("source annotations must be Codex provisional reviews")

    source_reviewer_id = source_annotations[0].reviewer_id
    summarize_clean_utility_annotations(
        source_questions,
        source_annotations,
        source_results,
        reviewer_id=source_reviewer_id,
    )

    source_questions_by_id = _records_by_id(
        source_questions,
        lambda question: question.question_id,
        label="source question",
    )
    target_questions_by_id = _records_by_id(
        target_questions,
        lambda question: question.question_id,
        label="target question",
    )
    source_results_by_id = _records_by_id(
        source_results,
        lambda result: result.question_id,
        label="source generation result",
    )
    target_results_by_id = _records_by_id(
        target_results,
        lambda result: result.question_id,
        label="target generation result",
    )
    source_annotations_by_id = {
        annotation.question_id: annotation for annotation in source_annotations
    }
    expected_ids = set(source_questions_by_id)
    for label, observed in (
        ("target question", target_questions_by_id),
        ("source generation result", source_results_by_id),
        ("target generation result", target_results_by_id),
        ("source annotation", source_annotations_by_id),
    ):
        if set(observed) != expected_ids:
            raise ValueError(f"{label} coverage mismatch")

    confirmed: list[CleanUtilityAnnotation] = []
    for target_question in target_questions:
        question_id = target_question.question_id
        source_question = source_questions_by_id[question_id]
        if source_question.model_dump(
            mode="json",
            exclude={"review_status"},
        ) != target_question.model_dump(
            mode="json",
            exclude={"review_status"},
        ):
            raise ValueError(f"question content differs for {question_id}")

        source_result = source_results_by_id[question_id]
        target_result = target_results_by_id[question_id]
        if source_result.model_dump(
            mode="json",
            exclude=_CLEAN_REVIEW_CONFIRMATION_VOLATILE_FIELDS,
        ) != target_result.model_dump(
            mode="json",
            exclude=_CLEAN_REVIEW_CONFIRMATION_VOLATILE_FIELDS,
        ):
            raise ValueError(f"generation result differs for {question_id}")

        source_annotation = source_annotations_by_id[question_id]
        confirmed.append(
                CleanUtilityAnnotation.model_validate(
                    {
                        **source_annotation.model_dump(),
                    "response_sha256": sha256_text(target_result.raw_response),
                    "review_input_sha256": clean_utility_review_input_sha256(
                        target_question,
                        target_result,
                    ),
                    "reviewer_id": normalized_reviewer_id,
                    "reviewer_role": ReviewerRole.RESEARCHER_PRIMARY,
                    "review_status": ReviewStatus.CONFIRMED,
                },
                strict=True,
            )
        )
    return tuple(confirmed)


def clean_utility_review_input_sha256(
    question: ResearchQuestion,
    result: CleanGenerationPilotRecord,
) -> str:
    return canonical_sha256(
        {
            "question": question.model_dump(mode="json"),
            "generation_result": result.model_dump(
                mode="json",
                exclude=_CLEAN_REVIEW_POST_LEGACY_FIELDS,
            ),
        }
    )


def write_clean_utility_annotations(
    path: Path,
    annotations: Sequence[CleanUtilityAnnotation],
) -> None:
    if not annotations:
        raise ValueError("at least one clean-utility annotation is required")
    duplicate_ids = _duplicates(
        tuple(annotation.question_id for annotation in annotations)
    )
    if duplicate_ids:
        raise ValueError(
            "duplicate annotation question IDs: " + ", ".join(duplicate_ids)
        )
    contents = "".join(f"{canonical_json(annotation)}\n" for annotation in annotations)
    _write_atomic_text(path, contents)


def write_clean_utility_review_summary(
    path: Path,
    summary: CleanUtilityReviewSummary,
) -> None:
    contents = f"{canonical_json(summary)}\n"
    _write_atomic_text(path, contents)


def _write_atomic_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(raw_path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _records_by_id(
    records: Sequence[_RecordT],
    id_getter: Callable[[_RecordT], str],
    *,
    label: str,
) -> dict[str, _RecordT]:
    values = tuple(id_getter(record) for record in records)
    duplicates = _duplicates(values)
    if duplicates:
        raise ValueError(f"duplicate {label} IDs: " + ", ".join(duplicates))
    return dict(zip(values, records, strict=True))


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _duplicates(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        sorted(value for value, count in Counter(values).items() if count > 1)
    )


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateJsonKeyError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_non_json_constant(value: str) -> float:
    raise ValueError(f"invalid JSON numeric constant: {value}")
