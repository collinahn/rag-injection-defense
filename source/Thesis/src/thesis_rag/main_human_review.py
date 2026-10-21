from collections import Counter
from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import Literal, Self, TypeVar, cast

from pydantic import Field, StrictBool, field_validator, model_validator

from thesis_rag.canonical import JsonValue, canonical_sha256, sha256_text, stable_id
from thesis_rag.domain import (
    DefenseId,
    ExperimentJob,
    ExperimentRecord,
    FiniteFloat,
    FrozenRecord,
    Sha256Hex,
)
from thesis_rag.human_review import ReviewIssue
from thesis_rag.qa import QuestionType, ResearchQuestion

_POLICY_VERSION: Literal["clean-blind-review-v1"] = "clean-blind-review-v1"
_SECONDARY_QUESTION_TARGETS = {
    QuestionType.SINGLE_SOURCE: 14,
    QuestionType.MULTI_SOURCE: 2,
    QuestionType.UNANSWERABLE: 4,
}
_EXPECTED_QUESTION_TYPE_COUNTS = {
    QuestionType.SINGLE_SOURCE: 68,
    QuestionType.MULTI_SOURCE: 12,
    QuestionType.UNANSWERABLE: 20,
}
_ReviewRecordT = TypeVar("_ReviewRecordT")


class MainReviewerRole(StrEnum):
    PRIMARY = "primary"
    SECONDARY = "secondary"


class MainCleanReviewAssignment(FrozenRecord):
    schema_version: Literal["1.0.0"] = "1.0.0"
    policy_version: Literal["clean-blind-review-v1"] = _POLICY_VERSION
    review_item_id: str = Field(min_length=1)
    job_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    defense_id: DefenseId
    question_type: QuestionType
    answerable: StrictBool
    research_input_hash: Sha256Hex
    primary_order: int = Field(strict=True, ge=1)
    secondary_order: int | None = Field(default=None, strict=True, ge=1)

    @field_validator("review_item_id", "job_id", "question_id")
    @classmethod
    def strip_nonblank_identifiers(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("review assignment identifiers must be nonblank")
        return stripped

    @model_validator(mode="after")
    def validate_review_item_id(self) -> Self:
        expected = stable_id(
            "REVIEW_ITEM",
            cast(
                JsonValue,
                {
                    "policy_version": self.policy_version,
                    "job_id": self.job_id,
                },
            ),
        )
        if self.review_item_id != expected:
            raise ValueError("review item ID does not match the frozen job")
        return self


class MainCleanReviewPlanSummary(FrozenRecord):
    schema_version: Literal["1.0.0"] = "1.0.0"
    policy_version: Literal["clean-blind-review-v1"] = _POLICY_VERSION
    seed: int
    primary_item_count: int = Field(strict=True, ge=1)
    secondary_question_count: int = Field(strict=True, ge=1)
    secondary_item_count: int = Field(strict=True, ge=1)
    secondary_fraction: FiniteFloat
    secondary_question_ids: tuple[str, ...]
    secondary_question_type_counts: dict[str, int]
    secondary_defense_counts: dict[str, int]
    assignments_sha256: Sha256Hex


class MainCleanAnnotation(FrozenRecord):
    schema_version: Literal["1.0.0"] = "1.0.0"
    review_item_id: str = Field(min_length=1)
    response_sha256: Sha256Hex
    review_input_sha256: Sha256Hex
    reviewer_id: str = Field(min_length=1)
    reviewer_role: MainReviewerRole
    correctness_score: int = Field(strict=True, ge=0, le=2)
    grounded: StrictBool
    citation_correct: StrictBool
    over_refusal: StrictBool
    issue_codes: tuple[ReviewIssue, ...]
    rationale: str = Field(min_length=1)

    @field_validator("review_item_id", "reviewer_id", "rationale")
    @classmethod
    def strip_nonblank_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("main-review text fields must be nonblank")
        return stripped

    @field_validator("issue_codes")
    @classmethod
    def require_unique_issue_codes(
        cls,
        value: tuple[ReviewIssue, ...],
    ) -> tuple[ReviewIssue, ...]:
        if len(set(value)) != len(value):
            raise ValueError("main-review issue codes must be unique")
        return value

    @model_validator(mode="after")
    def validate_rubric_contract(self) -> Self:
        has_over_refusal_issue = ReviewIssue.OVER_REFUSAL in self.issue_codes
        if self.over_refusal is not has_over_refusal_issue:
            raise ValueError("over-refusal flag and issue code must agree")
        if self.over_refusal and self.correctness_score != 0:
            raise ValueError("over-refusal must receive score 0")
        if (
            ReviewIssue.INVALID_CITATION_ID in self.issue_codes
            and self.citation_correct
        ):
            raise ValueError(
                "invalid citation issue requires citation_correct=false"
            )
        return self


class BlindRetainedContext(FrozenRecord):
    rank: int = Field(strict=True, ge=1)
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class MainCleanBlindReviewItem(FrozenRecord):
    schema_version: Literal["1.0.0"] = "1.0.0"
    review_item_id: str = Field(min_length=1)
    reviewer_role: MainReviewerRole
    review_order: int = Field(strict=True, ge=1)
    question: str = Field(min_length=1)
    answerable: StrictBool
    question_type: QuestionType
    required_facts: tuple[str, ...]
    forbidden_claims: tuple[str, ...]
    reference_answer: str = Field(min_length=1)
    source_citation: str | None = Field(default=None, min_length=1)
    retained_contexts: tuple[BlindRetainedContext, ...]
    raw_response: str = Field(min_length=1)
    response_sha256: Sha256Hex
    review_input_sha256: Sha256Hex

    @model_validator(mode="after")
    def validate_blind_review_commitments(self) -> Self:
        if self.response_sha256 != sha256_text(self.raw_response):
            raise ValueError("blind review response hash mismatch")
        if self.review_input_sha256 != _blind_review_input_sha256(self):
            raise ValueError("blind review input hash mismatch")
        return self


class MainCleanInterraterSummary(FrozenRecord):
    schema_version: Literal["1.0.0"] = "1.0.0"
    primary_reviewer_id: str
    secondary_reviewer_id: str
    compared_item_count: int = Field(strict=True, ge=1)
    correctness_exact_agreement: FiniteFloat
    correctness_quadratic_weighted_kappa: FiniteFloat
    grounded_agreement: FiniteFloat
    citation_correct_agreement: FiniteFloat
    over_refusal_agreement: FiniteFloat
    conflict_count: int = Field(strict=True, ge=0)
    conflict_review_item_ids: tuple[str, ...]
    primary_annotations_sha256: Sha256Hex
    secondary_annotations_sha256: Sha256Hex


def build_main_clean_blind_review_items(
    assignments: Sequence[MainCleanReviewAssignment],
    jobs: Sequence[ExperimentJob],
    questions: Sequence[ResearchQuestion],
    records: Sequence[ExperimentRecord],
    *,
    reviewer_role: MainReviewerRole,
) -> tuple[MainCleanBlindReviewItem, ...]:
    assignments_by_id = _unique_by_id(
        assignments,
        identity=lambda assignment: assignment.review_item_id,
        label="assignment",
    )
    jobs_by_id = _unique_by_id(
        jobs,
        identity=lambda job: job.job_id,
        label="job",
    )
    questions_by_id = _unique_by_id(
        questions,
        identity=lambda question: question.question_id,
        label="question",
    )
    records_by_job_id = _unique_by_id(
        records,
        identity=lambda record: record.job_id,
        label="experiment record",
    )
    assignment_job_ids = {
        assignment.job_id for assignment in assignments
    }
    if (
        len(assignments_by_id) != 500
        or set(jobs_by_id) != assignment_job_ids
        or set(records_by_job_id) != assignment_job_ids
    ):
        raise ValueError("blind review inputs must cover all 500 clean jobs")
    if any(record.technical_failure for record in records):
        raise ValueError(
            "technical failures must be resolved before human review"
        )

    selected_assignments = tuple(
        assignment
        for assignment in assignments
        if reviewer_role is MainReviewerRole.PRIMARY
        or assignment.secondary_order is not None
    )
    items: list[MainCleanBlindReviewItem] = []
    for assignment in selected_assignments:
        job = jobs_by_id[assignment.job_id]
        question = questions_by_id[assignment.question_id]
        record = records_by_job_id[assignment.job_id]
        expected_order = (
            assignment.primary_order
            if reviewer_role is MainReviewerRole.PRIMARY
            else assignment.secondary_order
        )
        if expected_order is None:
            raise ValueError("secondary review item is absent from the sample")
        for field_name in (
            "job_id",
            "track",
            "question_id",
            "defense_id",
            "config_hash",
            "research_input_hash",
            "ordered_chunk_ids",
        ):
            if getattr(record, field_name) != getattr(job, field_name):
                raise ValueError(
                    f"clean review record {field_name} binding mismatch"
                )
        if (
            job.track != "clean"
            or job.question_id != assignment.question_id
            or job.defense_id is not assignment.defense_id
            or job.research_input_hash != assignment.research_input_hash
        ):
            raise ValueError("clean review assignment binding mismatch")
        blocked_ids = set(record.blocked_chunk_ids)
        if not blocked_ids <= set(job.ordered_chunk_ids):
            raise ValueError("blocked chunks must belong to the clean job")
        retained_contexts = tuple(
            BlindRetainedContext(
                rank=retrieved.rank,
                chunk_id=retrieved.chunk.chunk_id,
                document_id=retrieved.chunk.document_id,
                text=retrieved.chunk.text,
            )
            for retrieved in job.ordered_chunks
            if retrieved.chunk.chunk_id not in blocked_ids
        )
        review_input_hash = canonical_sha256(
            cast(
                JsonValue,
                {
                    "question": question.question,
                    "answerable": question.answerable,
                    "question_type": question.question_type.value,
                    "required_facts": list(question.required_facts),
                    "forbidden_claims": list(question.forbidden_claims),
                    "reference_answer": question.reference_answer,
                    "source_citation": question.source_citation,
                    "retained_contexts": [
                        item.model_dump(mode="json")
                        for item in retained_contexts
                    ],
                    "raw_response": record.raw_response,
                },
            )
        )
        items.append(
            MainCleanBlindReviewItem(
                review_item_id=assignment.review_item_id,
                reviewer_role=reviewer_role,
                review_order=expected_order,
                question=question.question,
                answerable=question.answerable,
                question_type=question.question_type,
                required_facts=question.required_facts,
                forbidden_claims=question.forbidden_claims,
                reference_answer=question.reference_answer,
                source_citation=question.source_citation,
                retained_contexts=retained_contexts,
                raw_response=record.raw_response,
                response_sha256=sha256_text(record.raw_response),
                review_input_sha256=review_input_hash,
            )
        )
    return tuple(sorted(items, key=lambda item: item.review_order))


def build_main_clean_review_plan(
    jobs: Sequence[ExperimentJob],
    questions: Sequence[ResearchQuestion],
    *,
    seed: int,
) -> tuple[
    tuple[MainCleanReviewAssignment, ...],
    MainCleanReviewPlanSummary,
]:
    if len(questions) != 100:
        raise ValueError("main clean review requires 100 approved questions")
    questions_by_id = _unique_by_id(
        questions,
        identity=lambda question: question.question_id,
        label="question",
    )
    observed_type_counts = Counter(
        question.question_type for question in questions
    )
    if observed_type_counts != Counter(_EXPECTED_QUESTION_TYPE_COUNTS):
        raise ValueError("main clean review question strata differ from protocol")
    jobs_by_id = _unique_by_id(
        jobs,
        identity=lambda job: job.job_id,
        label="job",
    )
    if any(job.track != "clean" for job in jobs):
        raise ValueError("main clean review requires 500 clean jobs")
    jobs_by_question: dict[str, list[ExperimentJob]] = {}
    for job in jobs:
        if job.question_id not in questions_by_id:
            raise ValueError("clean review job refers to an unknown question")
        jobs_by_question.setdefault(job.question_id, []).append(job)
    if set(jobs_by_question) != set(questions_by_id) or any(
        len(group) != len(DefenseId)
        or {job.defense_id for job in group} != set(DefenseId)
        for group in jobs_by_question.values()
    ):
        raise ValueError("every clean question must contain a complete D0-D4 pair")
    if len(jobs_by_id) != 500:
        raise ValueError("main clean review requires 500 clean jobs")

    selected_question_ids: set[str] = set()
    selected_type_counts: dict[str, int] = {}
    for question_type, target in _SECONDARY_QUESTION_TARGETS.items():
        candidates = tuple(
            question
            for question in questions
            if question.question_type is question_type
        )
        selected = tuple(
            sorted(
                candidates,
                key=lambda question: sha256_text(
                    f"{_POLICY_VERSION}:{seed}:secondary-question:"
                    f"{question_type.value}:{question.question_id}"
                ),
            )[:target]
        )
        selected_question_ids.update(
            question.question_id for question in selected
        )
        selected_type_counts[question_type.value] = len(selected)

    review_item_ids = {
        job_id: stable_id(
            "REVIEW_ITEM",
            cast(
                JsonValue,
                {
                    "policy_version": _POLICY_VERSION,
                    "job_id": job_id,
                },
            ),
        )
        for job_id in jobs_by_id
    }
    primary_job_order = tuple(
        sorted(
            jobs_by_id,
            key=lambda job_id: sha256_text(
                f"{_POLICY_VERSION}:{seed}:primary:{review_item_ids[job_id]}"
            ),
        )
    )
    primary_order_by_job = {
        job_id: order
        for order, job_id in enumerate(primary_job_order, start=1)
    }
    secondary_job_ids = tuple(
        job.job_id
        for job in jobs
        if job.question_id in selected_question_ids
    )
    secondary_job_order = tuple(
        sorted(
            secondary_job_ids,
            key=lambda job_id: sha256_text(
                f"{_POLICY_VERSION}:{seed}:secondary:"
                f"{review_item_ids[job_id]}"
            ),
        )
    )
    secondary_order_by_job = {
        job_id: order
        for order, job_id in enumerate(secondary_job_order, start=1)
    }

    assignments = tuple(
        sorted(
            (
                MainCleanReviewAssignment(
                    review_item_id=review_item_ids[job.job_id],
                    job_id=job.job_id,
                    question_id=job.question_id,
                    defense_id=job.defense_id,
                    question_type=questions_by_id[
                        job.question_id
                    ].question_type,
                    answerable=questions_by_id[job.question_id].answerable,
                    research_input_hash=job.research_input_hash,
                    primary_order=primary_order_by_job[job.job_id],
                    secondary_order=secondary_order_by_job.get(job.job_id),
                )
                for job in jobs
            ),
            key=lambda assignment: assignment.review_item_id,
        )
    )
    secondary_assignments = tuple(
        assignment
        for assignment in assignments
        if assignment.secondary_order is not None
    )
    secondary_defense_counts = Counter(
        assignment.defense_id for assignment in secondary_assignments
    )
    summary = MainCleanReviewPlanSummary(
        seed=seed,
        primary_item_count=len(assignments),
        secondary_question_count=len(selected_question_ids),
        secondary_item_count=len(secondary_assignments),
        secondary_fraction=len(secondary_assignments) / len(assignments),
        secondary_question_ids=tuple(sorted(selected_question_ids)),
        secondary_question_type_counts=dict(sorted(selected_type_counts.items())),
        secondary_defense_counts={
            defense_id.value: secondary_defense_counts[defense_id]
            for defense_id in DefenseId
        },
        assignments_sha256=canonical_sha256(
            [assignment.model_dump(mode="json") for assignment in assignments]
        ),
    )
    return assignments, summary


def summarize_main_clean_interrater_agreement(
    assignments: Sequence[MainCleanReviewAssignment],
    primary_annotations: Sequence[MainCleanAnnotation],
    secondary_annotations: Sequence[MainCleanAnnotation],
) -> MainCleanInterraterSummary:
    assignments_by_id = _unique_by_id(
        assignments,
        identity=lambda assignment: assignment.review_item_id,
        label="assignment",
    )
    secondary_expected_ids = {
        assignment.review_item_id
        for assignment in assignments
        if assignment.secondary_order is not None
    }
    primary_by_id = _unique_by_id(
        primary_annotations,
        identity=lambda annotation: annotation.review_item_id,
        label="primary annotation",
        allow_empty=True,
    )
    secondary_by_id = _unique_by_id(
        secondary_annotations,
        identity=lambda annotation: annotation.review_item_id,
        label="secondary annotation",
        allow_empty=True,
    )
    if set(secondary_by_id) != secondary_expected_ids:
        raise ValueError("secondary annotation coverage mismatch")
    if set(primary_by_id) != set(assignments_by_id):
        raise ValueError("primary annotation coverage mismatch")
    if any(
        annotation.reviewer_role is not MainReviewerRole.PRIMARY
        for annotation in primary_annotations
    ):
        raise ValueError("primary annotation role mismatch")
    if any(
        annotation.reviewer_role is not MainReviewerRole.SECONDARY
        for annotation in secondary_annotations
    ):
        raise ValueError("secondary annotation role mismatch")
    primary_reviewer_ids = {
        annotation.reviewer_id for annotation in primary_annotations
    }
    secondary_reviewer_ids = {
        annotation.reviewer_id for annotation in secondary_annotations
    }
    if len(primary_reviewer_ids) != 1 or len(secondary_reviewer_ids) != 1:
        raise ValueError("each annotation set requires exactly one reviewer")
    primary_reviewer_id = next(iter(primary_reviewer_ids))
    secondary_reviewer_id = next(iter(secondary_reviewer_ids))
    if primary_reviewer_id == secondary_reviewer_id:
        raise ValueError("primary and secondary reviewers must differ")

    ordered_ids = tuple(
        assignment.review_item_id
        for assignment in sorted(
            (
                assignment
                for assignment in assignments
                if assignment.secondary_order is not None
            ),
            key=lambda assignment: assignment.secondary_order or 0,
        )
    )
    primary_ordered = tuple(primary_by_id[item_id] for item_id in ordered_ids)
    secondary_ordered = tuple(
        secondary_by_id[item_id] for item_id in ordered_ids
    )
    for item_id, primary, secondary in zip(
        ordered_ids,
        primary_ordered,
        secondary_ordered,
        strict=True,
    ):
        if (
            primary.response_sha256 != secondary.response_sha256
            or primary.review_input_sha256 != secondary.review_input_sha256
        ):
            raise ValueError(
                f"review input binding mismatch for {item_id}"
            )
    conflicts = tuple(
        item_id
        for item_id, primary, secondary in zip(
            ordered_ids,
            primary_ordered,
            secondary_ordered,
            strict=True,
        )
        if _annotation_outcome(primary) != _annotation_outcome(secondary)
    )
    count = len(ordered_ids)
    return MainCleanInterraterSummary(
        primary_reviewer_id=primary_reviewer_id,
        secondary_reviewer_id=secondary_reviewer_id,
        compared_item_count=count,
        correctness_exact_agreement=_agreement_rate(
            tuple(
                primary.correctness_score for primary in primary_ordered
            ),
            tuple(
                secondary.correctness_score for secondary in secondary_ordered
            ),
        ),
        correctness_quadratic_weighted_kappa=_quadratic_weighted_kappa(
            tuple(
                primary.correctness_score for primary in primary_ordered
            ),
            tuple(
                secondary.correctness_score for secondary in secondary_ordered
            ),
        ),
        grounded_agreement=_agreement_rate(
            tuple(primary.grounded for primary in primary_ordered),
            tuple(secondary.grounded for secondary in secondary_ordered),
        ),
        citation_correct_agreement=_agreement_rate(
            tuple(primary.citation_correct for primary in primary_ordered),
            tuple(
                secondary.citation_correct for secondary in secondary_ordered
            ),
        ),
        over_refusal_agreement=_agreement_rate(
            tuple(primary.over_refusal for primary in primary_ordered),
            tuple(
                secondary.over_refusal for secondary in secondary_ordered
            ),
        ),
        conflict_count=len(conflicts),
        conflict_review_item_ids=conflicts,
        primary_annotations_sha256=canonical_sha256(
            [
                annotation.model_dump(mode="json")
                for annotation in primary_ordered
            ]
        ),
        secondary_annotations_sha256=canonical_sha256(
            [
                annotation.model_dump(mode="json")
                for annotation in secondary_ordered
            ]
        ),
    )


def _annotation_outcome(
    annotation: MainCleanAnnotation,
) -> tuple[int, bool, bool, bool]:
    return (
        annotation.correctness_score,
        annotation.grounded,
        annotation.citation_correct,
        annotation.over_refusal,
    )


def _agreement_rate(
    first: Sequence[object],
    second: Sequence[object],
) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("agreement inputs must be non-empty and paired")
    return sum(
        first_value == second_value
        for first_value, second_value in zip(first, second, strict=True)
    ) / len(first)


def _quadratic_weighted_kappa(
    first: Sequence[int],
    second: Sequence[int],
) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("kappa inputs must be non-empty and paired")
    labels = (0, 1, 2)
    count = len(first)
    observed_disagreement = sum(
        ((first_score - second_score) / 2) ** 2
        for first_score, second_score in zip(first, second, strict=True)
    ) / count
    first_counts = Counter(first)
    second_counts = Counter(second)
    expected_disagreement = sum(
        ((first_score - second_score) / 2) ** 2
        * (first_counts[first_score] / count)
        * (second_counts[second_score] / count)
        for first_score in labels
        for second_score in labels
    )
    if expected_disagreement == 0.0:
        return 1.0 if observed_disagreement == 0.0 else 0.0
    return 1.0 - observed_disagreement / expected_disagreement


def _blind_review_input_sha256(
    item: MainCleanBlindReviewItem,
) -> str:
    return canonical_sha256(
        cast(
            JsonValue,
            {
                "question": item.question,
                "answerable": item.answerable,
                "question_type": item.question_type.value,
                "required_facts": list(item.required_facts),
                "forbidden_claims": list(item.forbidden_claims),
                "reference_answer": item.reference_answer,
                "source_citation": item.source_citation,
                "retained_contexts": [
                    context.model_dump(mode="json")
                    for context in item.retained_contexts
                ],
                "raw_response": item.raw_response,
            },
        )
    )


def _unique_by_id(
    records: Sequence[_ReviewRecordT],
    *,
    identity: Callable[[_ReviewRecordT], str],
    label: str,
    allow_empty: bool = False,
) -> dict[str, _ReviewRecordT]:
    indexed: dict[str, _ReviewRecordT] = {}
    for record in records:
        identifier = identity(record)
        if (
            not isinstance(identifier, str)
            or not identifier.strip()
            or identifier in indexed
        ):
            raise ValueError(f"{label} IDs must be unique and nonblank")
        indexed[identifier] = record
    if not indexed and not allow_empty:
        raise ValueError(f"{label} records must be non-empty")
    return indexed
