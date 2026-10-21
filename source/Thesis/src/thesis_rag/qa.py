from collections.abc import Sequence
from enum import StrEnum
from typing import Self

from pydantic import Field, field_validator, model_validator

from thesis_rag.domain import (
    Chunk,
    FrozenRecord,
    Question,
    Split,
    normalize_research_text,
)

_SCHOOL_CONFIRMATION_REVIEW_CODES = frozenset(
    {
        "CAPSTONE_THESIS_STAGE_TIMING_REVIEW",
        "CROSS_CAMPUS_DOUBLE_MAJOR_RESTRICTION_REVIEW",
    }
)
_EXCLUDED_GENERAL_QA_CHUNK_IDS = frozenset(
    {
        "CH_DOC_008_0024_af64e8e7552a",
        "CH_DOC_008_0025_4bd6219d2430",
        "CH_DOC_008_0026_02c7e1a02f6c",
        "CH_DOC_008_0027_495b63bf7678",
        "CH_DOC_008_0028_0a61c95d9264",
    }
)
class QuestionType(StrEnum):
    SINGLE_SOURCE = "single_source"
    MULTI_SOURCE = "multi_source"
    UNANSWERABLE = "unanswerable"


class QuestionReviewStatus(StrEnum):
    DRAFT = "draft"
    RESEARCHER_APPROVED = "researcher_approved"


class UserContextField(StrEnum):
    ADMISSION_YEAR = "admission_year"
    CAMPUS = "campus"
    COLLEGE = "college"
    DEPARTMENT = "department"
    PROGRAM_TYPE = "program_type"
    TEACHER_EDUCATION_STATUS = "teacher_education_status"
    DOUBLE_TEACHING_STATUS = "double_teaching_status"
    QUESTION_SCOPE = "question_scope"
    REGISTERED_SEMESTERS = "registered_semesters"
    APPROVAL_STATUS = "approval_status"
    TARGET_CAMPUS = "target_campus"
    TARGET_DEPARTMENT = "target_department"


_REVIEW_CODE_CONTEXT_REQUIREMENTS = {
    "CONVERGENCE_MAJOR_CREDIT_COHORT_REVIEW": frozenset(
        {
            UserContextField.ADMISSION_YEAR,
            UserContextField.QUESTION_SCOPE,
        }
    ),
    "DOUBLE_MAJOR_CREDIT_SCOPE_REVIEW": frozenset(
        {
            UserContextField.ADMISSION_YEAR,
            UserContextField.PROGRAM_TYPE,
            UserContextField.QUESTION_SCOPE,
        }
    ),
}


class ResearchQuestion(Question):
    question_type: QuestionType
    reference_answer: str = Field(min_length=1)
    acceptable_variants: tuple[str, ...] = ()
    gold_document_ids: tuple[str, ...] = ()
    source_citation: str | None = None
    required_user_context: tuple[UserContextField, ...] = ()
    review_code_resolutions: dict[str, str] = Field(default_factory=dict)
    notes: str = ""
    review_status: QuestionReviewStatus

    @field_validator(
        "required_facts",
        "forbidden_claims",
        "acceptable_variants",
        "gold_document_ids",
        "gold_chunk_ids",
    )
    @classmethod
    def validate_unique_nonblank_strings(
        cls,
        value: tuple[str, ...],
        info: object,
    ) -> tuple[str, ...]:
        field_name = getattr(info, "field_name", "field")
        normalized = tuple(item.strip() for item in value)
        if any(not item for item in normalized):
            raise ValueError(f"{field_name} must contain nonblank strings")
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"{field_name} must contain unique strings")
        return normalized

    @field_validator("required_user_context")
    @classmethod
    def validate_unique_user_context(
        cls,
        value: tuple[UserContextField, ...],
    ) -> tuple[UserContextField, ...]:
        if len(set(value)) != len(value):
            raise ValueError("required_user_context must contain unique fields")
        return value

    @field_validator("review_code_resolutions")
    @classmethod
    def validate_review_code_resolutions(
        cls,
        value: dict[str, str],
    ) -> dict[str, str]:
        normalized = {
            code.strip(): resolution.strip()
            for code, resolution in value.items()
        }
        if any(not code or not resolution for code, resolution in normalized.items()):
            raise ValueError(
                "review_code_resolutions must contain nonblank keys and values"
            )
        return normalized

    @field_validator("reference_answer")
    @classmethod
    def strip_reference_answer(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("reference_answer must be nonblank")
        return stripped

    @field_validator("source_citation")
    @classmethod
    def strip_source_citation(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("source_citation must be nonblank when present")
        return stripped

    @model_validator(mode="after")
    def validate_question_contract(self) -> Self:
        expected_prefix = "Q_DEV_" if self.split is Split.DEV else "Q_TEST_"
        if not self.question_id.startswith(expected_prefix):
            raise ValueError("question_id prefix must match split")

        if self.question_type is QuestionType.SINGLE_SOURCE:
            if (
                not self.answerable
                or len(self.gold_document_ids) != 1
                or not self.gold_chunk_ids
            ):
                raise ValueError(
                    "single-source question must be answerable with one gold "
                    "document and at least one gold chunk"
                )
        elif self.question_type is QuestionType.MULTI_SOURCE:
            if (
                not self.answerable
                or len(self.gold_document_ids) < 2
                or len(self.gold_chunk_ids) < 2
            ):
                raise ValueError(
                    "multi-source question must be answerable with at least two "
                    "gold documents and chunks"
                )
        elif (
            self.answerable
            or self.required_facts
            or self.gold_document_ids
            or self.gold_chunk_ids
            or self.source_citation is not None
        ):
            raise ValueError(
                "unanswerable question must not declare facts, citations, or gold sources"
            )

        if self.answerable and (
            not self.required_facts or self.source_citation is None
        ):
            raise ValueError(
                "answerable question must declare required facts and a source citation"
            )
        return self


class QuestionValidationSummary(FrozenRecord):
    question_count: int = Field(strict=True, ge=0)
    answerable_count: int = Field(strict=True, ge=0)
    single_source_count: int = Field(strict=True, ge=0)
    multi_source_count: int = Field(strict=True, ge=0)
    unanswerable_count: int = Field(strict=True, ge=0)
    draft_count: int = Field(strict=True, ge=0)
    researcher_approved_count: int = Field(strict=True, ge=0)
    source_document_count: int = Field(strict=True, ge=0)
    gold_chunk_count: int = Field(strict=True, ge=0)


def validate_research_questions(
    questions: Sequence[ResearchQuestion],
    chunks: Sequence[Chunk],
) -> QuestionValidationSummary:
    questions_by_normalized_text: dict[str, list[str]] = {}
    for question in questions:
        normalized = normalize_research_text(question.question)
        questions_by_normalized_text.setdefault(normalized, []).append(
            question.question_id
        )
    duplicate_question_ids = sorted(
        ",".join(sorted(question_ids))
        for question_ids in questions_by_normalized_text.values()
        if len(question_ids) > 1
    )
    if duplicate_question_ids:
        raise ValueError(
            "duplicate normalized questions: "
            f"{';'.join(duplicate_question_ids)}"
        )

    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    chunk_ids = set(chunks_by_id)
    for question in questions:
        missing_chunk_ids = sorted(set(question.gold_chunk_ids) - chunk_ids)
        if missing_chunk_ids:
            raise ValueError(
                "missing gold chunks for "
                f"{question.question_id}:{','.join(missing_chunk_ids)}"
            )
        excluded_chunk_ids = sorted(
            set(question.gold_chunk_ids) & _EXCLUDED_GENERAL_QA_CHUNK_IDS
        )
        if excluded_chunk_ids:
            raise ValueError(
                "excluded general-QA chunks for "
                f"{question.question_id}:{','.join(excluded_chunk_ids)}"
            )
        gold_document_ids = {
            chunks_by_id[chunk_id].document_id for chunk_id in question.gold_chunk_ids
        }
        if gold_document_ids != set(question.gold_document_ids):
            raise ValueError(
                "gold document mismatch for "
                f"{question.question_id}:declared={','.join(question.gold_document_ids)};"
                f"actual={','.join(sorted(gold_document_ids))}"
            )
        review_codes = {
            code
            for chunk_id in question.gold_chunk_ids
            for code in _chunk_review_codes(chunks_by_id[chunk_id])
        }
        unresolved_codes = sorted(review_codes & _SCHOOL_CONFIRMATION_REVIEW_CODES)
        if unresolved_codes:
            raise ValueError(
                "school confirmation required for "
                f"{question.question_id}:{','.join(unresolved_codes)}"
            )
        missing_resolutions = sorted(
            review_codes - set(question.review_code_resolutions)
        )
        if missing_resolutions:
            raise ValueError(
                "missing review-code resolutions for "
                f"{question.question_id}:{','.join(missing_resolutions)}"
            )
        unused_resolutions = sorted(
            set(question.review_code_resolutions) - review_codes
        )
        if unused_resolutions:
            raise ValueError(
                "unused review-code resolutions for "
                f"{question.question_id}:{','.join(unused_resolutions)}"
            )
        required_context = {
            field
            for code in review_codes
            for field in _REVIEW_CODE_CONTEXT_REQUIREMENTS.get(code, ())
        }
        missing_context = sorted(
            required_context - set(question.required_user_context),
            key=lambda field: field.value,
        )
        if missing_context:
            raise ValueError(
                "missing required user context for "
                f"{question.question_id}:"
                f"{','.join(field.value for field in missing_context)}"
            )
    return QuestionValidationSummary(
        question_count=len(questions),
        answerable_count=sum(question.answerable for question in questions),
        single_source_count=sum(
            question.question_type is QuestionType.SINGLE_SOURCE
            for question in questions
        ),
        multi_source_count=sum(
            question.question_type is QuestionType.MULTI_SOURCE
            for question in questions
        ),
        unanswerable_count=sum(
            question.question_type is QuestionType.UNANSWERABLE
            for question in questions
        ),
        draft_count=sum(
            question.review_status is QuestionReviewStatus.DRAFT
            for question in questions
        ),
        researcher_approved_count=sum(
            question.review_status is QuestionReviewStatus.RESEARCHER_APPROVED
            for question in questions
        ),
        source_document_count=len(
            {
                document_id
                for question in questions
                for document_id in question.gold_document_ids
            }
        ),
        gold_chunk_count=len(
            {
                chunk_id
                for question in questions
                for chunk_id in question.gold_chunk_ids
            }
        ),
    )


def _chunk_review_codes(chunk: Chunk) -> tuple[str, ...]:
    raw_value = chunk.metadata.get("review_codes")
    if not isinstance(raw_value, str) or not raw_value:
        return ()
    return tuple(code for code in raw_value.split(",") if code)


def render_question_review_sheet(
    questions: Sequence[ResearchQuestion],
    chunks: Sequence[Chunk],
    *,
    questions_sha256: str,
    chunks_sha256: str,
) -> str:
    summary = validate_research_questions(questions, chunks)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    question_splits = {question.split for question in questions}
    dataset_label = "테스트" if question_splits == {Split.TEST} else "개발"
    all_questions_approved = (
        summary.question_count > 0 and summary.draft_count == 0
    )
    if all_questions_approved:
        review_notice = [
            "이 문서는 자동 생성된 검토 보조자료다. 모든 질문은 연구자가 승인했다.",
            "이 승인은 protocol freeze 또는 본실험 실행 승인을 의미하지 않는다.",
        ]
    else:
        review_notice = (
            [
                "이 문서는 자동 생성된 검토 보조자료다. 테스트 데이터 승인 전까지",
                "확정 정답이나 동결 데이터로 취급하지 않는다.",
            ]
            if dataset_label == "테스트"
            else [
                "이 문서는 자동 생성된 검토 보조자료다. 모든 질문은 연구자가 직접 근거를",
                "확인하기 전까지 확정 정답이나 동결 데이터로 취급하지 않는다.",
            ]
        )
    lines = [
        f"# {dataset_label} QA 연구자 검토표",
        "",
        *review_notice,
        "",
        f"- 질문 파일 SHA-256: `{questions_sha256}`",
        f"- 코퍼스 파일 SHA-256: `{chunks_sha256}`",
        f"- 전체 질문: {summary.question_count}",
        (
            "- 구성: "
            f"단일 출처 {summary.single_source_count}, "
            f"복수 출처 {summary.multi_source_count}, "
            f"답변 불가 {summary.unanswerable_count}"
        ),
        (
            "- 검토 상태: "
            f"초안 {summary.draft_count}, "
            f"연구자 승인 {summary.researcher_approved_count}"
        ),
        "",
    ]
    for question in questions:
        approval_mark = (
            "x"
            if question.review_status is QuestionReviewStatus.RESEARCHER_APPROVED
            else " "
        )
        lines.extend(
            [
                f"## {question.question_id}",
                "",
                f"- 상태: `{question.review_status.value}`",
                f"- 유형: `{question.question_type.value}`",
                f"- 질문: {question.question}",
                f"- 기준 답변: {question.reference_answer}",
                f"- 출처 표기: {question.source_citation or '없음'}",
                (
                    "- 필요한 사용자 문맥: "
                    + (
                        ", ".join(
                            field.value for field in question.required_user_context
                        )
                        or "없음"
                    )
                ),
                f"- 작성 메모: {question.notes or '없음'}",
                "",
                "### 필수 사실",
                "",
            ]
        )
        lines.extend(
            f"- {fact}" for fact in question.required_facts
        )
        if not question.required_facts:
            lines.append("- 없음 (답변 불가 질문)")
        lines.extend(["", "### 금지 주장", ""])
        lines.extend(
            f"- {claim}" for claim in question.forbidden_claims
        )
        if not question.forbidden_claims:
            lines.append("- 없음")
        lines.extend(["", "### 검토 코드 해소 기록", ""])
        lines.extend(
            f"- `{code}`: {resolution}"
            for code, resolution in sorted(
                question.review_code_resolutions.items()
            )
        )
        if not question.review_code_resolutions:
            lines.append("- 없음")
        lines.extend(["", "### 골드 근거", ""])
        if question.gold_chunk_ids:
            for chunk_id in question.gold_chunk_ids:
                chunk = chunks_by_id[chunk_id]
                source_title = chunk.metadata.get("source_title")
                source_label = (
                    source_title
                    if isinstance(source_title, str) and source_title
                    else chunk.document_id
                )
                lines.extend(
                    [
                        f"#### `{chunk_id}` · {source_label}",
                        "",
                    ]
                )
                lines.extend(
                    f"> {line}" if line else ">"
                    for line in chunk.text.splitlines()
                )
                lines.append("")
        else:
            lines.extend(
                [
                    "- 없음. 코퍼스 범위 밖 질문인지 확인한다.",
                    "",
                ]
            )
        lines.extend(
            [
                "### 연구자 판정",
                "",
                f"- [{approval_mark}] 질문이 하나의 의미로 명확하게 읽힌다.",
                f"- [{approval_mark}] 정답과 필수 사실이 근거 문서와 일치한다.",
                f"- [{approval_mark}] 금지 주장이 핵심 오류를 적절히 포착한다.",
                f"- [{approval_mark}] 골드 청크가 충분하며 불필요한 청크가 없다.",
                (
                    f"- [{approval_mark}] 사용자 문맥 또는 답변 불가 판정이 "
                    "적절하다."
                ),
                (
                    f"- [{approval_mark}] 이 질문을 연구자 승인 상태로 "
                    "전환한다."
                ),
                "- [ ] 수정 필요",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"
