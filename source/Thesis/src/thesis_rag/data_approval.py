import hashlib
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Final, Literal, Self, TypeVar

from pydantic import Field, StrictBool, field_validator, model_validator

from thesis_rag.canonical import canonical_json, canonical_sha256
from thesis_rag.detector_pilot import (
    DetectorCaseReviewStatus,
    DetectorEvaluationCase,
)
from thesis_rag.domain import FrozenRecord, Sha256Hex, Split
from thesis_rag.qa import (
    QuestionReviewStatus,
    ResearchQuestion,
)

EXACT_TEST_DATA_APPROVAL_STATEMENT: Final[
    Literal["테스트 데이터 전체 승인"]
] = "테스트 데이터 전체 승인"
_QUESTION_PATH = "data/questions/test.jsonl"
_DETECTOR_CASE_PATH = "data/detector/test-cases.jsonl"
_REQUIRED_DATA_PATHS = frozenset(
    {
        _QUESTION_PATH,
        "data/attacks/test.jsonl",
        "data/detector/test-hard-negatives.jsonl",
        _DETECTOR_CASE_PATH,
    }
)
_RecordT = TypeVar(
    "_RecordT",
    ResearchQuestion,
    DetectorEvaluationCase,
)


class ApprovalArtifact(FrozenRecord):
    path: str
    sha256: Sha256Hex
    size_bytes: int = Field(strict=True, ge=0)
    record_count: int | None = Field(
        default=None,
        strict=True,
        ge=0,
        exclude_if=lambda value: value is None,
    )

    @field_validator("path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        normalized = PurePosixPath(value)
        if (
            not value
            or normalized.is_absolute()
            or any(part in {"", ".", ".."} for part in normalized.parts)
            or normalized.as_posix() != value
        ):
            raise ValueError("approval artifact path must be a normalized relative path")
        return value


class ApprovedTestDataCounts(FrozenRecord):
    questions: int = Field(strict=True, ge=1)
    attacks: int = Field(strict=True, ge=1)
    hard_negatives: int = Field(strict=True, ge=1)
    detector_cases: int = Field(strict=True, ge=1)


class TestDataApprovalRecord(FrozenRecord):
    schema_version: Literal["1.0.0"]
    approval_id: str
    approval_statement_exact: Literal["테스트 데이터 전체 승인"]
    approved_on: date
    approver_role: Literal["researcher_primary"]
    scope: tuple[str, ...]
    counts: ApprovedTestDataCounts
    before_approval_sha256: dict[str, Sha256Hex]
    after_approval_sha256: dict[str, Sha256Hex]
    artifacts: tuple[ApprovalArtifact, ...]
    artifact_set_sha256: Sha256Hex
    development_data_approval_sha256: Sha256Hex
    test_data_approved: StrictBool
    protocol_freeze_approved: StrictBool
    main_run_approved: StrictBool

    @field_validator("approval_id")
    @classmethod
    def require_nonblank_approval_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("approval_id must be nonblank")
        return value

    @model_validator(mode="after")
    def validate_commitments_and_gate_flags(self) -> Self:
        paths = tuple(artifact.path for artifact in self.artifacts)
        if tuple(sorted(paths)) != paths or len(set(paths)) != len(paths):
            raise ValueError("approval artifacts must be unique and sorted")
        expected_artifact_set = canonical_sha256(
            [
                artifact.model_dump(mode="json")
                for artifact in self.artifacts
            ]
        )
        if expected_artifact_set != self.artifact_set_sha256:
            raise ValueError("approval artifact-set hash mismatch")
        expected_after_hashes = {
            artifact.path: artifact.sha256 for artifact in self.artifacts
        }
        if self.after_approval_sha256 != expected_after_hashes:
            raise ValueError("approval after-hash map does not match artifacts")
        if not _REQUIRED_DATA_PATHS.issubset(paths):
            raise ValueError("approval record is missing required test data artifacts")
        if not self.test_data_approved:
            raise ValueError("test-data approval record must approve test data")
        if self.protocol_freeze_approved or self.main_run_approved:
            raise ValueError(
                "test-data approval must not approve protocol freeze or main run"
            )
        return self


class TestDataApprovalVerification(FrozenRecord):
    missing_artifacts: tuple[str, ...] = ()
    changed_artifacts: tuple[str, ...] = ()
    invalid_review_status_artifacts: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not (
            self.missing_artifacts
            or self.changed_artifacts
            or self.invalid_review_status_artifacts
        )


class TestDataStatusTransition(FrozenRecord):
    question_count: int = Field(strict=True, ge=1)
    detector_case_count: int = Field(strict=True, ge=1)
    before_sha256: dict[str, Sha256Hex]
    after_sha256: dict[str, Sha256Hex]


def mark_test_questions_approved(
    questions: Sequence[ResearchQuestion],
) -> tuple[ResearchQuestion, ...]:
    if not questions:
        raise ValueError("test questions must not be empty")
    if any(question.split is not Split.TEST for question in questions):
        raise ValueError("test questions may only contain the test split")
    return tuple(
        question.model_copy(
            update={"review_status": QuestionReviewStatus.RESEARCHER_APPROVED}
        )
        for question in questions
    )


def mark_test_detector_cases_approved(
    cases: Sequence[DetectorEvaluationCase],
) -> tuple[DetectorEvaluationCase, ...]:
    if not cases:
        raise ValueError("test detector cases must not be empty")
    if any(case.split is not Split.TEST for case in cases):
        raise ValueError("test detector cases may only contain the test split")
    return tuple(
        case.model_copy(
            update={
                "review_status": (
                    DetectorCaseReviewStatus.RESEARCHER_APPROVED
                )
            }
        )
        for case in cases
    )


def apply_test_data_review_statuses(
    root: Path,
) -> TestDataStatusTransition:
    question_path = _resolve_artifact_path(root, _QUESTION_PATH)
    detector_case_path = _resolve_artifact_path(
        root,
        _DETECTOR_CASE_PATH,
    )
    original_contents = {
        _QUESTION_PATH: question_path.read_bytes(),
        _DETECTOR_CASE_PATH: detector_case_path.read_bytes(),
    }
    questions = _read_jsonl(question_path, ResearchQuestion)
    detector_cases = _read_jsonl(
        detector_case_path,
        DetectorEvaluationCase,
    )
    approved_questions = mark_test_questions_approved(questions)
    approved_detector_cases = mark_test_detector_cases_approved(
        detector_cases
    )
    approved_contents = {
        _QUESTION_PATH: _canonical_jsonl(approved_questions).encode("utf-8"),
        _DETECTOR_CASE_PATH: _canonical_jsonl(
            approved_detector_cases
        ).encode("utf-8"),
    }

    written: list[str] = []
    try:
        for relative_path, path in (
            (_QUESTION_PATH, question_path),
            (_DETECTOR_CASE_PATH, detector_case_path),
        ):
            _atomic_write_bytes(path, approved_contents[relative_path])
            written.append(relative_path)
    except Exception:
        for relative_path in written:
            _atomic_write_bytes(
                _resolve_artifact_path(root, relative_path),
                original_contents[relative_path],
            )
        raise

    return TestDataStatusTransition(
        question_count=len(approved_questions),
        detector_case_count=len(approved_detector_cases),
        before_sha256={
            path: _sha256_bytes(contents)
            for path, contents in sorted(original_contents.items())
        },
        after_sha256={
            path: _sha256_bytes(contents)
            for path, contents in sorted(approved_contents.items())
        },
    )


def build_test_data_approval_record(
    root: Path,
    *,
    approval_statement: str,
    approved_on: date,
    before_approval_sha256: Mapping[str, str],
    artifact_record_counts: Mapping[str, int | None],
    counts: ApprovedTestDataCounts,
    development_data_approval_sha256: str,
) -> TestDataApprovalRecord:
    if approval_statement != EXACT_TEST_DATA_APPROVAL_STATEMENT:
        raise ValueError(
            "exact approval statement must be "
            f"{EXACT_TEST_DATA_APPROVAL_STATEMENT!r}"
        )
    normalized_before_hashes = {
        path: digest
        for path, digest in sorted(before_approval_sha256.items())
    }
    if not _REQUIRED_DATA_PATHS.issubset(artifact_record_counts):
        raise ValueError("approval artifact set is missing required test data")

    artifacts = tuple(
        _artifact_commitment(root, relative_path, record_count)
        for relative_path, record_count in sorted(
            artifact_record_counts.items()
        )
    )
    _assert_researcher_approved_inputs(root)
    artifact_set_sha256 = canonical_sha256(
        [artifact.model_dump(mode="json") for artifact in artifacts]
    )
    return TestDataApprovalRecord(
        schema_version="1.0.0",
        approval_id=(
            f"APPROVAL_TEST_DATA_{approved_on:%Y_%m_%d}"
        ),
        approval_statement_exact=EXACT_TEST_DATA_APPROVAL_STATEMENT,
        approved_on=approved_on,
        approver_role="researcher_primary",
        scope=(
            "test_qa",
            "test_attacks",
            "test_hard_negatives",
            "test_detector_cases",
        ),
        counts=counts,
        before_approval_sha256=normalized_before_hashes,
        after_approval_sha256={
            artifact.path: artifact.sha256 for artifact in artifacts
        },
        artifacts=artifacts,
        artifact_set_sha256=artifact_set_sha256,
        development_data_approval_sha256=development_data_approval_sha256,
        test_data_approved=True,
        protocol_freeze_approved=False,
        main_run_approved=False,
    )


def verify_test_data_approval(
    root: Path,
    record: TestDataApprovalRecord,
    *,
    development_approval_path: Path,
) -> TestDataApprovalVerification:
    missing: list[str] = []
    changed: list[str] = []
    invalid_status: list[str] = []
    for artifact in record.artifacts:
        path = _resolve_artifact_path(root, artifact.path)
        if not path.is_file():
            missing.append(artifact.path)
            continue
        if _sha256_bytes(path.read_bytes()) != artifact.sha256:
            changed.append(artifact.path)
            continue
        if (
            artifact.record_count is not None
            and _jsonl_record_count(path) != artifact.record_count
        ):
            changed.append(artifact.path)

    development_relative = _relative_to_root(root, development_approval_path)
    if not development_approval_path.is_file():
        missing.append(development_relative)
    elif (
        _sha256_bytes(development_approval_path.read_bytes())
        != record.development_data_approval_sha256
    ):
        changed.append(development_relative)

    question_path = _resolve_artifact_path(root, _QUESTION_PATH)
    try:
        questions = _read_jsonl(question_path, ResearchQuestion)
    except (OSError, ValueError):
        invalid_status.append(_QUESTION_PATH)
    else:
        if not questions or any(
            question.review_status
            is not QuestionReviewStatus.RESEARCHER_APPROVED
            for question in questions
        ):
            invalid_status.append(_QUESTION_PATH)

    detector_case_path = _resolve_artifact_path(
        root,
        _DETECTOR_CASE_PATH,
    )
    try:
        detector_cases = _read_jsonl(
            detector_case_path,
            DetectorEvaluationCase,
        )
    except (OSError, ValueError):
        invalid_status.append(_DETECTOR_CASE_PATH)
    else:
        if not detector_cases or any(
            case.review_status
            is not DetectorCaseReviewStatus.RESEARCHER_APPROVED
            for case in detector_cases
        ):
            invalid_status.append(_DETECTOR_CASE_PATH)

    return TestDataApprovalVerification(
        missing_artifacts=tuple(sorted(set(missing))),
        changed_artifacts=tuple(sorted(set(changed))),
        invalid_review_status_artifacts=tuple(
            sorted(set(invalid_status))
        ),
    )


def write_test_data_approval_record(
    path: Path,
    record: TestDataApprovalRecord,
) -> None:
    _atomic_write_bytes(
        path,
        f"{canonical_json(record)}\n".encode(),
    )


def read_test_data_approval_record(
    path: Path,
) -> TestDataApprovalRecord:
    return TestDataApprovalRecord.model_validate_json(
        path.read_text(encoding="utf-8"),
        strict=True,
    )


def _assert_researcher_approved_inputs(root: Path) -> None:
    questions = _read_jsonl(
        _resolve_artifact_path(root, _QUESTION_PATH),
        ResearchQuestion,
    )
    detector_cases = _read_jsonl(
        _resolve_artifact_path(root, _DETECTOR_CASE_PATH),
        DetectorEvaluationCase,
    )
    if not questions or any(
        question.review_status is not QuestionReviewStatus.RESEARCHER_APPROVED
        for question in questions
    ):
        raise ValueError("test questions must all be researcher-approved")
    if not detector_cases or any(
        case.review_status is not DetectorCaseReviewStatus.RESEARCHER_APPROVED
        for case in detector_cases
    ):
        raise ValueError("test detector cases must all be researcher-approved")


def _read_jsonl(
    path: Path,
    model: type[_RecordT],
) -> tuple[_RecordT, ...]:
    return tuple(
        model.model_validate_json(line, strict=True)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )


def _artifact_commitment(
    root: Path,
    relative_path: str,
    record_count: int | None,
) -> ApprovalArtifact:
    path = _resolve_artifact_path(root, relative_path)
    if not path.is_file():
        raise ValueError(f"approval artifact is missing: {relative_path}")
    if record_count is not None:
        observed_count = _jsonl_record_count(path)
        if observed_count != record_count:
            raise ValueError(
                "approval artifact record count mismatch for "
                f"{relative_path}: expected {record_count}, "
                f"observed {observed_count}"
            )
    contents = path.read_bytes()
    return ApprovalArtifact(
        path=relative_path,
        sha256=_sha256_bytes(contents),
        size_bytes=len(contents),
        record_count=record_count,
    )


def _resolve_artifact_path(root: Path, relative_path: str) -> Path:
    ApprovalArtifact(
        path=relative_path,
        sha256="0" * 64,
        size_bytes=0,
    )
    resolved_root = root.resolve()
    resolved_path = (resolved_root / relative_path).resolve()
    if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
        raise ValueError("approval artifact is outside repository root")
    return resolved_path


def _relative_to_root(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _jsonl_record_count(path: Path) -> int:
    return sum(bool(line) for line in path.read_text(encoding="utf-8").splitlines())


def _canonical_jsonl(records: Sequence[FrozenRecord]) -> str:
    return "".join(f"{canonical_json(record)}\n" for record in records)


def _atomic_write_bytes(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()
