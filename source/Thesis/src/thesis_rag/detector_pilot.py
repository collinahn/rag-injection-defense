import os
import re
import tempfile
from collections import Counter
from collections.abc import Callable, Sequence
from enum import StrEnum
from math import isfinite
from pathlib import Path
from typing import Protocol, Self, TypeVar

from pydantic import Field, StrictBool, field_validator, model_validator

from thesis_rag.attacks import insert_payload
from thesis_rag.canonical import (
    JsonValue,
    canonical_json,
    canonical_sha256,
    sha256_text,
    stable_id,
)
from thesis_rag.detector import (
    PromptGuardModelIdentity,
    PromptGuardScore,
    PromptGuardScoringPolicy,
    resolve_prompt_guard_malicious_label_index,
)
from thesis_rag.domain import (
    AttackCase,
    Chunk,
    FiniteFloat,
    FrozenRecord,
    Language,
    MetadataValue,
    Probability,
    Sha256Hex,
    Split,
    normalize_research_text,
)
from thesis_rag.qa import ResearchQuestion

_T = TypeVar("_T")


class DetectorCaseClass(StrEnum):
    MALICIOUS = "malicious"
    HARD_NEGATIVE = "hard_negative"
    ORDINARY_BENIGN = "ordinary_benign"


class DetectorCaseReviewStatus(StrEnum):
    DRAFT = "draft"
    RESEARCHER_APPROVED = "researcher_approved"


class HardNegativeCandidate(FrozenRecord):
    case_id: str = Field(min_length=1)
    split: Split = Split.DEV
    language: Language
    text: str = Field(min_length=1)
    trigger_terms: tuple[str, ...] = Field(min_length=1)
    rationale: str = Field(min_length=1)
    template_family: str | None = None
    category: str | None = None
    format_type: str | None = None
    source_locator: str | None = None

    @field_validator("case_id", "text", "rationale")
    @classmethod
    def strip_hard_negative_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("hard-negative text fields must be nonblank")
        return stripped

    @field_validator(
        "template_family",
        "category",
        "format_type",
        "source_locator",
    )
    @classmethod
    def strip_optional_hard_negative_metadata(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("hard-negative metadata fields must be nonblank")
        return stripped

    @field_validator("trigger_terms")
    @classmethod
    def validate_trigger_terms(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        stripped = tuple(term.strip() for term in value)
        if any(not term for term in stripped):
            raise ValueError("hard-negative trigger terms must be nonblank")
        if len(set(stripped)) != len(stripped):
            raise ValueError("hard-negative trigger terms must be unique")
        return stripped

    @model_validator(mode="after")
    def require_declared_terms_in_text(self) -> Self:
        expected_prefix = "HN_DEV_" if self.split is Split.DEV else "HN_TEST_"
        if not self.case_id.startswith(expected_prefix):
            raise ValueError("hard-negative case_id prefix must match split")
        normalized_text = normalize_research_text(self.text)
        if any(normalize_research_text(term) not in normalized_text for term in self.trigger_terms):
            raise ValueError("hard-negative trigger term is absent from text")
        return self


class TestHardNegativeDatasetSummary(FrozenRecord):
    case_count: int = Field(strict=True, ge=0)
    korean_count: int = Field(strict=True, ge=0)
    english_count: int = Field(strict=True, ge=0)
    template_family_count: int = Field(strict=True, ge=0)
    category_count: int = Field(strict=True, ge=0)
    format_type_count: int = Field(strict=True, ge=0)


class HardNegativeSplitIntegritySummary(FrozenRecord):
    development_count: int = Field(strict=True, ge=0)
    test_count: int = Field(strict=True, ge=0)
    shared_normalized_text_count: int = Field(strict=True, ge=0)
    shared_template_fingerprint_count: int = Field(strict=True, ge=0)


_HARD_NEGATIVE_TEMPLATE_VARIABLE_PATTERN = re.compile(
    r"(?:https?://\S+)|(?:[\w.+-]+@[\w.-]+\.[A-Za-z]{2,})|(?:\d+(?:[.,]\d+)?)"
)


def validate_test_hard_negative_dataset(
    candidates: Sequence[HardNegativeCandidate],
) -> TestHardNegativeDatasetSummary:
    if len(candidates) != 100:
        raise ValueError("test hard-negative dataset must contain exactly 100 cases")
    if any(candidate.split is not Split.TEST for candidate in candidates):
        raise ValueError("test hard-negative dataset may only contain test cases")

    case_ids = tuple(candidate.case_id for candidate in candidates)
    if _duplicates(case_ids):
        raise ValueError("duplicate test hard-negative case IDs")
    normalized_texts = tuple(normalize_research_text(candidate.text) for candidate in candidates)
    if _duplicates(normalized_texts):
        raise ValueError("duplicate test hard-negative text")

    language_counts = Counter(candidate.language for candidate in candidates)
    if language_counts != Counter({Language.KO: 50, Language.EN: 50}):
        raise ValueError("test hard-negative dataset requires 50 Korean and 50 English cases")

    metadata_fields = (
        "template_family",
        "category",
        "format_type",
        "source_locator",
    )
    for candidate in candidates:
        missing = tuple(
            field_name for field_name in metadata_fields if getattr(candidate, field_name) is None
        )
        if missing:
            raise ValueError(
                f"{candidate.case_id} missing test hard-negative metadata: {','.join(missing)}"
            )

    family_counts = Counter(candidate.template_family for candidate in candidates)
    if len(family_counts) != 24:
        raise ValueError("test hard-negative dataset requires exactly 24 template families")
    if Counter(family_counts.values()) != Counter({6: 2, 4: 22}):
        raise ValueError(
            "test hard-negative template families must contain two six-case "
            "families and twenty-two four-case families"
        )
    for family in family_counts:
        family_language_counts = Counter(
            candidate.language for candidate in candidates if candidate.template_family == family
        )
        if family_language_counts[Language.KO] != family_language_counts[Language.EN]:
            raise ValueError(
                f"test hard-negative template families must be language-balanced: {family}"
            )

    categories = {candidate.category for candidate in candidates}
    formats = {candidate.format_type for candidate in candidates}
    if len(categories) < 6:
        raise ValueError("test hard-negative dataset requires at least six categories")
    if len(formats) < 5:
        raise ValueError("test hard-negative dataset requires at least five format types")
    source_locators = tuple(candidate.source_locator for candidate in candidates)
    if len(set(source_locators)) != len(source_locators):
        raise ValueError("test hard-negative source locators must be unique")

    return TestHardNegativeDatasetSummary(
        case_count=len(candidates),
        korean_count=language_counts[Language.KO],
        english_count=language_counts[Language.EN],
        template_family_count=len(family_counts),
        category_count=len(categories),
        format_type_count=len(formats),
    )


def validate_hard_negative_split_integrity(
    development_candidates: Sequence[HardNegativeCandidate],
    test_candidates: Sequence[HardNegativeCandidate],
) -> HardNegativeSplitIntegritySummary:
    if any(candidate.split is not Split.DEV for candidate in development_candidates):
        raise ValueError("development hard negatives must use dev split")
    if any(candidate.split is not Split.TEST for candidate in test_candidates):
        raise ValueError("test hard negatives must use test split")

    development_ids = {candidate.case_id for candidate in development_candidates}
    test_ids = {candidate.case_id for candidate in test_candidates}
    shared_ids = sorted(development_ids & test_ids)
    if shared_ids:
        raise ValueError("hard-negative case IDs shared by dev and test: " + ", ".join(shared_ids))

    development_texts = {
        normalize_research_text(candidate.text) for candidate in development_candidates
    }
    test_texts = {normalize_research_text(candidate.text) for candidate in test_candidates}
    shared_texts = development_texts & test_texts
    if shared_texts:
        raise ValueError("hard-negative normalized texts shared by dev and test")

    development_fingerprints = {
        _hard_negative_template_fingerprint(candidate.text) for candidate in development_candidates
    }
    test_fingerprints = {
        _hard_negative_template_fingerprint(candidate.text) for candidate in test_candidates
    }
    shared_fingerprints = development_fingerprints & test_fingerprints
    if shared_fingerprints:
        raise ValueError("hard-negative template fingerprints shared by dev and test")

    return HardNegativeSplitIntegritySummary(
        development_count=len(development_candidates),
        test_count=len(test_candidates),
        shared_normalized_text_count=0,
        shared_template_fingerprint_count=0,
    )


def _hard_negative_template_fingerprint(text: str) -> str:
    templated = _HARD_NEGATIVE_TEMPLATE_VARIABLE_PATTERN.sub(
        "<value>",
        text,
    )
    return normalize_research_text(templated)


class DetectorEvaluationCase(FrozenRecord):
    case_id: str = Field(min_length=1)
    split: Split
    case_class: DetectorCaseClass
    chunk: Chunk
    source_attack_id: str | None = None
    attack_language: Language | None = None
    review_status: DetectorCaseReviewStatus

    @field_validator("case_id")
    @classmethod
    def strip_case_id(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("detector case_id must be nonblank")
        return stripped

    @field_validator("source_attack_id")
    @classmethod
    def strip_source_attack_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("detector source_attack_id must be nonblank")
        return stripped

    @model_validator(mode="after")
    def validate_attack_attribution(self) -> Self:
        metadata_is_malicious = self.chunk.metadata.get("is_malicious") is True
        metadata_attack_id = self.chunk.metadata.get("attack_id")
        if self.case_class is DetectorCaseClass.MALICIOUS:
            if self.source_attack_id is None:
                raise ValueError("malicious detector case requires source_attack_id")
            if not metadata_is_malicious:
                raise ValueError("malicious detector case requires malicious chunk metadata")
            if metadata_attack_id != self.source_attack_id:
                raise ValueError("malicious detector case attack attribution mismatch")
        elif self.source_attack_id is not None or metadata_is_malicious:
            raise ValueError("benign detector case must not carry attack attribution")
        if self.case_class is not DetectorCaseClass.MALICIOUS and self.attack_language is not None:
            raise ValueError("benign detector case must not carry attack language")
        return self


class DetectorPilotRecord(FrozenRecord):
    case_id: str
    split: Split
    case_class: DetectorCaseClass
    language: Language
    chunk_id: str
    chunk_text_sha256: Sha256Hex
    source_attack_id: str | None
    attack_language: Language | None = None
    review_status: DetectorCaseReviewStatus
    case_commitment: Sha256Hex
    threshold: Probability
    predicted_malicious: StrictBool
    score_detail: PromptGuardScore


class DetectorPilotSummary(FrozenRecord):
    schema_version: str
    split: Split
    model_identity: PromptGuardModelIdentity
    scoring_policy: PromptGuardScoringPolicy
    scoring_policy_sha256: Sha256Hex
    threshold: Probability
    record_count: int = Field(strict=True, ge=1)
    malicious_count: int = Field(strict=True, ge=0)
    malicious_true_positive_count: int = Field(strict=True, ge=0)
    malicious_recall: FiniteFloat
    hard_negative_count: int = Field(strict=True, ge=0)
    hard_negative_false_positive_count: int = Field(strict=True, ge=0)
    hard_negative_false_positive_rate: FiniteFloat
    ordinary_benign_count: int = Field(strict=True, ge=0)
    ordinary_benign_false_positive_count: int = Field(strict=True, ge=0)
    ordinary_benign_false_positive_rate: FiniteFloat
    overall_benign_false_positive_rate: FiniteFloat
    precision: FiniteFloat
    f1: FiniteFloat
    case_set_sha256: Sha256Hex
    score_set_sha256: Sha256Hex
    results_file_sha256: Sha256Hex

    @model_validator(mode="after")
    def validate_scoring_policy_hash(self) -> Self:
        if canonical_sha256(self.scoring_policy) != self.scoring_policy_sha256:
            raise ValueError("detector scoring policy hash mismatch")
        return self


class DetectorThresholdSelection(FrozenRecord):
    schema_version: str
    maximum_false_positive_rate: Probability
    candidate_count: int = Field(strict=True, ge=1)
    constraint_satisfied: StrictBool
    threshold: Probability
    malicious_recall: FiniteFloat
    hard_negative_false_positive_rate: FiniteFloat
    ordinary_benign_false_positive_rate: FiniteFloat
    precision: FiniteFloat
    record_set_sha256: Sha256Hex


class CalibratedDetectorThreshold(FrozenRecord):
    schema_version: str
    selection: DetectorThresholdSelection
    model_identity: PromptGuardModelIdentity
    detector_case_set_sha256: Sha256Hex
    detector_score_set_sha256: Sha256Hex
    detector_scoring_policy_sha256: Sha256Hex
    detector_results_file_sha256: Sha256Hex
    detector_summary_file_sha256: Sha256Hex


class DetailedDetectorAdapter(Protocol):
    model_id: str
    model_revision: str

    @property
    def scoring_policy(self) -> PromptGuardScoringPolicy: ...

    def verify_model(self) -> PromptGuardModelIdentity: ...

    def score_with_metadata(self, chunk: Chunk) -> PromptGuardScore: ...


def build_detector_evaluation_cases(
    attacks: Sequence[AttackCase],
    questions: Sequence[ResearchQuestion],
    chunks: Sequence[Chunk],
    hard_negatives: Sequence[HardNegativeCandidate],
    *,
    ordinary_count: int,
    split: Split = Split.DEV,
) -> tuple[DetectorEvaluationCase, ...]:
    if ordinary_count < 1:
        raise ValueError("ordinary detector case count must be positive")
    questions_by_id = _unique_by_id(
        questions,
        identity=lambda question: question.question_id,
        label="question",
    )
    chunks_by_id = _unique_by_id(
        chunks,
        identity=lambda chunk: chunk.chunk_id,
        label="chunk",
    )

    cases: list[DetectorEvaluationCase] = []
    for attack in attacks:
        if attack.split is not split:
            raise ValueError(f"{split.value} detector attacks must use {split.value} split")
        try:
            question = questions_by_id[attack.base_context_id]
        except KeyError as error:
            raise ValueError(
                f"unknown detector attack base context: {attack.base_context_id}"
            ) from error
        if not question.answerable or not question.gold_chunk_ids:
            raise ValueError("detector attack base context must be answerable")
        if question.split is not split:
            raise ValueError(f"{split.value} detector base contexts must use {split.value} split")
        try:
            clean_chunk = chunks_by_id[question.gold_chunk_ids[0]]
        except KeyError as error:
            raise ValueError(
                f"missing detector attack gold chunk: {question.gold_chunk_ids[0]}"
            ) from error
        malicious_chunk = insert_payload(clean_chunk, attack)
        cases.append(
            DetectorEvaluationCase(
                case_id=f"DET_{attack.attack_id}",
                split=split,
                case_class=DetectorCaseClass.MALICIOUS,
                chunk=malicious_chunk,
                source_attack_id=attack.attack_id,
                attack_language=(attack.language if split is Split.TEST else None),
                review_status=DetectorCaseReviewStatus.DRAFT,
            )
        )

    if any(candidate.split is not split for candidate in hard_negatives):
        raise ValueError(f"{split.value} hard negatives must use {split.value} split")
    hard_negative_ids = tuple(candidate.case_id for candidate in hard_negatives)
    if _duplicates(hard_negative_ids):
        raise ValueError("duplicate hard-negative case IDs")
    normalized_hard_negative_texts = tuple(
        normalize_research_text(candidate.text) for candidate in hard_negatives
    )
    if _duplicates(normalized_hard_negative_texts):
        raise ValueError("duplicate hard-negative text")
    for candidate in hard_negatives:
        stable_payload: dict[str, JsonValue] = {
            "case_id": candidate.case_id,
            "language": candidate.language.value,
            "text": candidate.text,
        }
        if split is Split.TEST:
            stable_payload["split"] = split.value
        metadata: dict[str, MetadataValue] = {
            "detector_case_class": DetectorCaseClass.HARD_NEGATIVE.value,
            "trigger_terms": "|".join(candidate.trigger_terms),
            "rationale": candidate.rationale,
        }
        for key, value in {
            "template_family": candidate.template_family,
            "category": candidate.category,
            "format_type": candidate.format_type,
            "source_locator": candidate.source_locator,
        }.items():
            if value is not None:
                metadata[key] = value
        chunk = Chunk(
            chunk_id=stable_id(
                "CH_HN",
                stable_payload,
            ),
            document_id=f"DOC_HARD_NEGATIVE_{split.value.upper()}",
            text=candidate.text,
            language=candidate.language,
            metadata=metadata,
        )
        cases.append(
            DetectorEvaluationCase(
                case_id=candidate.case_id,
                split=split,
                case_class=DetectorCaseClass.HARD_NEGATIVE,
                chunk=chunk,
                source_attack_id=None,
                review_status=DetectorCaseReviewStatus.DRAFT,
            )
        )

    ordinary_chunks = _round_robin_by_document(chunks, ordinary_count)
    for index, chunk in enumerate(ordinary_chunks, start=1):
        cases.append(
            DetectorEvaluationCase(
                case_id=f"OB_{split.value.upper()}_{index:03d}",
                split=split,
                case_class=DetectorCaseClass.ORDINARY_BENIGN,
                chunk=chunk,
                source_attack_id=None,
                review_status=DetectorCaseReviewStatus.DRAFT,
            )
        )
    return tuple(cases)


def run_detector_pilot(
    cases: Sequence[DetectorEvaluationCase],
    detector: DetailedDetectorAdapter,
    *,
    threshold: float,
    results_path: Path,
    summary_path: Path,
) -> DetectorPilotSummary:
    if not isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("detector threshold must be a finite value in [0, 1]")
    if results_path.resolve(strict=False) == summary_path.resolve(strict=False):
        raise ValueError("detector pilot paths must be distinct")
    if not cases:
        raise ValueError("detector pilot requires at least one case")

    case_ids = tuple(case.case_id for case in cases)
    duplicate_case_ids = _duplicates(case_ids)
    if duplicate_case_ids:
        raise ValueError("duplicate detector case IDs: " + ", ".join(duplicate_case_ids))
    chunk_ids = tuple(case.chunk.chunk_id for case in cases)
    duplicate_chunk_ids = _duplicates(chunk_ids)
    if duplicate_chunk_ids:
        raise ValueError("duplicate detector chunk IDs: " + ", ".join(duplicate_chunk_ids))
    splits = {case.split for case in cases}
    if len(splits) != 1:
        raise ValueError("detector pilot cases must use one split")

    identity = detector.verify_model()
    if identity.model_id != detector.model_id or identity.model_revision != detector.model_revision:
        raise ValueError("verified detector identity does not match adapter")
    scoring_policy = detector.scoring_policy
    if scoring_policy.resolved_malicious_label_index != (
        resolve_prompt_guard_malicious_label_index(
            identity.labels,
            scoring_policy.configured_malicious_label,
        )
    ):
        raise ValueError("detector scoring policy label mapping mismatch")

    records: list[DetectorPilotRecord] = []
    for case in cases:
        score_detail = detector.score_with_metadata(case.chunk)
        if score_detail.chunk_id != case.chunk.chunk_id:
            raise ValueError(f"detector score chunk ID mismatch for {case.case_id}")
        records.append(
            DetectorPilotRecord(
                case_id=case.case_id,
                split=case.split,
                case_class=case.case_class,
                language=case.chunk.language,
                chunk_id=case.chunk.chunk_id,
                chunk_text_sha256=sha256_text(case.chunk.text),
                source_attack_id=case.source_attack_id,
                attack_language=case.attack_language,
                review_status=case.review_status,
                case_commitment=canonical_sha256(case),
                threshold=threshold,
                predicted_malicious=score_detail.score >= threshold,
                score_detail=score_detail,
            )
        )

    result_contents = "".join(f"{canonical_json(record)}\n" for record in records)
    _atomic_write_text(results_path, result_contents)

    malicious_records = tuple(
        record for record in records if record.case_class is DetectorCaseClass.MALICIOUS
    )
    hard_negative_records = tuple(
        record for record in records if record.case_class is DetectorCaseClass.HARD_NEGATIVE
    )
    ordinary_records = tuple(
        record for record in records if record.case_class is DetectorCaseClass.ORDINARY_BENIGN
    )
    benign_records = hard_negative_records + ordinary_records
    true_positive_count = sum(record.predicted_malicious for record in malicious_records)
    hard_negative_false_positive_count = sum(
        record.predicted_malicious for record in hard_negative_records
    )
    ordinary_false_positive_count = sum(record.predicted_malicious for record in ordinary_records)
    false_positive_count = hard_negative_false_positive_count + ordinary_false_positive_count
    recall = _rate(true_positive_count, len(malicious_records))
    precision = _rate(
        true_positive_count,
        true_positive_count + false_positive_count,
    )
    summary = DetectorPilotSummary(
        schema_version="1.0.0",
        split=next(iter(splits)),
        model_identity=identity,
        scoring_policy=scoring_policy,
        scoring_policy_sha256=canonical_sha256(scoring_policy),
        threshold=threshold,
        record_count=len(records),
        malicious_count=len(malicious_records),
        malicious_true_positive_count=true_positive_count,
        malicious_recall=recall,
        hard_negative_count=len(hard_negative_records),
        hard_negative_false_positive_count=hard_negative_false_positive_count,
        hard_negative_false_positive_rate=_rate(
            hard_negative_false_positive_count,
            len(hard_negative_records),
        ),
        ordinary_benign_count=len(ordinary_records),
        ordinary_benign_false_positive_count=ordinary_false_positive_count,
        ordinary_benign_false_positive_rate=_rate(
            ordinary_false_positive_count,
            len(ordinary_records),
        ),
        overall_benign_false_positive_rate=_rate(
            false_positive_count,
            len(benign_records),
        ),
        precision=precision,
        f1=(0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)),
        case_set_sha256=canonical_sha256([case.model_dump(mode="json") for case in cases]),
        score_set_sha256=canonical_sha256([record.model_dump(mode="json") for record in records]),
        results_file_sha256=sha256_text(result_contents),
    )
    _atomic_write_text(summary_path, f"{canonical_json(summary)}\n")
    return summary


def select_detector_threshold(
    records: Sequence[DetectorPilotRecord],
    *,
    maximum_false_positive_rate: float,
) -> DetectorThresholdSelection:
    if not isfinite(maximum_false_positive_rate) or not 0.0 <= maximum_false_positive_rate <= 1.0:
        raise ValueError("maximum false-positive rate must be in [0, 1]")
    if not records:
        raise ValueError("threshold selection requires detector records")
    if any(record.split is not Split.DEV for record in records):
        raise ValueError("threshold selection may only use development records")
    record_ids = tuple(record.case_id for record in records)
    if _duplicates(record_ids):
        raise ValueError("threshold selection record IDs must be unique")

    malicious = tuple(
        record for record in records if record.case_class is DetectorCaseClass.MALICIOUS
    )
    hard_negatives = tuple(
        record for record in records if record.case_class is DetectorCaseClass.HARD_NEGATIVE
    )
    ordinary = tuple(
        record for record in records if record.case_class is DetectorCaseClass.ORDINARY_BENIGN
    )
    if not malicious or not hard_negatives or not ordinary:
        raise ValueError(
            "threshold selection requires malicious, hard-negative, and ordinary cases"
        )

    candidates = tuple(
        sorted(
            {1.0, *(record.score_detail.score for record in records)},
        )
    )
    evaluations = tuple(
        _evaluate_threshold(
            threshold,
            malicious,
            hard_negatives,
            ordinary,
        )
        for threshold in candidates
    )
    feasible = tuple(
        evaluation
        for evaluation in evaluations
        if (
            evaluation[2] <= maximum_false_positive_rate
            and evaluation[3] <= maximum_false_positive_rate
        )
    )
    if feasible:
        selected = max(
            feasible,
            key=lambda evaluation: (
                evaluation[1],
                evaluation[4],
                evaluation[0],
            ),
        )
        constraint_satisfied = True
    else:
        selected = min(
            evaluations,
            key=lambda evaluation: (
                evaluation[2],
                evaluation[3],
                -evaluation[0],
            ),
        )
        constraint_satisfied = False

    return DetectorThresholdSelection(
        schema_version="1.0.0",
        maximum_false_positive_rate=maximum_false_positive_rate,
        candidate_count=len(candidates),
        constraint_satisfied=constraint_satisfied,
        threshold=selected[0],
        malicious_recall=selected[1],
        hard_negative_false_positive_rate=selected[2],
        ordinary_benign_false_positive_rate=selected[3],
        precision=selected[4],
        record_set_sha256=canonical_sha256([record.model_dump(mode="json") for record in records]),
    )


def write_detector_threshold_selection(
    path: Path,
    selection: DetectorThresholdSelection | CalibratedDetectorThreshold,
) -> None:
    _atomic_write_text(path, f"{canonical_json(selection)}\n")


def bind_detector_threshold(
    selection: DetectorThresholdSelection,
    summary: DetectorPilotSummary,
    *,
    results_file_sha256: str,
    summary_file_sha256: str,
) -> CalibratedDetectorThreshold:
    if summary.split is not Split.DEV:
        raise ValueError("detector threshold may only bind development results")
    if selection.record_set_sha256 != summary.score_set_sha256:
        raise ValueError("detector threshold record set does not match pilot scores")
    if results_file_sha256 != summary.results_file_sha256:
        raise ValueError("detector pilot results file hash mismatch")
    return CalibratedDetectorThreshold(
        schema_version="1.0.0",
        selection=selection,
        model_identity=summary.model_identity,
        detector_case_set_sha256=summary.case_set_sha256,
        detector_score_set_sha256=summary.score_set_sha256,
        detector_scoring_policy_sha256=summary.scoring_policy_sha256,
        detector_results_file_sha256=results_file_sha256,
        detector_summary_file_sha256=summary_file_sha256,
    )


def verify_calibrated_detector_threshold(
    calibrated: CalibratedDetectorThreshold,
    summary: DetectorPilotSummary,
    *,
    results_file_sha256: str,
    summary_file_sha256: str,
    expected_model_id: str,
    expected_model_revision: str,
    expected_scoring_policy_sha256: str,
) -> float:
    if not calibrated.selection.constraint_satisfied:
        raise ValueError(
            "calibrated detector threshold does not satisfy the false-positive constraint"
        )
    if (
        calibrated.model_identity.model_id != expected_model_id
        or calibrated.model_identity.model_revision != expected_model_revision
    ):
        raise ValueError("calibrated detector model identity mismatch")
    if calibrated.model_identity != summary.model_identity:
        raise ValueError("calibrated detector summary model identity mismatch")
    if calibrated.detector_case_set_sha256 != summary.case_set_sha256:
        raise ValueError("calibrated detector case set hash mismatch")
    if calibrated.detector_score_set_sha256 != summary.score_set_sha256:
        raise ValueError("calibrated detector score set hash mismatch")
    if (
        calibrated.detector_scoring_policy_sha256 != summary.scoring_policy_sha256
        or calibrated.detector_scoring_policy_sha256 != expected_scoring_policy_sha256
    ):
        raise ValueError("calibrated detector scoring policy mismatch")
    if calibrated.selection.record_set_sha256 != calibrated.detector_score_set_sha256:
        raise ValueError("calibrated detector threshold record set hash mismatch")
    if calibrated.detector_results_file_sha256 != results_file_sha256:
        raise ValueError("detector results file hash mismatch")
    if summary.results_file_sha256 != results_file_sha256:
        raise ValueError("detector summary results file hash mismatch")
    if calibrated.detector_summary_file_sha256 != summary_file_sha256:
        raise ValueError("detector summary file hash mismatch")
    return calibrated.selection.threshold


def _evaluate_threshold(
    threshold: float,
    malicious: Sequence[DetectorPilotRecord],
    hard_negatives: Sequence[DetectorPilotRecord],
    ordinary: Sequence[DetectorPilotRecord],
) -> tuple[float, float, float, float, float]:
    true_positive_count = sum(record.score_detail.score >= threshold for record in malicious)
    hard_negative_false_positive_count = sum(
        record.score_detail.score >= threshold for record in hard_negatives
    )
    ordinary_false_positive_count = sum(
        record.score_detail.score >= threshold for record in ordinary
    )
    false_positive_count = hard_negative_false_positive_count + ordinary_false_positive_count
    recall = _rate(true_positive_count, len(malicious))
    return (
        threshold,
        recall,
        _rate(hard_negative_false_positive_count, len(hard_negatives)),
        _rate(ordinary_false_positive_count, len(ordinary)),
        _rate(
            true_positive_count,
            true_positive_count + false_positive_count,
        ),
    )


def _atomic_write_text(path: Path, contents: str) -> None:
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


def _duplicates(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(value for value, count in Counter(values).items() if count > 1))


def _unique_by_id(
    values: Sequence[_T],
    *,
    identity: Callable[[_T], str],
    label: str,
) -> dict[str, _T]:
    result: dict[str, _T] = {}
    for value in values:
        item_id = identity(value)
        if item_id in result:
            raise ValueError(f"duplicate detector {label} IDs: {item_id}")
        result[item_id] = value
    return result


def _round_robin_by_document(
    chunks: Sequence[Chunk],
    count: int,
) -> tuple[Chunk, ...]:
    if count > len(chunks):
        raise ValueError("ordinary detector case count exceeds corpus size")
    chunks_by_document: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        chunks_by_document.setdefault(chunk.document_id, []).append(chunk)

    selected: list[Chunk] = []
    offset = 0
    document_ids = sorted(chunks_by_document)
    while len(selected) < count:
        added = False
        for document_id in document_ids:
            document_chunks = chunks_by_document[document_id]
            if offset < len(document_chunks):
                selected.append(document_chunks[offset])
                added = True
                if len(selected) == count:
                    break
        if not added:
            raise ValueError("unable to select ordinary detector cases")
        offset += 1
    return tuple(selected)


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator
