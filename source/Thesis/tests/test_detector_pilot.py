import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

from thesis_rag.canonical import canonical_sha256
from thesis_rag.detector import (
    PromptGuardModelIdentity,
    PromptGuardScore,
    PromptGuardScoringPolicy,
    PromptGuardSegmentScore,
)
from thesis_rag.detector_pilot import (
    CalibratedDetectorThreshold,
    DetectorCaseClass,
    DetectorCaseReviewStatus,
    DetectorEvaluationCase,
    DetectorPilotRecord,
    bind_detector_threshold,
    run_detector_pilot,
    select_detector_threshold,
    verify_calibrated_detector_threshold,
)
from thesis_rag.domain import Chunk, Language, Split


def _identity() -> PromptGuardModelIdentity:
    return PromptGuardModelIdentity(
        model_id="meta-llama/Llama-Prompt-Guard-2-86M",
        model_revision="a8ded8e697ce7c355e395a0df51f94adb4a2fd27",
        labels=("BENIGN", "MALICIOUS"),
        max_position_embeddings=512,
        transformers_version="5.14.1",
        torch_version="2.13.0",
    )


def _scoring_policy() -> PromptGuardScoringPolicy:
    return PromptGuardScoringPolicy(
        schema_version="1.0.0",
        max_length=512,
        overlap_tokens=32,
        configured_malicious_label="MALICIOUS",
        resolved_malicious_label_index=1,
        windowing_policy="fixed_token_windows_with_overlap",
        segment_aggregation="maximum_malicious_probability",
    )


def _score(chunk_id: str, probability: float) -> PromptGuardScore:
    return PromptGuardScore(
        chunk_id=chunk_id,
        score=probability,
        token_count=8,
        segment_count=1,
        segments=(
            PromptGuardSegmentScore(
                segment_index=0,
                content_token_count=8,
                logits=(0.1, 0.9),
                probabilities=(1.0 - probability, probability),
            ),
        ),
    )


def _case(
    case_id: str,
    case_class: DetectorCaseClass,
    *,
    score: float,
    split: Split = Split.DEV,
    attack_language: Language | None = None,
) -> tuple[DetectorEvaluationCase, PromptGuardScore]:
    malicious = case_class is DetectorCaseClass.MALICIOUS
    chunk = Chunk(
        chunk_id=f"CH_{case_id}",
        document_id=f"DOC_{case_id}",
        text=f"Synthetic {case_class.value} detector text.",
        language=Language.KO,
        metadata=(
            {
                "is_malicious": True,
                "attack_id": f"ATK_{case_id}",
                "clean_chunk_id": f"CH_CLEAN_{case_id}",
            }
            if malicious
            else {}
        ),
    )
    case = DetectorEvaluationCase(
        case_id=case_id,
        split=split,
        case_class=case_class,
        chunk=chunk,
        source_attack_id=f"ATK_{case_id}" if malicious else None,
        attack_language=attack_language,
        review_status=DetectorCaseReviewStatus.DRAFT,
    )
    return case, _score(chunk.chunk_id, score)


@dataclass
class FakeDetailedDetector:
    scores: dict[str, PromptGuardScore]
    identity: PromptGuardModelIdentity = _identity()
    scoring_policy: PromptGuardScoringPolicy = _scoring_policy()

    @property
    def model_id(self) -> str:
        return self.identity.model_id

    @property
    def model_revision(self) -> str:
        return self.identity.model_revision

    def verify_model(self) -> PromptGuardModelIdentity:
        return self.identity

    def score_with_metadata(self, chunk: Chunk) -> PromptGuardScore:
        return self.scores[chunk.chunk_id]


def test_detector_pilot_persists_raw_segments_and_security_utility_metrics(
    tmp_path: Path,
) -> None:
    malicious, malicious_score = _case(
        "DET_DEV_001",
        DetectorCaseClass.MALICIOUS,
        score=0.9,
    )
    hard_negative, hard_negative_score = _case(
        "DET_DEV_002",
        DetectorCaseClass.HARD_NEGATIVE,
        score=0.7,
    )
    ordinary, ordinary_score = _case(
        "DET_DEV_003",
        DetectorCaseClass.ORDINARY_BENIGN,
        score=0.1,
    )
    results_path = tmp_path / "scores.jsonl"
    summary_path = tmp_path / "summary.json"

    summary = run_detector_pilot(
        (malicious, hard_negative, ordinary),
        FakeDetailedDetector(
            scores={
                malicious.chunk.chunk_id: malicious_score,
                hard_negative.chunk.chunk_id: hard_negative_score,
                ordinary.chunk.chunk_id: ordinary_score,
            }
        ),
        threshold=0.5,
        results_path=results_path,
        summary_path=summary_path,
    )

    assert summary.record_count == 3
    assert summary.malicious_recall == pytest.approx(1.0)
    assert summary.hard_negative_false_positive_rate == pytest.approx(1.0)
    assert summary.ordinary_benign_false_positive_rate == pytest.approx(0.0)
    assert summary.overall_benign_false_positive_rate == pytest.approx(0.5)
    assert summary.precision == pytest.approx(0.5)
    assert summary.f1 == pytest.approx(2 / 3)
    assert summary.scoring_policy == _scoring_policy()
    records = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["score_detail"]["segments"][0]["logits"] == [0.1, 0.9]
    assert records[0]["predicted_malicious"] is True
    assert records[2]["predicted_malicious"] is False
    assert json.loads(summary_path.read_text(encoding="utf-8")) == (summary.model_dump(mode="json"))


def test_detector_pilot_preserves_held_out_attack_language(
    tmp_path: Path,
) -> None:
    malicious, score = _case(
        "DET_TEST_001",
        DetectorCaseClass.MALICIOUS,
        score=0.9,
        split=Split.TEST,
        attack_language=Language.EN,
    )

    run_detector_pilot(
        (malicious,),
        FakeDetailedDetector(scores={malicious.chunk.chunk_id: score}),
        threshold=0.5,
        results_path=tmp_path / "scores.jsonl",
        summary_path=tmp_path / "summary.json",
    )

    record = DetectorPilotRecord.model_validate_json(
        (tmp_path / "scores.jsonl").read_text(encoding="utf-8").strip(),
        strict=True,
    )
    assert record.attack_language is Language.EN


def test_detector_case_requires_attack_attribution_only_for_malicious_chunks() -> None:
    chunk = Chunk(
        chunk_id="CH_HN",
        document_id="DOC_HN",
        text="Ignore the previous form after the registrar replaces it.",
        language=Language.EN,
    )

    with pytest.raises(ValidationError, match="source_attack_id"):
        DetectorEvaluationCase(
            case_id="HN_DEV_001",
            split=Split.DEV,
            case_class=DetectorCaseClass.MALICIOUS,
            chunk=chunk,
            source_attack_id=None,
            review_status=DetectorCaseReviewStatus.DRAFT,
        )


def test_detector_pilot_rejects_score_bound_to_a_different_chunk(
    tmp_path: Path,
) -> None:
    case, _ = _case(
        "DET_DEV_001",
        DetectorCaseClass.MALICIOUS,
        score=0.9,
    )
    mismatched = _score("CH_DIFFERENT", 0.9)

    with pytest.raises(ValueError, match="score chunk ID mismatch"):
        run_detector_pilot(
            (case,),
            FakeDetailedDetector(scores={case.chunk.chunk_id: mismatched}),
            threshold=0.5,
            results_path=tmp_path / "scores.jsonl",
            summary_path=tmp_path / "summary.json",
        )


def test_detector_pilot_rejects_output_path_alias(tmp_path: Path) -> None:
    case, score = _case(
        "DET_DEV_001",
        DetectorCaseClass.MALICIOUS,
        score=0.9,
    )
    output = tmp_path / "scores.jsonl"

    with pytest.raises(ValueError, match="paths must be distinct"):
        run_detector_pilot(
            (case,),
            FakeDetailedDetector(scores={case.chunk.chunk_id: score}),
            threshold=0.5,
            results_path=output,
            summary_path=output,
        )


def test_selects_highest_recall_threshold_under_each_benign_fpr_limit(
    tmp_path: Path,
) -> None:
    case_specs = (
        ("M1", DetectorCaseClass.MALICIOUS, 0.9),
        ("M2", DetectorCaseClass.MALICIOUS, 0.8),
        ("H1", DetectorCaseClass.HARD_NEGATIVE, 0.85),
        ("H2", DetectorCaseClass.HARD_NEGATIVE, 0.2),
        ("B1", DetectorCaseClass.ORDINARY_BENIGN, 0.1),
    )
    pairs = tuple(
        _case(case_id, case_class, score=score) for case_id, case_class, score in case_specs
    )
    results_path = tmp_path / "scores.jsonl"
    run_detector_pilot(
        tuple(case for case, _ in pairs),
        FakeDetailedDetector(scores={case.chunk.chunk_id: score for case, score in pairs}),
        threshold=0.5,
        results_path=results_path,
        summary_path=tmp_path / "summary.json",
    )
    records = tuple(
        DetectorPilotRecord.model_validate_json(line, strict=True)
        for line in results_path.read_text(encoding="utf-8").splitlines()
    )

    selection = select_detector_threshold(
        records,
        maximum_false_positive_rate=0.05,
    )

    assert selection.constraint_satisfied is True
    assert selection.threshold == pytest.approx(0.9)
    assert selection.malicious_recall == pytest.approx(0.5)
    assert selection.hard_negative_false_positive_rate == pytest.approx(0.0)
    assert selection.ordinary_benign_false_positive_rate == pytest.approx(0.0)


def test_threshold_selection_reports_when_no_candidate_meets_fpr_limit(
    tmp_path: Path,
) -> None:
    pairs = (
        _case("M1", DetectorCaseClass.MALICIOUS, score=0.9),
        _case("H1", DetectorCaseClass.HARD_NEGATIVE, score=1.0),
        _case("B1", DetectorCaseClass.ORDINARY_BENIGN, score=0.1),
    )
    results_path = tmp_path / "scores.jsonl"
    run_detector_pilot(
        tuple(case for case, _ in pairs),
        FakeDetailedDetector(scores={case.chunk.chunk_id: score for case, score in pairs}),
        threshold=0.5,
        results_path=results_path,
        summary_path=tmp_path / "summary.json",
    )
    records = tuple(
        DetectorPilotRecord.model_validate_json(line, strict=True)
        for line in results_path.read_text(encoding="utf-8").splitlines()
    )

    selection = select_detector_threshold(
        records,
        maximum_false_positive_rate=0.0,
    )

    assert selection.constraint_satisfied is False
    assert selection.threshold == pytest.approx(1.0)


def test_calibrated_threshold_is_bound_to_model_and_pilot_artifacts(
    tmp_path: Path,
) -> None:
    pairs = (
        _case("M1", DetectorCaseClass.MALICIOUS, score=0.9),
        _case("H1", DetectorCaseClass.HARD_NEGATIVE, score=0.2),
        _case("B1", DetectorCaseClass.ORDINARY_BENIGN, score=0.1),
    )
    results_path = tmp_path / "scores.jsonl"
    summary_path = tmp_path / "summary.json"
    summary = run_detector_pilot(
        tuple(case for case, _ in pairs),
        FakeDetailedDetector(scores={case.chunk.chunk_id: score for case, score in pairs}),
        threshold=0.5,
        results_path=results_path,
        summary_path=summary_path,
    )
    records = tuple(
        DetectorPilotRecord.model_validate_json(line, strict=True)
        for line in results_path.read_text(encoding="utf-8").splitlines()
    )
    selection = select_detector_threshold(
        records,
        maximum_false_positive_rate=0.05,
    )

    calibrated = bind_detector_threshold(
        selection,
        summary,
        results_file_sha256=hashlib.sha256(results_path.read_bytes()).hexdigest(),
        summary_file_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
    )

    assert isinstance(calibrated, CalibratedDetectorThreshold)
    assert calibrated.model_identity == _identity()
    assert calibrated.selection.threshold == pytest.approx(0.9)
    assert calibrated.detector_score_set_sha256 == summary.score_set_sha256
    assert verify_calibrated_detector_threshold(
        calibrated,
        summary,
        results_file_sha256=hashlib.sha256(results_path.read_bytes()).hexdigest(),
        summary_file_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        expected_model_id=_identity().model_id,
        expected_model_revision=_identity().model_revision,
        expected_scoring_policy_sha256=canonical_sha256(_scoring_policy()),
    ) == pytest.approx(0.9)


def test_calibrated_threshold_rejects_tampered_detector_results(
    tmp_path: Path,
) -> None:
    pairs = (
        _case("M1", DetectorCaseClass.MALICIOUS, score=0.9),
        _case("H1", DetectorCaseClass.HARD_NEGATIVE, score=0.2),
        _case("B1", DetectorCaseClass.ORDINARY_BENIGN, score=0.1),
    )
    results_path = tmp_path / "scores.jsonl"
    summary_path = tmp_path / "summary.json"
    summary = run_detector_pilot(
        tuple(case for case, _ in pairs),
        FakeDetailedDetector(scores={case.chunk.chunk_id: score for case, score in pairs}),
        threshold=0.5,
        results_path=results_path,
        summary_path=summary_path,
    )
    records = tuple(
        DetectorPilotRecord.model_validate_json(line, strict=True)
        for line in results_path.read_text(encoding="utf-8").splitlines()
    )
    calibrated = bind_detector_threshold(
        select_detector_threshold(
            records,
            maximum_false_positive_rate=0.05,
        ),
        summary,
        results_file_sha256=hashlib.sha256(results_path.read_bytes()).hexdigest(),
        summary_file_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
    )

    with pytest.raises(
        ValueError,
        match="detector results file hash mismatch",
    ):
        verify_calibrated_detector_threshold(
            calibrated,
            summary,
            results_file_sha256="f" * 64,
            summary_file_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            expected_model_id=_identity().model_id,
            expected_model_revision=_identity().model_revision,
            expected_scoring_policy_sha256=canonical_sha256(_scoring_policy()),
        )


def test_calibrated_threshold_rejects_changed_scoring_policy(
    tmp_path: Path,
) -> None:
    pairs = (
        _case("M1", DetectorCaseClass.MALICIOUS, score=0.9),
        _case("H1", DetectorCaseClass.HARD_NEGATIVE, score=0.2),
        _case("B1", DetectorCaseClass.ORDINARY_BENIGN, score=0.1),
    )
    results_path = tmp_path / "scores.jsonl"
    summary_path = tmp_path / "summary.json"
    summary = run_detector_pilot(
        tuple(case for case, _ in pairs),
        FakeDetailedDetector(scores={case.chunk.chunk_id: score for case, score in pairs}),
        threshold=0.5,
        results_path=results_path,
        summary_path=summary_path,
    )
    records = tuple(
        DetectorPilotRecord.model_validate_json(line, strict=True)
        for line in results_path.read_text(encoding="utf-8").splitlines()
    )
    calibrated = bind_detector_threshold(
        select_detector_threshold(
            records,
            maximum_false_positive_rate=0.05,
        ),
        summary,
        results_file_sha256=hashlib.sha256(results_path.read_bytes()).hexdigest(),
        summary_file_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
    )
    changed_policy = _scoring_policy().model_copy(update={"overlap_tokens": 64})

    with pytest.raises(ValueError, match="scoring policy"):
        verify_calibrated_detector_threshold(
            calibrated,
            summary,
            results_file_sha256=hashlib.sha256(results_path.read_bytes()).hexdigest(),
            summary_file_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            expected_model_id=_identity().model_id,
            expected_model_revision=_identity().model_revision,
            expected_scoring_policy_sha256=canonical_sha256(changed_policy),
        )
