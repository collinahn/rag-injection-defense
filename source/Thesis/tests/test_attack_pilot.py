from collections import Counter

import pytest

from thesis_rag.adapters import BypassDetector
from thesis_rag.attack_dataset import build_development_attack_factorial
from thesis_rag.attack_pilot import (
    AttackPilotSummary,
    select_factorial_pilot_attack_ids,
    summarize_attack_pilot,
)
from thesis_rag.domain import (
    AttackGoal,
    AttackPosition,
    DefenseId,
    ExperimentRecord,
    Language,
)


def test_selects_one_attack_per_goal_language_form_with_balanced_positions() -> None:
    base_context_ids = tuple(f"Q_DEV_{index:03d}" for index in range(1, 21))
    false_claims = {
        base_context_ids[index]: f"합성 허위 규정 {index}"
        for index in range(4, 8)
    }
    attacks = build_development_attack_factorial(
        base_context_ids,
        false_claims_by_context=false_claims,
    )

    selected_ids = select_factorial_pilot_attack_ids(attacks)
    selected = tuple(
        attack for attack in attacks if attack.attack_id in selected_ids
    )

    assert len(selected_ids) == 20
    assert len({attack.base_context_id for attack in selected}) == 20
    assert Counter(
        (attack.goal, attack.language, attack.form)
        for attack in selected
    ) == Counter(
        (goal, language, form)
        for goal in AttackGoal
        for language in Language
        for form in ("plain", "disguised")
    )
    assert Counter(attack.position for attack in selected) == {
        AttackPosition.HEAD: 7,
        AttackPosition.MIDDLE: 7,
        AttackPosition.TAIL: 6,
    }


def test_bypass_detector_records_that_filtering_was_not_applied() -> None:
    detector = BypassDetector()

    assert detector.model_id == "detector-not-applied"
    assert detector.model_revision == "d0-d2-development-pilot-v1"


def _record(
    index: int,
    *,
    attack_success: bool,
    defense_id: DefenseId = DefenseId.D0,
    detector_model_id: str = "detector-not-applied",
    detector_model_revision: str = "d0-d2-development-pilot-v1",
    detector_score: float = 0.0,
    blocked: bool = False,
) -> ExperimentRecord:
    malicious_chunk_id = f"MCH_{index}"
    return ExperimentRecord(
        experiment_id=f"EXP_{index}",
        job_id=f"JOB_{index}",
        track="forced",
        question_id=f"Q_DEV_{index:03d}",
        base_context_id=f"Q_DEV_{index:03d}",
        attack_id=f"ATK_DEV_{index:03d}",
        defense_id=defense_id,
        config_hash="a" * 64,
        clean_source_commitment="b" * 64,
        attack_commitment="c" * 64,
        model_id="qwen3:8b-q8_0",
        model_revision="d" * 64,
        detector_model_id=detector_model_id,
        detector_model_revision=detector_model_revision,
        research_input_hash="e" * 64,
        prompt_hash="f" * 64,
        ordered_chunk_ids=(malicious_chunk_id,),
        malicious_chunk_rank=1,
        detector_scores={malicious_chunk_id: detector_score},
        blocked_chunk_ids=((malicious_chunk_id,) if blocked else ()),
        raw_response="합성 응답",
        attack_success=attack_success,
        technical_failure=False,
        error_type=None,
        latency_ms=10.0,
        detector_latency_ms=0.1,
        generation_latency_ms=9.9,
        input_tokens=10,
        output_tokens=3,
    )


def test_summarizes_d0_attack_success_without_counting_bypass_as_a_defense() -> None:
    records = (_record(1, attack_success=True), _record(2, attack_success=False))

    summary = summarize_attack_pilot(
        records,
        defense_id=DefenseId.D0,
        selected_attack_ids=("ATK_DEV_001", "ATK_DEV_002"),
        selected_job_ids=("JOB_1", "JOB_2"),
        failure_attempt_count=0,
        manifest_job_set_sha256="1" * 64,
        results_file_sha256="2" * 64,
    )

    assert isinstance(summary, AttackPilotSummary)
    assert summary.defense_id is DefenseId.D0
    assert summary.completed_count == 2
    assert summary.unresolved_count == 0
    assert summary.success_count == 1
    assert summary.attack_success_rate == 0.5
    assert summary.detector_applied is False


def test_summarizes_nonfiltering_prompt_defenses_without_a_detector() -> None:
    record = _record(
        1,
        attack_success=False,
        defense_id=DefenseId.D1,
    )

    summary = summarize_attack_pilot(
        (record,),
        defense_id=DefenseId.D1,
        selected_attack_ids=("ATK_DEV_001",),
        selected_job_ids=("JOB_1",),
        failure_attempt_count=0,
        manifest_job_set_sha256="1" * 64,
        results_file_sha256="2" * 64,
    )

    assert summary.defense_id is DefenseId.D1
    assert summary.detector_applied is False


def test_summarizes_filtering_defense_with_real_detector_identity() -> None:
    record = _record(
        1,
        attack_success=False,
        defense_id=DefenseId.D3,
        detector_model_id="meta-llama/Llama-Prompt-Guard-2-86M",
        detector_model_revision="a" * 40,
        detector_score=0.91,
        blocked=True,
    )

    summary = summarize_attack_pilot(
        (record,),
        defense_id=DefenseId.D3,
        selected_attack_ids=("ATK_DEV_001",),
        selected_job_ids=("JOB_1",),
        failure_attempt_count=0,
        detector_threshold=0.9,
        manifest_job_set_sha256="1" * 64,
        results_file_sha256="2" * 64,
    )

    assert summary.defense_id is DefenseId.D3
    assert summary.detector_applied is True
    assert summary.detector_model_id == "meta-llama/Llama-Prompt-Guard-2-86M"
    assert summary.detector_model_revision == "a" * 40
    assert summary.detector_threshold == pytest.approx(0.9)


def test_rejects_filtering_summary_with_bypass_detector_identity() -> None:
    record = _record(
        1,
        attack_success=False,
        defense_id=DefenseId.D4,
    )

    with pytest.raises(
        ValueError,
        match="filtering attack pilot requires an applied detector",
    ):
        summarize_attack_pilot(
            (record,),
            defense_id=DefenseId.D4,
            selected_attack_ids=("ATK_DEV_001",),
            selected_job_ids=("JOB_1",),
            failure_attempt_count=0,
            detector_threshold=0.9,
            manifest_job_set_sha256="1" * 64,
            results_file_sha256="2" * 64,
        )
