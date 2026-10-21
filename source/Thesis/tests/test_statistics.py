import math

import pytest

from thesis_rag.domain import DefenseId, ExperimentRecord
from thesis_rag.statistics import (
    AttackOutcome,
    attack_outcomes_from_experiment_records,
    cluster_bootstrap_asr_reduction,
    fit_paired_defense_gee,
    holm_adjust,
    summarize_attack_outcomes,
)


def _outcomes() -> tuple[AttackOutcome, ...]:
    rows: list[AttackOutcome] = []
    patterns = {
        "CTX_001": ((True, False), (True, False), (False, False)),
        "CTX_002": ((True, True), (True, False), (True, False)),
        "CTX_003": ((False, False), (True, True), (True, False)),
        "CTX_004": ((True, False), (False, False), (True, True)),
        "CTX_005": ((True, False), (True, False), (False, True)),
        "CTX_006": ((False, False), (True, False), (False, True)),
    }
    for context_id, attacks in patterns.items():
        for attack_index, (d0_success, d4_success) in enumerate(attacks, start=1):
            attack_id = f"ATK_{context_id}_{attack_index}"
            rows.extend(
                (
                    AttackOutcome(
                        track="forced",
                        base_context_id=context_id,
                        attack_id=attack_id,
                        defense_id=DefenseId.D0,
                        attack_success=d0_success,
                    ),
                    AttackOutcome(
                        track="forced",
                        base_context_id=context_id,
                        attack_id=attack_id,
                        defense_id=DefenseId.D4,
                        attack_success=d4_success,
                    ),
                )
            )
    return tuple(rows)


def _experiment_records(
    outcomes: tuple[AttackOutcome, ...],
) -> tuple[ExperimentRecord, ...]:
    return tuple(
        ExperimentRecord(
            experiment_id=f"EXP_{index:04d}",
            job_id=f"JOB_{index:04d}",
            track=outcome.track,
            question_id=outcome.base_context_id,
            base_context_id=outcome.base_context_id,
            attack_id=outcome.attack_id,
            defense_id=outcome.defense_id,
            config_hash="a" * 64,
            clean_source_commitment="b" * 64,
            attack_commitment="c" * 64,
            model_id="generator",
            model_revision="d" * 64,
            detector_model_id="detector",
            detector_model_revision="e" * 40,
            research_input_hash=f"{index:064x}",
            prompt_hash="f" * 64,
            ordered_chunk_ids=("CH_001",),
            malicious_chunk_rank=1,
            detector_scores={"CH_001": 0.1},
            blocked_chunk_ids=(),
            raw_response="합성 공격 응답",
            attack_success=outcome.attack_success,
            technical_failure=False,
            error_type=None,
            latency_ms=1.0,
            detector_latency_ms=0.1,
            generation_latency_ms=0.9,
            input_tokens=100,
            output_tokens=20,
        )
        for index, outcome in enumerate(outcomes, start=1)
    )


def test_converts_verified_experiment_records_to_paired_attack_outcomes() -> None:
    outcomes = _outcomes()
    records = _experiment_records(outcomes)

    converted = attack_outcomes_from_experiment_records(
        records,
        expected_track="forced",
        required_defense_ids=(DefenseId.D0, DefenseId.D4),
    )

    assert converted == outcomes


def test_attack_outcome_conversion_rejects_technical_failures() -> None:
    records = list(_experiment_records(_outcomes()))
    records[0] = records[0].model_copy(
        update={
            "raw_response": "",
            "attack_success": None,
            "technical_failure": True,
            "error_type": "timeout",
        }
    )

    with pytest.raises(ValueError, match="technical failures"):
        attack_outcomes_from_experiment_records(
            tuple(records),
            expected_track="forced",
            required_defense_ids=(DefenseId.D0, DefenseId.D4),
        )


def test_attack_outcome_conversion_rejects_incomplete_defense_pairs() -> None:
    records = _experiment_records(_outcomes())[:-1]

    with pytest.raises(ValueError, match="required defense coverage"):
        attack_outcomes_from_experiment_records(
            records,
            expected_track="forced",
            required_defense_ids=(DefenseId.D0, DefenseId.D4),
        )


def test_summarizes_attack_success_rates_and_asr_reduction() -> None:
    summary = summarize_attack_outcomes(_outcomes())

    assert summary.cluster_count == 6
    assert summary.attack_count == 18
    assert summary.record_count == 36
    assert summary.by_defense[0].defense_id is DefenseId.D0
    assert summary.by_defense[0].success_count == 12
    assert summary.by_defense[0].trial_count == 18
    assert summary.by_defense[0].asr == pytest.approx(2 / 3)
    assert summary.by_defense[1].defense_id is DefenseId.D4
    assert summary.by_defense[1].success_count == 5
    assert summary.by_defense[1].asr == pytest.approx(5 / 18)


def test_cluster_bootstrap_is_deterministic_and_reports_asr_reduction() -> None:
    first = cluster_bootstrap_asr_reduction(
        _outcomes(),
        baseline=DefenseId.D0,
        treatment=DefenseId.D4,
        iterations=2_000,
        seed=42,
    )
    second = cluster_bootstrap_asr_reduction(
        tuple(reversed(_outcomes())),
        baseline=DefenseId.D0,
        treatment=DefenseId.D4,
        iterations=2_000,
        seed=42,
    )

    assert first == second
    assert first.cluster_count == 6
    assert first.attack_count == 18
    assert first.baseline_asr == pytest.approx(2 / 3)
    assert first.treatment_asr == pytest.approx(5 / 18)
    assert first.asr_reduction == pytest.approx(7 / 18)
    assert first.ci_lower <= first.asr_reduction <= first.ci_upper
    assert 0.0 <= first.ci_lower <= first.ci_upper <= 1.0


def test_cluster_bootstrap_rejects_missing_paired_defense_row() -> None:
    rows = _outcomes()[:-1]

    with pytest.raises(
        ValueError,
        match="paired defense coverage mismatch.*ATK_CTX_006_3",
    ):
        cluster_bootstrap_asr_reduction(
            rows,
            baseline=DefenseId.D0,
            treatment=DefenseId.D4,
            iterations=100,
            seed=42,
        )


def test_attack_summary_rejects_duplicate_attack_defense_rows() -> None:
    rows = _outcomes()

    with pytest.raises(ValueError, match="duplicate attack-defense outcome"):
        summarize_attack_outcomes((*rows, rows[0]))


def test_holm_adjustment_is_monotone_in_sorted_p_value_order() -> None:
    adjusted = holm_adjust(
        {
            "D1_vs_D0": 0.01,
            "D2_vs_D0": 0.04,
            "D3_vs_D0": 0.03,
            "D4_vs_D1": 0.20,
        }
    )

    assert adjusted.raw_p_values == {
        "D1_vs_D0": 0.01,
        "D2_vs_D0": 0.04,
        "D3_vs_D0": 0.03,
        "D4_vs_D1": 0.20,
    }
    assert adjusted.adjusted_p_values == pytest.approx(
        {
            "D1_vs_D0": 0.04,
            "D2_vs_D0": 0.09,
            "D3_vs_D0": 0.09,
            "D4_vs_D1": 0.20,
        }
    )


@pytest.mark.parametrize("invalid", (-0.1, 1.1, math.nan, math.inf))
def test_holm_rejects_invalid_p_values(invalid: float) -> None:
    with pytest.raises(ValueError, match=r"p-values must be finite and in \[0, 1\]"):
        holm_adjust({"comparison": invalid})


def test_gee_reports_d4_odds_ratio_against_d0_with_robust_covariance() -> None:
    result = fit_paired_defense_gee(
        _outcomes(),
        baseline=DefenseId.D0,
        treatment=DefenseId.D4,
    )

    assert result.baseline is DefenseId.D0
    assert result.treatment is DefenseId.D4
    assert result.cluster_count == 6
    assert result.attack_count == 18
    assert result.covariance_type == "robust"
    assert result.working_correlation == "exchangeable"
    assert 0.0 < result.odds_ratio < 1.0
    assert result.ci_lower < result.odds_ratio < result.ci_upper
    assert 0.0 <= result.p_value <= 1.0


def test_gee_rejects_outcome_without_variation() -> None:
    rows = tuple(
        row.model_copy(update={"attack_success": False})
        for row in _outcomes()
    )

    with pytest.raises(ValueError, match="GEE requires both success and failure outcomes"):
        fit_paired_defense_gee(
            rows,
            baseline=DefenseId.D0,
            treatment=DefenseId.D4,
        )
