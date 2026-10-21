import pytest

from thesis_rag.attack_pilot import AttackPilotSummary
from thesis_rag.domain import DefenseId, ExperimentRecord
from thesis_rag.filtering_audit import audit_filtering_pilots
from thesis_rag.generation import OllamaModelIdentity
from thesis_rag.pilot import (
    CleanGenerationPilotRecord,
    CleanGenerationPilotSummary,
)

_DETECTOR_ID = "meta-llama/Llama-Prompt-Guard-2-86M"
_DETECTOR_REVISION = "a" * 40


def _attack_record(defense_id: DefenseId) -> ExperimentRecord:
    return ExperimentRecord(
        experiment_id=f"EXP_{defense_id.value}",
        job_id=f"JOB_{defense_id.value}",
        track="forced",
        question_id="Q_DEV_001",
        base_context_id="Q_DEV_001",
        attack_id="ATK_DEV_001",
        defense_id=defense_id,
        config_hash="1" * 64,
        clean_source_commitment="2" * 64,
        attack_commitment="3" * 64,
        model_id="qwen3:8b-q8_0",
        model_revision="4" * 64,
        detector_model_id=_DETECTOR_ID,
        detector_model_revision=_DETECTOR_REVISION,
        research_input_hash="5" * 64,
        prompt_hash="6" * 64,
        ordered_chunk_ids=("CH_1", "CH_2"),
        malicious_chunk_rank=1,
        detector_scores={"CH_1": 0.91, "CH_2": 0.02},
        blocked_chunk_ids=("CH_1",),
        raw_response="합성 응답",
        attack_success=False,
        technical_failure=False,
        error_type=None,
        latency_ms=10.0,
        detector_latency_ms=2.0,
        generation_latency_ms=8.0,
        input_tokens=10,
        output_tokens=3,
    )


def _attack_summary(defense_id: DefenseId) -> AttackPilotSummary:
    return AttackPilotSummary(
        schema_version="1.0.0",
        track="forced",
        defense_id=defense_id,
        selection_method="one_per_goal_language_form_cyclic_position",
        selected_attack_ids=("ATK_DEV_001",),
        selected_job_ids=(f"JOB_{defense_id.value}",),
        selected_count=1,
        completed_count=1,
        unresolved_count=0,
        success_count=0,
        attack_success_rate=0.0,
        failure_attempt_count=0,
        detector_applied=True,
        model_id="qwen3:8b-q8_0",
        model_revision="4" * 64,
        detector_model_id=_DETECTOR_ID,
        detector_model_revision=_DETECTOR_REVISION,
        manifest_job_set_sha256="7" * 64,
        results_file_sha256="8" * 64,
        detector_threshold=0.9,
    )


def _clean_record(defense_id: DefenseId) -> CleanGenerationPilotRecord:
    return CleanGenerationPilotRecord(
        question_id="Q_DEV_001",
        expected_answerable=True,
        defense_id=defense_id,
        ordered_chunk_ids=("CH_2",),
        prompt_sha256="1" * 64,
        raw_response="합성 응답",
        raw_response_sha256="2" * 64,
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
        latency_ms=10.0,
        retrieved_chunk_ids=("CH_1", "CH_2"),
        detector_scores={"CH_1": 0.91, "CH_2": 0.02},
        blocked_chunk_ids=("CH_1",),
        detector_model_id=_DETECTOR_ID,
        detector_model_revision=_DETECTOR_REVISION,
        detector_threshold=0.9,
        detector_latency_ms=2.0,
        generation_latency_ms=8.0,
    )


def _clean_summary(defense_id: DefenseId) -> CleanGenerationPilotSummary:
    identity = OllamaModelIdentity(
        model_id="qwen3:8b-q8_0",
        model_revision="4" * 64,
        chat_template_sha256="5" * 64,
        quantization="Q8_0",
        parameter_size="8.2B",
        runtime_version="ollama 0.32.4",
    )
    return CleanGenerationPilotSummary(
        schema_version="1.0.0",
        corpus_sha256="1" * 64,
        questions_sha256="2" * 64,
        embedding_cache_sha256="3" * 64,
        generation_config_sha256="4" * 64,
        model_identity=identity,
        defense_id=defense_id,
        top_k=5,
        record_count=1,
        json_parse_success_count=0,
        json_parse_success_rate=0.0,
        valid_citation_count=0,
        valid_citation_rate=0.0,
        answerability_match_count=0,
        answerability_match_rate=0.0,
        empty_response_count=0,
        repetitive_response_count=0,
        length_limited_count=0,
        technical_failure_count=0,
        technical_failure_rate=0.0,
        automated_gate_pass=True,
        human_accuracy_review_required=True,
        response_set_sha256="6" * 64,
        results_file_sha256="7" * 64,
        detector_applied=True,
        detector_model_id=_DETECTOR_ID,
        detector_model_revision=_DETECTOR_REVISION,
        detector_threshold=0.9,
        blocked_chunk_count=1,
        all_chunks_blocked_question_count=0,
    )


def _audit(
    *,
    d4_attack: ExperimentRecord | None = None,
    d4_clean: CleanGenerationPilotRecord | None = None,
):
    return audit_filtering_pilots(
        d3_attack_records=(_attack_record(DefenseId.D3),),
        d4_attack_records=(d4_attack or _attack_record(DefenseId.D4),),
        d3_attack_summary=_attack_summary(DefenseId.D3),
        d4_attack_summary=_attack_summary(DefenseId.D4),
        d3_clean_records=(_clean_record(DefenseId.D3),),
        d4_clean_records=(d4_clean or _clean_record(DefenseId.D4),),
        d3_clean_summary=_clean_summary(DefenseId.D3),
        d4_clean_summary=_clean_summary(DefenseId.D4),
        expected_clean_question_ids=("Q_DEV_001",),
        source_file_sha256={
            "d3_attack_results": "8" * 64,
            "d4_attack_results": "8" * 64,
            "d3_attack_summary": "c" * 64,
            "d4_attack_summary": "d" * 64,
            "d3_clean_results": "7" * 64,
            "d4_clean_results": "7" * 64,
            "d3_clean_summary": "0" * 64,
            "d4_clean_summary": "9" * 64,
        },
    )


def test_audits_equivalent_d3_d4_detector_decisions() -> None:
    report = _audit()

    assert report.attack.record_count == 1
    assert report.clean.record_count == 1
    assert report.attack.detector_threshold == pytest.approx(0.9)
    assert report.clean.detector_model_id == _DETECTOR_ID


def test_rejects_a_single_d3_d4_detector_score_difference() -> None:
    mismatched = _attack_record(DefenseId.D4).model_copy(
        update={"detector_scores": {"CH_1": 0.90, "CH_2": 0.02}}
    )

    with pytest.raises(ValueError, match="attack detector decisions differ"):
        _audit(d4_attack=mismatched)


def test_rejects_a_single_d3_d4_blocked_chunk_difference() -> None:
    mismatched = _clean_record(DefenseId.D4).model_copy(
        update={"blocked_chunk_ids": ()}
    )

    with pytest.raises(ValueError, match="clean detector decisions differ"):
        _audit(d4_clean=mismatched)


def test_rejects_clean_record_detector_attribution_mismatch() -> None:
    mismatched = _clean_record(DefenseId.D4).model_copy(
        update={"detector_threshold": 0.8}
    )

    with pytest.raises(ValueError, match="clean detector attribution mismatch"):
        _audit(d4_clean=mismatched)


def test_rejects_equally_incomplete_attack_runs() -> None:
    d3_summary = _attack_summary(DefenseId.D3).model_copy(
        update={
            "selected_attack_ids": ("ATK_DEV_001", "ATK_DEV_002"),
            "selected_job_ids": ("JOB_D3", "JOB_D3_002"),
            "selected_count": 2,
            "completed_count": 1,
            "unresolved_count": 1,
        }
    )
    d4_summary = _attack_summary(DefenseId.D4).model_copy(
        update={
            "selected_attack_ids": ("ATK_DEV_001", "ATK_DEV_002"),
            "selected_job_ids": ("JOB_D4", "JOB_D4_002"),
            "selected_count": 2,
            "completed_count": 1,
            "unresolved_count": 1,
        }
    )

    with pytest.raises(ValueError, match="attack pilots are incomplete"):
        audit_filtering_pilots(
            d3_attack_records=(_attack_record(DefenseId.D3),),
            d4_attack_records=(_attack_record(DefenseId.D4),),
            d3_attack_summary=d3_summary,
            d4_attack_summary=d4_summary,
            d3_clean_records=(_clean_record(DefenseId.D3),),
            d4_clean_records=(_clean_record(DefenseId.D4),),
            d3_clean_summary=_clean_summary(DefenseId.D3),
            d4_clean_summary=_clean_summary(DefenseId.D4),
            expected_clean_question_ids=("Q_DEV_001",),
            source_file_sha256={
                "d3_attack_results": "8" * 64,
                "d4_attack_results": "8" * 64,
                "d3_attack_summary": "c" * 64,
                "d4_attack_summary": "d" * 64,
                "d3_clean_results": "7" * 64,
                "d4_clean_results": "7" * 64,
                "d3_clean_summary": "0" * 64,
                "d4_clean_summary": "9" * 64,
            },
        )


def test_rejects_attack_records_not_bound_to_selected_ids() -> None:
    d3_summary = _attack_summary(DefenseId.D3).model_copy(
        update={"selected_attack_ids": ("ATK_DEV_999",)}
    )
    d4_summary = _attack_summary(DefenseId.D4).model_copy(
        update={"selected_attack_ids": ("ATK_DEV_999",)}
    )

    with pytest.raises(ValueError, match="selected attack IDs"):
        audit_filtering_pilots(
            d3_attack_records=(_attack_record(DefenseId.D3),),
            d4_attack_records=(_attack_record(DefenseId.D4),),
            d3_attack_summary=d3_summary,
            d4_attack_summary=d4_summary,
            d3_clean_records=(_clean_record(DefenseId.D3),),
            d4_clean_records=(_clean_record(DefenseId.D4),),
            d3_clean_summary=_clean_summary(DefenseId.D3),
            d4_clean_summary=_clean_summary(DefenseId.D4),
            expected_clean_question_ids=("Q_DEV_001",),
            source_file_sha256={
                "d3_attack_results": "8" * 64,
                "d4_attack_results": "8" * 64,
                "d3_attack_summary": "c" * 64,
                "d4_attack_summary": "d" * 64,
                "d3_clean_results": "7" * 64,
                "d4_clean_results": "7" * 64,
                "d3_clean_summary": "0" * 64,
                "d4_clean_summary": "9" * 64,
            },
        )


def test_rejects_equally_incomplete_clean_runs() -> None:
    with pytest.raises(ValueError, match="expected question IDs"):
        audit_filtering_pilots(
            d3_attack_records=(_attack_record(DefenseId.D3),),
            d4_attack_records=(_attack_record(DefenseId.D4),),
            d3_attack_summary=_attack_summary(DefenseId.D3),
            d4_attack_summary=_attack_summary(DefenseId.D4),
            d3_clean_records=(_clean_record(DefenseId.D3),),
            d4_clean_records=(_clean_record(DefenseId.D4),),
            d3_clean_summary=_clean_summary(DefenseId.D3),
            d4_clean_summary=_clean_summary(DefenseId.D4),
            expected_clean_question_ids=("Q_DEV_001", "Q_DEV_002"),
            source_file_sha256={
                "d3_attack_results": "8" * 64,
                "d4_attack_results": "8" * 64,
                "d3_attack_summary": "c" * 64,
                "d4_attack_summary": "d" * 64,
                "d3_clean_results": "7" * 64,
                "d4_clean_results": "7" * 64,
                "d3_clean_summary": "0" * 64,
                "d4_clean_summary": "9" * 64,
            },
        )
