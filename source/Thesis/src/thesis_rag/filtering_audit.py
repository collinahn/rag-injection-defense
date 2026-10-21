import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal, TypeVar, cast

from pydantic import Field

from thesis_rag.attack_pilot import AttackPilotSummary
from thesis_rag.canonical import JsonValue, canonical_json, canonical_sha256
from thesis_rag.domain import (
    DefenseId,
    ExperimentRecord,
    FrozenRecord,
    Probability,
    Sha256Hex,
)
from thesis_rag.pilot import (
    CleanGenerationPilotRecord,
    CleanGenerationPilotSummary,
)

_SOURCE_KEYS = {
    "d3_attack_results",
    "d4_attack_results",
    "d3_attack_summary",
    "d4_attack_summary",
    "d3_clean_results",
    "d4_clean_results",
    "d3_clean_summary",
    "d4_clean_summary",
}
_RecordT = TypeVar("_RecordT")


class FilteringDecisionAudit(FrozenRecord):
    track: Literal["forced", "clean"]
    record_count: int = Field(strict=True, ge=1)
    detector_model_id: str = Field(min_length=1)
    detector_model_revision: str = Field(min_length=1)
    detector_threshold: Probability
    decision_set_sha256: Sha256Hex


class FilteringPairedAuditReport(FrozenRecord):
    schema_version: str
    attack: FilteringDecisionAudit
    clean: FilteringDecisionAudit
    source_file_sha256: dict[str, Sha256Hex]


def audit_filtering_pilots(
    *,
    d3_attack_records: Sequence[ExperimentRecord],
    d4_attack_records: Sequence[ExperimentRecord],
    d3_attack_summary: AttackPilotSummary,
    d4_attack_summary: AttackPilotSummary,
    d3_clean_records: Sequence[CleanGenerationPilotRecord],
    d4_clean_records: Sequence[CleanGenerationPilotRecord],
    d3_clean_summary: CleanGenerationPilotSummary,
    d4_clean_summary: CleanGenerationPilotSummary,
    expected_clean_question_ids: Sequence[str],
    source_file_sha256: Mapping[str, str],
) -> FilteringPairedAuditReport:
    if set(source_file_sha256) != _SOURCE_KEYS:
        raise ValueError("filtering audit source file hashes are incomplete")
    attack = _audit_attack_decisions(
        d3_attack_records,
        d4_attack_records,
        d3_attack_summary,
        d4_attack_summary,
        source_file_sha256,
    )
    clean = _audit_clean_decisions(
        d3_clean_records,
        d4_clean_records,
        d3_clean_summary,
        d4_clean_summary,
        expected_clean_question_ids,
        source_file_sha256,
    )
    return FilteringPairedAuditReport(
        schema_version="1.0.0",
        attack=attack,
        clean=clean,
        source_file_sha256=dict(source_file_sha256),
    )


def write_filtering_paired_audit(
    path: Path,
    report: FilteringPairedAuditReport,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(raw_path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(f"{canonical_json(report)}\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _audit_attack_decisions(
    d3_records: Sequence[ExperimentRecord],
    d4_records: Sequence[ExperimentRecord],
    d3_summary: AttackPilotSummary,
    d4_summary: AttackPilotSummary,
    source_hashes: Mapping[str, str],
) -> FilteringDecisionAudit:
    detector_model_id, detector_model_revision, threshold = (
        _validate_attack_summaries(
            d3_summary,
            d4_summary,
            source_hashes,
        )
    )
    d3_by_id = _unique_records(
        d3_records,
        identity=lambda record: record.attack_id,
        label="D3 attack",
    )
    d4_by_id = _unique_records(
        d4_records,
        identity=lambda record: record.attack_id,
        label="D4 attack",
    )
    if set(d3_by_id) != set(d4_by_id) or not d3_by_id:
        raise ValueError("D3 and D4 attack record sets differ")
    if len(d3_by_id) != d3_summary.completed_count:
        raise ValueError("D3 attack summary record count mismatch")
    if len(d4_by_id) != d4_summary.completed_count:
        raise ValueError("D4 attack summary record count mismatch")
    selected_attack_ids = set(d3_summary.selected_attack_ids)
    if set(d3_by_id) != selected_attack_ids:
        raise ValueError("attack records do not match selected attack IDs")

    decisions: list[JsonValue] = []
    for attack_id in sorted(d3_by_id):
        d3 = d3_by_id[attack_id]
        d4 = d4_by_id[attack_id]
        _validate_attack_record(
            d3,
            DefenseId.D3,
            detector_model_id,
            detector_model_revision,
        )
        _validate_attack_record(
            d4,
            DefenseId.D4,
            detector_model_id,
            detector_model_revision,
        )
        d3_decision = _attack_decision(d3)
        d4_decision = _attack_decision(d4)
        if d3_decision != d4_decision:
            raise ValueError(
                f"attack detector decisions differ for {attack_id}"
            )
        decisions.append(d3_decision)
    return FilteringDecisionAudit(
        track="forced",
        record_count=len(decisions),
        detector_model_id=detector_model_id,
        detector_model_revision=detector_model_revision,
        detector_threshold=threshold,
        decision_set_sha256=canonical_sha256(decisions),
    )


def _audit_clean_decisions(
    d3_records: Sequence[CleanGenerationPilotRecord],
    d4_records: Sequence[CleanGenerationPilotRecord],
    d3_summary: CleanGenerationPilotSummary,
    d4_summary: CleanGenerationPilotSummary,
    expected_question_ids: Sequence[str],
    source_hashes: Mapping[str, str],
) -> FilteringDecisionAudit:
    detector_model_id, detector_model_revision, threshold = (
        _validate_clean_summaries(
            d3_summary,
            d4_summary,
            source_hashes,
        )
    )
    d3_by_id = _unique_records(
        d3_records,
        identity=lambda record: record.question_id,
        label="D3 clean",
    )
    d4_by_id = _unique_records(
        d4_records,
        identity=lambda record: record.question_id,
        label="D4 clean",
    )
    if set(d3_by_id) != set(d4_by_id) or not d3_by_id:
        raise ValueError("D3 and D4 clean record sets differ")
    expected_id_set = set(expected_question_ids)
    if (
        not expected_id_set
        or len(expected_id_set) != len(expected_question_ids)
        or set(d3_by_id) != expected_id_set
    ):
        raise ValueError("clean records do not match expected question IDs")
    if len(d3_by_id) != d3_summary.record_count:
        raise ValueError("D3 clean summary record count mismatch")
    if len(d4_by_id) != d4_summary.record_count:
        raise ValueError("D4 clean summary record count mismatch")

    decisions: list[JsonValue] = []
    for question_id in sorted(d3_by_id):
        d3 = d3_by_id[question_id]
        d4 = d4_by_id[question_id]
        _validate_clean_record(
            d3,
            DefenseId.D3,
            detector_model_id,
            detector_model_revision,
            threshold,
        )
        _validate_clean_record(
            d4,
            DefenseId.D4,
            detector_model_id,
            detector_model_revision,
            threshold,
        )
        d3_decision = _clean_decision(d3)
        d4_decision = _clean_decision(d4)
        if d3_decision != d4_decision:
            raise ValueError(
                f"clean detector decisions differ for {question_id}"
            )
        decisions.append(d3_decision)
    return FilteringDecisionAudit(
        track="clean",
        record_count=len(decisions),
        detector_model_id=detector_model_id,
        detector_model_revision=detector_model_revision,
        detector_threshold=threshold,
        decision_set_sha256=canonical_sha256(decisions),
    )


def _validate_attack_summaries(
    d3: AttackPilotSummary,
    d4: AttackPilotSummary,
    source_hashes: Mapping[str, str],
) -> tuple[str, str, float]:
    if d3.defense_id is not DefenseId.D3 or d4.defense_id is not DefenseId.D4:
        raise ValueError("attack audit requires D3 and D4 summaries")
    if not d3.detector_applied or not d4.detector_applied:
        raise ValueError("attack audit requires applied detectors")
    if d3.detector_threshold is None or d4.detector_threshold is None:
        raise ValueError("attack audit requires detector thresholds")
    if (
        d3.completed_count != d3.selected_count
        or d4.completed_count != d4.selected_count
        or d3.unresolved_count != 0
        or d4.unresolved_count != 0
    ):
        raise ValueError("D3/D4 attack pilots are incomplete")
    comparable = (
        "selected_attack_ids",
        "selected_count",
        "completed_count",
        "unresolved_count",
        "model_id",
        "model_revision",
        "detector_model_id",
        "detector_model_revision",
        "detector_threshold",
    )
    if any(getattr(d3, field) != getattr(d4, field) for field in comparable):
        raise ValueError("D3 and D4 attack summary bindings differ")
    if d3.results_file_sha256 != source_hashes["d3_attack_results"]:
        raise ValueError("D3 attack results file hash mismatch")
    if d4.results_file_sha256 != source_hashes["d4_attack_results"]:
        raise ValueError("D4 attack results file hash mismatch")
    return (
        d3.detector_model_id,
        d3.detector_model_revision,
        d3.detector_threshold,
    )


def _validate_clean_summaries(
    d3: CleanGenerationPilotSummary,
    d4: CleanGenerationPilotSummary,
    source_hashes: Mapping[str, str],
) -> tuple[str, str, float]:
    if d3.defense_id is not DefenseId.D3 or d4.defense_id is not DefenseId.D4:
        raise ValueError("clean audit requires D3 and D4 summaries")
    if not d3.detector_applied or not d4.detector_applied:
        raise ValueError("clean audit requires applied detectors")
    if d3.detector_threshold is None or d4.detector_threshold is None:
        raise ValueError("clean audit requires detector thresholds")
    comparable = (
        "corpus_sha256",
        "questions_sha256",
        "embedding_cache_sha256",
        "generation_config_sha256",
        "model_identity",
        "top_k",
        "record_count",
        "detector_model_id",
        "detector_model_revision",
        "detector_threshold",
    )
    if any(getattr(d3, field) != getattr(d4, field) for field in comparable):
        raise ValueError("D3 and D4 clean summary bindings differ")
    if d3.results_file_sha256 != source_hashes["d3_clean_results"]:
        raise ValueError("D3 clean results file hash mismatch")
    if d4.results_file_sha256 != source_hashes["d4_clean_results"]:
        raise ValueError("D4 clean results file hash mismatch")
    return (
        d3.detector_model_id,
        d3.detector_model_revision,
        d3.detector_threshold,
    )


def _validate_attack_record(
    record: ExperimentRecord,
    defense_id: DefenseId,
    detector_model_id: str,
    detector_model_revision: str,
) -> None:
    if (
        record.defense_id is not defense_id
        or record.track != "forced"
        or record.technical_failure
        or record.attack_id is None
    ):
        raise ValueError(f"invalid {defense_id.value} attack audit record")
    if (
        record.detector_model_id != detector_model_id
        or record.detector_model_revision != detector_model_revision
    ):
        raise ValueError(f"{defense_id.value} attack detector identity mismatch")


def _validate_clean_record(
    record: CleanGenerationPilotRecord,
    defense_id: DefenseId,
    detector_model_id: str,
    detector_model_revision: str,
    detector_threshold: float,
) -> None:
    if record.defense_id is not defense_id or record.technical_failure:
        raise ValueError(f"invalid {defense_id.value} clean audit record")
    if (
        record.detector_model_id != detector_model_id
        or record.detector_model_revision != detector_model_revision
        or record.detector_threshold != detector_threshold
    ):
        raise ValueError(f"{defense_id.value} clean detector attribution mismatch")


def _attack_decision(record: ExperimentRecord) -> JsonValue:
    return {
        "question_id": record.question_id,
        "base_context_id": record.base_context_id,
        "attack_id": record.attack_id,
        "config_hash": record.config_hash,
        "clean_source_commitment": record.clean_source_commitment,
        "attack_commitment": record.attack_commitment,
        "research_input_hash": record.research_input_hash,
        "ordered_chunk_ids": list(record.ordered_chunk_ids),
        "malicious_chunk_rank": record.malicious_chunk_rank,
        "detector_scores": cast(JsonValue, record.detector_scores),
        "blocked_chunk_ids": list(record.blocked_chunk_ids),
    }


def _clean_decision(record: CleanGenerationPilotRecord) -> JsonValue:
    return {
        "question_id": record.question_id,
        "expected_answerable": record.expected_answerable,
        "retrieved_chunk_ids": list(record.retrieved_chunk_ids),
        "ordered_chunk_ids": list(record.ordered_chunk_ids),
        "detector_scores": cast(JsonValue, record.detector_scores),
        "blocked_chunk_ids": list(record.blocked_chunk_ids),
    }


def _unique_records(
    records: Sequence[_RecordT],
    *,
    identity: Callable[[_RecordT], str | None],
    label: str,
) -> dict[str, _RecordT]:
    result: dict[str, _RecordT] = {}
    for record in records:
        record_id = identity(record)
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"{label} record identity must be nonblank")
        if record_id in result:
            raise ValueError(f"duplicate {label} record identity: {record_id}")
        result[record_id] = record
    return result
