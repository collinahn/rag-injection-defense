import os
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import Field, StrictBool

from thesis_rag.canonical import canonical_json
from thesis_rag.domain import (
    AttackCase,
    AttackGoal,
    AttackPosition,
    DefenseId,
    ExperimentRecord,
    FiniteFloat,
    FrozenRecord,
    Language,
    Sha256Hex,
)

_POSITIONS = (
    AttackPosition.HEAD,
    AttackPosition.MIDDLE,
    AttackPosition.TAIL,
)


class AttackPilotSummary(FrozenRecord):
    schema_version: str
    track: Literal["forced"]
    defense_id: DefenseId
    selection_method: str
    selected_attack_ids: tuple[str, ...]
    selected_job_ids: tuple[str, ...]
    selected_count: int = Field(strict=True, ge=1)
    completed_count: int = Field(strict=True, ge=0)
    unresolved_count: int = Field(strict=True, ge=0)
    success_count: int = Field(strict=True, ge=0)
    attack_success_rate: FiniteFloat
    failure_attempt_count: int = Field(strict=True, ge=0)
    detector_applied: StrictBool
    model_id: str
    model_revision: str
    detector_model_id: str
    detector_model_revision: str
    manifest_job_set_sha256: Sha256Hex
    results_file_sha256: Sha256Hex
    detector_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


def select_factorial_pilot_attack_ids(
    attacks: Sequence[AttackCase],
) -> tuple[str, ...]:
    attack_ids = tuple(attack.attack_id for attack in attacks)
    if len(set(attack_ids)) != len(attack_ids):
        raise ValueError("attack pilot requires unique attack IDs")

    grouped: dict[
        tuple[AttackGoal, Language, str],
        list[AttackCase],
    ] = defaultdict(list)
    for attack in attacks:
        grouped[(attack.goal, attack.language, attack.form)].append(attack)

    expected_keys = {
        (goal, language, form)
        for goal in AttackGoal
        for language in Language
        for form in ("plain", "disguised")
    }
    if set(grouped) != expected_keys:
        raise ValueError("attack pilot requires the full goal-language-form factorial")

    selected: list[AttackCase] = []
    for index, key in enumerate(
        sorted(
            expected_keys,
            key=lambda item: (item[0].value, item[1].value, item[2]),
        )
    ):
        candidates = grouped[key]
        by_position = {candidate.position: candidate for candidate in candidates}
        if len(candidates) != len(_POSITIONS) or set(by_position) != set(_POSITIONS):
            raise ValueError(
                "each attack pilot factor group requires one head, middle, and tail case"
            )
        selected.append(by_position[_POSITIONS[index % len(_POSITIONS)]])

    if len({attack.base_context_id for attack in selected}) != len(selected):
        raise ValueError("attack pilot selection requires unique base contexts")
    return tuple(attack.attack_id for attack in selected)


def summarize_attack_pilot(
    records: Sequence[ExperimentRecord],
    *,
    defense_id: DefenseId,
    selected_attack_ids: tuple[str, ...],
    selected_job_ids: tuple[str, ...],
    failure_attempt_count: int,
    manifest_job_set_sha256: str,
    results_file_sha256: str,
    detector_threshold: float | None = None,
) -> AttackPilotSummary:
    if not selected_attack_ids or len(selected_attack_ids) != len(selected_job_ids):
        raise ValueError("attack pilot selection IDs must be non-empty and paired")
    if len(set(selected_attack_ids)) != len(selected_attack_ids):
        raise ValueError("attack pilot attack IDs must be unique")
    if len(set(selected_job_ids)) != len(selected_job_ids):
        raise ValueError("attack pilot job IDs must be unique")
    if failure_attempt_count < 0:
        raise ValueError("attack pilot failure count must be nonnegative")
    if not records:
        raise ValueError("attack pilot summary requires at least one completed record")

    selected_job_set = set(selected_job_ids)
    if any(record.job_id not in selected_job_set for record in records):
        raise ValueError("attack pilot record is absent from the selected jobs")
    record_job_ids = tuple(record.job_id for record in records)
    if len(set(record_job_ids)) != len(record_job_ids):
        raise ValueError("attack pilot records contain duplicate jobs")
    selected_attack_set = set(selected_attack_ids)
    if any(record.attack_id not in selected_attack_set for record in records):
        raise ValueError("attack pilot record has an unexpected attack")
    if any(
        record.track != "forced"
        or record.defense_id is not defense_id
        or record.technical_failure
        or record.attack_success is None
        for record in records
    ):
        raise ValueError(
            "attack pilot summary requires completed forced records "
            "for the selected defense"
        )

    identities = {
        (
            record.model_id,
            record.model_revision,
            record.detector_model_id,
            record.detector_model_revision,
        )
        for record in records
    }
    if len(identities) != 1:
        raise ValueError("attack pilot records must share model identities")
    model_id, model_revision, detector_model_id, detector_model_revision = next(
        iter(identities)
    )
    detector_applied = defense_id in {DefenseId.D3, DefenseId.D4}
    if detector_applied:
        if detector_model_id == "detector-not-applied":
            raise ValueError(
                "filtering attack pilot requires an applied detector"
            )
        if detector_threshold is None:
            raise ValueError(
                "filtering attack pilot requires a calibrated detector threshold"
            )
    elif detector_model_id != "detector-not-applied":
        raise ValueError(
            "nonfiltering attack pilot must record a non-applied detector"
        )
    elif detector_threshold is not None:
        raise ValueError(
            "nonfiltering attack pilot must not record a detector threshold"
        )

    completed_count = len(records)
    success_count = sum(record.attack_success is True for record in records)
    return AttackPilotSummary(
        schema_version="1.0.0",
        track="forced",
        defense_id=defense_id,
        selection_method="one_per_goal_language_form_cyclic_position",
        selected_attack_ids=selected_attack_ids,
        selected_job_ids=selected_job_ids,
        selected_count=len(selected_job_ids),
        completed_count=completed_count,
        unresolved_count=len(selected_job_ids) - completed_count,
        success_count=success_count,
        attack_success_rate=success_count / completed_count,
        failure_attempt_count=failure_attempt_count,
        detector_applied=detector_applied,
        model_id=model_id,
        model_revision=model_revision,
        detector_model_id=detector_model_id,
        detector_model_revision=detector_model_revision,
        manifest_job_set_sha256=manifest_job_set_sha256,
        results_file_sha256=results_file_sha256,
        detector_threshold=detector_threshold,
    )


def write_attack_pilot_summary(
    path: Path,
    summary: AttackPilotSummary,
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
            stream.write(f"{canonical_json(summary)}\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
