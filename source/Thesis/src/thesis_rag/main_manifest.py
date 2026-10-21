import hashlib
import math
import os
import tempfile
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, TypeVar, cast

import numpy as np
from pydantic import Field, field_validator, model_validator

from thesis_rag.attack_dataset import validate_test_attack_factorial
from thesis_rag.canonical import (
    JsonValue,
    canonical_json,
    canonical_sha256,
    sha256_text,
    stable_id,
)
from thesis_rag.detector_pilot import (
    DetectorCaseClass,
    DetectorCaseReviewStatus,
    DetectorEvaluationCase,
)
from thesis_rag.domain import (
    AttackCase,
    AttackForm,
    AttackGoal,
    AttackPosition,
    Chunk,
    DefenseId,
    ExperimentJob,
    FrozenRecord,
    Language,
    Probability,
    Sha256Hex,
    Split,
)
from thesis_rag.manifests import (
    build_clean_jobs,
    build_e2e_jobs,
    reconstruct_job_from_research_inputs,
    verify_e2e_job_from_sources,
)
from thesis_rag.qa import (
    QuestionReviewStatus,
    QuestionType,
    ResearchQuestion,
)
from thesis_rag.retrieval import exact_cosine_top_k

_E2E_SELECTION_POLICY: Literal["balanced-factorial-milp-v1"] = (
    "balanced-factorial-milp-v1"
)
_T = TypeVar("_T")


class E2ESelectionSummary(FrozenRecord):
    schema_version: Literal["1.0.0"] = "1.0.0"
    policy_version: Literal["balanced-factorial-milp-v1"] = (
        _E2E_SELECTION_POLICY
    )
    context_count: int = Field(strict=True, ge=1)
    attack_count: int = Field(strict=True, ge=1)
    single_source_context_count: int = Field(strict=True, ge=0)
    multi_source_context_count: int = Field(strict=True, ge=0)
    selected_base_context_ids: tuple[str, ...]
    selected_attack_ids: tuple[str, ...]
    goal_counts: dict[str, int]
    language_counts: dict[str, int]
    form_counts: dict[str, int]
    position_counts: dict[str, int]
    factorial_cell_counts: dict[str, int]
    selection_sha256: Sha256Hex

    @model_validator(mode="after")
    def validate_selection_commitment(self) -> "E2ESelectionSummary":
        expected = canonical_sha256(
            cast(
                JsonValue,
                {
                    "policy_version": self.policy_version,
                    "selected_base_context_ids": list(
                        self.selected_base_context_ids
                    ),
                    "selected_attack_ids": list(self.selected_attack_ids),
                },
            )
        )
        if self.selection_sha256 != expected:
            raise ValueError("E2E selection hash mismatch")
        return self


class GenerationManifestSummary(FrozenRecord):
    schema_version: Literal["1.0.0"] = "1.0.0"
    track: Literal["forced", "e2e", "clean"]
    job_count: int = Field(strict=True, ge=1)
    condition_count: int = Field(strict=True, ge=1)
    defense_ids: tuple[DefenseId, ...]
    config_sha256: Sha256Hex
    shuffle_seed: int
    manifest_sha256: Sha256Hex


class DetectorManifestEntry(FrozenRecord):
    schema_version: Literal["1.0.0"] = "1.0.0"
    job_id: str = Field(min_length=1)
    case: DetectorEvaluationCase
    case_commitment: Sha256Hex
    config_sha256: Sha256Hex
    detector_model_id: str = Field(min_length=1)
    detector_model_revision: str = Field(min_length=1)
    scoring_policy_sha256: Sha256Hex
    threshold: Probability

    @field_validator(
        "job_id",
        "detector_model_id",
        "detector_model_revision",
    )
    @classmethod
    def strip_nonblank_fields(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("detector manifest strings must be nonblank")
        return stripped

    @model_validator(mode="after")
    def validate_commitments(self) -> "DetectorManifestEntry":
        if canonical_sha256(self.case) != self.case_commitment:
            raise ValueError("detector case commitment mismatch")
        payload = self.model_dump(mode="json", exclude={"job_id"})
        expected_job_id = stable_id("DET_JOB", cast(JsonValue, payload))
        if self.job_id != expected_job_id:
            raise ValueError("detector manifest job ID mismatch")
        return self


class DetectorManifestSummary(FrozenRecord):
    schema_version: Literal["1.0.0"] = "1.0.0"
    track: Literal["detector"] = "detector"
    job_count: int = Field(strict=True, ge=1)
    malicious_count: int = Field(strict=True, ge=0)
    hard_negative_count: int = Field(strict=True, ge=0)
    ordinary_benign_count: int = Field(strict=True, ge=0)
    config_sha256: Sha256Hex
    detector_model_id: str
    detector_model_revision: str
    scoring_policy_sha256: Sha256Hex
    threshold: Probability
    shuffle_seed: int
    manifest_sha256: Sha256Hex


def select_main_e2e_contexts(
    questions: Sequence[ResearchQuestion],
    attacks: Sequence[AttackCase],
    *,
    context_count: int = 30,
    multi_source_context_count: int = 5,
) -> E2ESelectionSummary:
    if any(
        question.split is not Split.TEST
        or question.review_status is not QuestionReviewStatus.RESEARCHER_APPROVED
        for question in questions
    ):
        raise ValueError("E2E selection requires approved test questions")
    validate_test_attack_factorial(attacks, questions)
    questions_by_id = _unique_by_id(
        questions,
        identity=lambda question: question.question_id,
        label="question",
    )
    attacks_by_context: dict[str, list[AttackCase]] = {}
    for attack in attacks:
        attacks_by_context.setdefault(attack.base_context_id, []).append(attack)
    context_ids = tuple(sorted(attacks_by_context))
    if context_count != 30 or multi_source_context_count != 5:
        raise ValueError("main E2E design requires 30 contexts including 5 multi-source")
    if any(len(group) != 3 for group in attacks_by_context.values()):
        raise ValueError("main E2E base contexts must contain three attacks")

    selected_attack_count = context_count * 3
    factor_targets = {
        "goal": selected_attack_count // len(AttackGoal),
        "language": selected_attack_count // len(Language),
        "form": selected_attack_count // len(AttackForm),
        "position": selected_attack_count // len(AttackPosition),
    }
    if any(
        factor_targets[name] * level_count != selected_attack_count
        for name, level_count in (
            ("goal", len(AttackGoal)),
            ("language", len(Language)),
            ("form", len(AttackForm)),
            ("position", len(AttackPosition)),
        )
    ):
        raise ValueError("main E2E factor targets must divide selected attacks")

    rows: list[list[int]] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []

    def add_constraint(
        values: list[int],
        lower: int,
        upper: int,
    ) -> None:
        rows.append(values)
        lower_bounds.append(float(lower))
        upper_bounds.append(float(upper))

    add_constraint([1] * len(context_ids), context_count, context_count)
    add_constraint(
        [
            int(
                questions_by_id[context_id].question_type
                is QuestionType.MULTI_SOURCE
            )
            for context_id in context_ids
        ],
        multi_source_context_count,
        multi_source_context_count,
    )
    for field_name, values in (
        ("goal", tuple(AttackGoal)),
        ("language", tuple(Language)),
        ("form", tuple(AttackForm)),
        ("position", tuple(AttackPosition)),
    ):
        target = factor_targets[field_name]
        for value in values:
            add_constraint(
                [
                    sum(
                        getattr(attack, field_name) is value
                        for attack in attacks_by_context[context_id]
                    )
                    for context_id in context_ids
                ],
                target,
                target,
            )

    factorial_cells = tuple(
        (goal, language, form, position)
        for goal in AttackGoal
        for language in Language
        for form in AttackForm
        for position in AttackPosition
    )
    lower_cell_count = selected_attack_count // len(factorial_cells)
    upper_cell_count = math.ceil(
        selected_attack_count / len(factorial_cells)
    )
    for cell in factorial_cells:
        add_constraint(
            [
                sum(
                    (
                        attack.goal,
                        attack.language,
                        attack.form,
                        attack.position,
                    )
                    == cell
                    for attack in attacks_by_context[context_id]
                )
                for context_id in context_ids
            ],
            lower_cell_count,
            upper_cell_count,
        )

    try:
        from scipy.optimize import (  # type: ignore[import-untyped]
            Bounds,
            LinearConstraint,
            milp,
        )
    except ImportError as error:
        raise RuntimeError(
            "E2E selection requires the project analysis dependencies"
        ) from error
    objective = np.asarray(
        [
            int(
                hashlib.sha256(
                    f"{_E2E_SELECTION_POLICY}:{context_id}".encode("ascii")
                ).hexdigest(),
                16,
            )
            / 2**256
            for context_id in context_ids
        ],
        dtype=np.float64,
    )
    result = milp(
        objective,
        integrality=np.ones(len(context_ids), dtype=np.int8),
        bounds=Bounds(
            np.zeros(len(context_ids)),
            np.ones(len(context_ids)),
        ),
        constraints=LinearConstraint(
            np.asarray(rows, dtype=np.float64),
            np.asarray(lower_bounds, dtype=np.float64),
            np.asarray(upper_bounds, dtype=np.float64),
        ),
        options={"presolve": True},
    )
    if not bool(result.success) or result.x is None:
        raise ValueError("unable to satisfy balanced main E2E selection")
    selected_context_ids = tuple(
        context_id
        for context_id, selected in zip(
            context_ids,
            result.x,
            strict=True,
        )
        if selected > 0.5
    )
    if len(selected_context_ids) != context_count:
        raise ValueError("E2E optimizer returned an invalid context count")
    selected_context_set = set(selected_context_ids)
    selected_attacks = tuple(
        sorted(
            (
                attack
                for attack in attacks
                if attack.base_context_id in selected_context_set
            ),
            key=lambda attack: attack.attack_id,
        )
    )
    cell_counts = Counter(
        _factorial_cell_key(attack) for attack in selected_attacks
    )
    selection_payload = cast(
        JsonValue,
        {
            "policy_version": _E2E_SELECTION_POLICY,
            "selected_base_context_ids": list(selected_context_ids),
            "selected_attack_ids": [
                attack.attack_id for attack in selected_attacks
            ],
        },
    )
    return E2ESelectionSummary(
        context_count=len(selected_context_ids),
        attack_count=len(selected_attacks),
        single_source_context_count=sum(
            questions_by_id[context_id].question_type
            is QuestionType.SINGLE_SOURCE
            for context_id in selected_context_ids
        ),
        multi_source_context_count=sum(
            questions_by_id[context_id].question_type
            is QuestionType.MULTI_SOURCE
            for context_id in selected_context_ids
        ),
        selected_base_context_ids=selected_context_ids,
        selected_attack_ids=tuple(
            attack.attack_id for attack in selected_attacks
        ),
        goal_counts=_enum_counts(selected_attacks, "goal", AttackGoal),
        language_counts=_enum_counts(selected_attacks, "language", Language),
        form_counts=_enum_counts(selected_attacks, "form", AttackForm),
        position_counts=_enum_counts(
            selected_attacks,
            "position",
            AttackPosition,
        ),
        factorial_cell_counts={
            key: cell_counts.get(key, 0)
            for key in sorted(
                _factorial_cell_key_from_values(*cell)
                for cell in factorial_cells
            )
        },
        selection_sha256=canonical_sha256(selection_payload),
    )


def build_clean_manifest(
    chunks: Sequence[Chunk],
    questions: Sequence[ResearchQuestion],
    *,
    chunk_vectors: object,
    question_vectors: object,
    top_k: int,
    defense_ids: tuple[DefenseId, ...],
    config_hash: str,
    shuffle_seed: int,
) -> tuple[ExperimentJob, ...]:
    if not config_hash.strip():
        raise ValueError("clean manifest config hash must be nonblank")
    if any(
        question.split is not Split.TEST
        or question.review_status is not QuestionReviewStatus.RESEARCHER_APPROVED
        for question in questions
    ):
        raise ValueError("clean manifest requires approved test questions")
    chunks_by_id = _unique_by_id(
        chunks,
        identity=lambda chunk: chunk.chunk_id,
        label="chunk",
    )
    questions_by_id = _unique_by_id(
        questions,
        identity=lambda question: question.question_id,
        label="question",
    )
    chunk_matrix = _matrix(chunk_vectors, "chunk vectors")
    question_matrix = _matrix(question_vectors, "question vectors")
    if chunk_matrix.shape[0] != len(chunks_by_id):
        raise ValueError("chunk vector row count must match chunks")
    if question_matrix.shape[0] != len(questions_by_id):
        raise ValueError("question vector row count must match questions")
    if chunk_matrix.shape[1] != question_matrix.shape[1]:
        raise ValueError("chunk and question vector dimensions must match")

    chunk_index = {
        chunk.chunk_id: index for index, chunk in enumerate(chunks)
    }
    question_index = {
        question.question_id: index
        for index, question in enumerate(questions)
    }
    canonical_chunks = tuple(
        chunks_by_id[chunk_id] for chunk_id in sorted(chunks_by_id)
    )
    canonical_chunk_matrix = np.asarray(
        [chunk_matrix[chunk_index[chunk.chunk_id]] for chunk in canonical_chunks],
        dtype=np.float64,
    )
    jobs: list[ExperimentJob] = []
    for question_id in sorted(questions_by_id):
        question = questions_by_id[question_id]
        retrieved = exact_cosine_top_k(
            canonical_chunks,
            canonical_chunk_matrix,
            question_matrix[question_index[question_id]],
            top_k=top_k,
        )
        jobs.extend(
            build_clean_jobs(
                question=question,
                ordered_chunks=retrieved,
                defense_ids=defense_ids,
                config_hash=config_hash,
            )
        )
    return tuple(
        sorted(
            jobs,
            key=lambda job: sha256_text(f"{shuffle_seed}:{job.job_id}"),
        )
    )


def build_e2e_manifest(
    chunks: Sequence[Chunk],
    questions: Sequence[ResearchQuestion],
    attacks: Sequence[AttackCase],
    *,
    selection: E2ESelectionSummary,
    chunk_vectors: object,
    question_vectors: object,
    malicious_attack_ids: Sequence[str],
    malicious_vectors: object,
    top_k: int,
    defense_ids: tuple[DefenseId, ...],
    config_hash: str,
    shuffle_seed: int,
) -> tuple[ExperimentJob, ...]:
    if selection != select_main_e2e_contexts(questions, attacks):
        raise ValueError("E2E selection does not match the frozen policy")
    if not config_hash.strip():
        raise ValueError("E2E manifest config hash must be nonblank")
    chunks_by_id = _unique_by_id(
        chunks,
        identity=lambda chunk: chunk.chunk_id,
        label="chunk",
    )
    questions_by_id = _unique_by_id(
        questions,
        identity=lambda question: question.question_id,
        label="question",
    )
    attacks_by_id = _unique_by_id(
        attacks,
        identity=lambda attack: attack.attack_id,
        label="attack",
    )
    chunk_matrix = _matrix(chunk_vectors, "chunk vectors")
    question_matrix = _matrix(question_vectors, "question vectors")
    malicious_matrix = _matrix(malicious_vectors, "malicious vectors")
    if chunk_matrix.shape[0] != len(chunks):
        raise ValueError("chunk vector row count must match chunks")
    if question_matrix.shape[0] != len(questions):
        raise ValueError("question vector row count must match questions")
    if malicious_matrix.shape[0] != len(malicious_attack_ids):
        raise ValueError(
            "malicious vector row count must match malicious attack IDs"
        )
    if (
        chunk_matrix.shape[1] != question_matrix.shape[1]
        or chunk_matrix.shape[1] != malicious_matrix.shape[1]
    ):
        raise ValueError("E2E vector dimensions must match")
    if (
        len(set(malicious_attack_ids)) != len(malicious_attack_ids)
        or any(not attack_id.strip() for attack_id in malicious_attack_ids)
    ):
        raise ValueError("malicious attack IDs must be unique and nonblank")
    malicious_index = {
        attack_id: index
        for index, attack_id in enumerate(malicious_attack_ids)
    }
    missing_vectors = sorted(
        set(selection.selected_attack_ids) - set(malicious_index)
    )
    if missing_vectors:
        raise ValueError(
            "missing malicious vectors for attacks: "
            + ",".join(missing_vectors)
        )

    chunk_index = {
        chunk.chunk_id: index for index, chunk in enumerate(chunks)
    }
    question_index = {
        question.question_id: index
        for index, question in enumerate(questions)
    }
    canonical_chunks = tuple(
        chunks_by_id[chunk_id] for chunk_id in sorted(chunks_by_id)
    )
    canonical_chunk_matrix = np.asarray(
        [chunk_matrix[chunk_index[chunk.chunk_id]] for chunk in canonical_chunks],
        dtype=np.float64,
    )
    jobs: list[ExperimentJob] = []
    for attack_id in selection.selected_attack_ids:
        attack = attacks_by_id[attack_id]
        question = questions_by_id[attack.base_context_id]
        gold_chunk_id = question.gold_chunk_ids[0]
        query_vector = question_matrix[
            question_index[question.question_id]
        ]
        malicious_vector = malicious_matrix[malicious_index[attack_id]]
        condition_jobs = build_e2e_jobs(
            question=question,
            base_context_id=attack.base_context_id,
            corpus_chunks=canonical_chunks,
            corpus_matrix=canonical_chunk_matrix,
            query_vector=query_vector,
            malicious_vector=malicious_vector,
            gold_chunk_id=gold_chunk_id,
            attack=attack,
            top_k=top_k,
            defense_ids=defense_ids,
            config_hash=config_hash,
        )
        verify_e2e_job_from_sources(
            condition_jobs[0],
            corpus_chunks=canonical_chunks,
            corpus_matrix=canonical_chunk_matrix,
            query_vector=query_vector,
            malicious_vector=malicious_vector,
        )
        jobs.extend(condition_jobs)
    return tuple(
        sorted(
            jobs,
            key=lambda job: sha256_text(f"{shuffle_seed}:{job.job_id}"),
        )
    )


def write_generation_manifest(
    jobs: Sequence[ExperimentJob],
    *,
    track: Literal["forced", "e2e", "clean"],
    manifest_path: Path,
    summary_path: Path,
    shuffle_seed: int,
) -> GenerationManifestSummary:
    if manifest_path.resolve(strict=False) == summary_path.resolve(strict=False):
        raise ValueError("manifest and summary paths must differ")
    if not jobs:
        raise ValueError("generation manifest cannot be empty")
    if any(job.track != track for job in jobs):
        raise ValueError("generation manifest contains an unexpected track")
    job_ids = tuple(job.job_id for job in jobs)
    if len(set(job_ids)) != len(job_ids):
        raise ValueError("generation manifest job IDs must be unique")
    reconstructed = tuple(
        reconstruct_job_from_research_inputs(job) for job in jobs
    )
    if reconstructed != tuple(jobs):
        raise ValueError("generation manifest reconstruction mismatch")

    config_hashes = {job.config_hash for job in jobs}
    if len(config_hashes) != 1:
        raise ValueError("generation manifest jobs must share one config hash")
    grouped: dict[str, list[ExperimentJob]] = {}
    for job in jobs:
        grouped.setdefault(job.research_input_hash, []).append(job)
    defense_sets = {
        frozenset(job.defense_id for job in group)
        for group in grouped.values()
    }
    if len(defense_sets) != 1:
        raise ValueError(
            "generation conditions must share the same complete defense set"
        )
    defense_set = next(iter(defense_sets))
    if any(
        len(group) != len(defense_set)
        or len({job.defense_id for job in group}) != len(group)
        for group in grouped.values()
    ):
        raise ValueError(
            "generation conditions must share the same complete defense set"
        )
    defense_ids = tuple(
        defense_id for defense_id in DefenseId if defense_id in defense_set
    )
    contents = "".join(f"{canonical_json(job)}\n" for job in jobs)
    _atomic_write_text(manifest_path, contents)
    summary = GenerationManifestSummary(
        track=track,
        job_count=len(jobs),
        condition_count=len(grouped),
        defense_ids=defense_ids,
        config_sha256=next(iter(config_hashes)),
        shuffle_seed=shuffle_seed,
        manifest_sha256=sha256_text(contents),
    )
    _atomic_write_text(summary_path, f"{canonical_json(summary)}\n")
    return summary


def build_detector_manifest(
    cases: Sequence[DetectorEvaluationCase],
    *,
    config_hash: str,
    detector_model_id: str,
    detector_model_revision: str,
    scoring_policy_sha256: str,
    threshold: float,
    shuffle_seed: int,
) -> tuple[DetectorManifestEntry, ...]:
    if any(
        case.split is not Split.TEST
        or case.review_status is not DetectorCaseReviewStatus.RESEARCHER_APPROVED
        for case in cases
    ):
        raise ValueError("detector manifest requires approved test cases")
    _unique_by_id(
        cases,
        identity=lambda case: case.case_id,
        label="detector case",
    )
    _unique_by_id(
        cases,
        identity=lambda case: case.chunk.chunk_id,
        label="detector chunk",
    )
    entries: list[DetectorManifestEntry] = []
    for case in sorted(cases, key=lambda item: item.case_id):
        payload = cast(
            JsonValue,
            {
                "schema_version": "1.0.0",
                "case": case.model_dump(mode="json"),
                "case_commitment": canonical_sha256(case),
                "config_sha256": config_hash,
                "detector_model_id": detector_model_id,
                "detector_model_revision": detector_model_revision,
                "scoring_policy_sha256": scoring_policy_sha256,
                "threshold": threshold,
            },
        )
        entries.append(
            DetectorManifestEntry(
                job_id=stable_id("DET_JOB", payload),
                case=case,
                case_commitment=canonical_sha256(case),
                config_sha256=config_hash,
                detector_model_id=detector_model_id,
                detector_model_revision=detector_model_revision,
                scoring_policy_sha256=scoring_policy_sha256,
                threshold=threshold,
            )
        )
    return tuple(
        sorted(
            entries,
            key=lambda entry: sha256_text(
                f"{shuffle_seed}:{entry.job_id}"
            ),
        )
    )


def write_detector_manifest(
    entries: Sequence[DetectorManifestEntry],
    *,
    manifest_path: Path,
    summary_path: Path,
    shuffle_seed: int,
) -> DetectorManifestSummary:
    if manifest_path.resolve(strict=False) == summary_path.resolve(strict=False):
        raise ValueError("detector manifest and summary paths must differ")
    if not entries:
        raise ValueError("detector manifest cannot be empty")
    job_ids = tuple(entry.job_id for entry in entries)
    case_ids = tuple(entry.case.case_id for entry in entries)
    if len(set(job_ids)) != len(job_ids):
        raise ValueError("detector manifest job IDs must be unique")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("detector manifest case IDs must be unique")
    bindings = {
        (
            entry.config_sha256,
            entry.detector_model_id,
            entry.detector_model_revision,
            entry.scoring_policy_sha256,
            entry.threshold,
        )
        for entry in entries
    }
    if len(bindings) != 1:
        raise ValueError("detector manifest entries must share frozen bindings")
    (
        config_hash,
        detector_model_id,
        detector_model_revision,
        scoring_policy_hash,
        threshold,
    ) = next(iter(bindings))
    class_counts = Counter(entry.case.case_class for entry in entries)
    contents = "".join(f"{canonical_json(entry)}\n" for entry in entries)
    _atomic_write_text(manifest_path, contents)
    summary = DetectorManifestSummary(
        job_count=len(entries),
        malicious_count=class_counts[DetectorCaseClass.MALICIOUS],
        hard_negative_count=class_counts[DetectorCaseClass.HARD_NEGATIVE],
        ordinary_benign_count=class_counts[
            DetectorCaseClass.ORDINARY_BENIGN
        ],
        config_sha256=config_hash,
        detector_model_id=detector_model_id,
        detector_model_revision=detector_model_revision,
        scoring_policy_sha256=scoring_policy_hash,
        threshold=threshold,
        shuffle_seed=shuffle_seed,
        manifest_sha256=sha256_text(contents),
    )
    _atomic_write_text(summary_path, f"{canonical_json(summary)}\n")
    return summary


def _factorial_cell_key(attack: AttackCase) -> str:
    return _factorial_cell_key_from_values(
        attack.goal,
        attack.language,
        attack.form,
        attack.position,
    )


def _factorial_cell_key_from_values(
    goal: AttackGoal,
    language: Language,
    form: AttackForm,
    position: AttackPosition,
) -> str:
    return "|".join(
        (goal.value, language.value, form.value, position.value)
    )


def _enum_counts(
    attacks: Sequence[AttackCase],
    field_name: str,
    values: type[AttackGoal]
    | type[Language]
    | type[AttackForm]
    | type[AttackPosition],
) -> dict[str, int]:
    counts = Counter(getattr(attack, field_name) for attack in attacks)
    return {value.value: counts[value] for value in values}


def _unique_by_id(
    records: Sequence[_T],
    *,
    identity: Callable[[_T], str],
    label: str,
) -> dict[str, _T]:
    indexed: dict[str, _T] = {}
    for record in records:
        raw_identifier = identity(record)
        if not isinstance(raw_identifier, str) or not raw_identifier.strip():
            raise ValueError(f"{label} IDs must be nonblank")
        if raw_identifier in indexed:
            raise ValueError(f"duplicate {label} ID: {raw_identifier}")
        indexed[raw_identifier] = record
    if not indexed:
        raise ValueError(f"{label} records must be non-empty")
    return indexed


def _matrix(value: object, label: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if matrix.ndim != 2:
        raise ValueError(f"{label} must be 2-D")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{label} must be finite")
    return matrix


def _atomic_write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(raw_path)
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
