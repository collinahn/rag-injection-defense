import os
import re
import tempfile
from collections import Counter
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path

import numpy as np
from pydantic import Field

from thesis_rag.canonical import canonical_json, sha256_text
from thesis_rag.domain import (
    AttackCase,
    AttackGoal,
    Chunk,
    DefenseId,
    ExperimentJob,
    ForcedResearchInputs,
    FrozenRecord,
    Sha256Hex,
)
from thesis_rag.manifests import (
    build_forced_jobs,
    reconstruct_job_from_research_inputs,
)
from thesis_rag.qa import ResearchQuestion
from thesis_rag.retrieval import exact_cosine_top_k

_CREDIT_VALUE_PATTERN = re.compile(
    r"(?<!\d)(?P<value>\d+(?:\.\d+)?)\s*학점"
)


class ForcedManifestSummary(FrozenRecord):
    schema_version: str
    track: str
    job_count: int = Field(strict=True, ge=1)
    attack_count: int = Field(strict=True, ge=1)
    defense_ids: tuple[DefenseId, ...]
    malicious_rank_counts: tuple[int, ...]
    config_sha256: Sha256Hex
    shuffle_seed: int
    manifest_sha256: Sha256Hex
    gold_context_fallback_count: int = Field(default=0, strict=True, ge=0)


def build_forced_manifest(
    chunks: Sequence[Chunk],
    questions: Sequence[ResearchQuestion],
    attacks: Sequence[AttackCase],
    *,
    chunk_vectors: object,
    question_vectors: object,
    top_k: int,
    defense_ids: tuple[DefenseId, ...],
    config_hash: str,
    shuffle_seed: int,
    ensure_gold_context: bool = False,
) -> tuple[ExperimentJob, ...]:
    if not config_hash.strip():
        raise ValueError("forced manifest config hash must be nonblank")
    if not attacks:
        raise ValueError("forced manifest requires attacks")

    chunk_ids = tuple(chunk.chunk_id for chunk in chunks)
    question_ids = tuple(question.question_id for question in questions)
    attack_ids = tuple(attack.attack_id for attack in attacks)
    _require_unique(chunk_ids, "chunk")
    _require_unique(question_ids, "question")
    _require_unique(attack_ids, "attack")

    chunk_matrix = _matrix(chunk_vectors, "chunk vectors")
    question_matrix = _matrix(question_vectors, "question vectors")
    if chunk_matrix.shape[0] != len(chunks):
        raise ValueError("chunk vector row count must match chunks")
    if question_matrix.shape[0] != len(questions):
        raise ValueError("question vector row count must match questions")
    if chunk_matrix.shape[1] != question_matrix.shape[1]:
        raise ValueError("chunk and question vector dimensions must match")

    questions_by_id = {
        question.question_id: (index, question)
        for index, question in enumerate(questions)
    }
    jobs: list[ExperimentJob] = []
    ordered_attacks = tuple(sorted(attacks, key=lambda attack: attack.attack_id))
    for attack_index, attack in enumerate(ordered_attacks):
        try:
            question_index, question = questions_by_id[attack.base_context_id]
        except KeyError as error:
            raise ValueError(
                f"unknown forced manifest base context: {attack.base_context_id}"
            ) from error
        retrieved = exact_cosine_top_k(
            chunks,
            chunk_matrix,
            question_matrix[question_index],
            top_k=top_k,
        )
        retrieved_ids = {item.chunk.chunk_id for item in retrieved}
        gold_chunk_id = next(
            (
                chunk_id
                for chunk_id in question.gold_chunk_ids
                if chunk_id in retrieved_ids
            ),
            None,
        )
        gold_context_fallback = False
        if gold_chunk_id is None:
            if not ensure_gold_context:
                raise ValueError(
                    f"forced manifest retrieval missed every gold chunk: "
                    f"{question.question_id}"
                )
            full_ranking = exact_cosine_top_k(
                chunks,
                chunk_matrix,
                question_matrix[question_index],
                top_k=len(chunks),
            )
            fallback_gold = next(
                (
                    item
                    for item in full_ranking
                    if item.chunk.chunk_id in set(question.gold_chunk_ids)
                ),
                None,
            )
            if fallback_gold is None:
                raise ValueError(
                    f"forced manifest cannot locate a declared gold chunk: "
                    f"{question.question_id}"
                )
            retrieved = tuple(
                item.model_copy(update={"rank": rank})
                for rank, item in enumerate(
                    (*retrieved[:-1], fallback_gold),
                    start=1,
                )
            )
            gold_chunk_id = fallback_gold.chunk.chunk_id
            gold_context_fallback = True
        _reject_false_claim_credit_collisions(
            attack,
            tuple(item.chunk for item in retrieved),
        )
        jobs.extend(
            build_forced_jobs(
                question=question,
                base_context_id=attack.base_context_id,
                ordered_benign_chunks=retrieved,
                gold_chunk_id=gold_chunk_id,
                attack=attack,
                malicious_rank=(attack_index % top_k) + 1,
                defense_ids=defense_ids,
                config_hash=config_hash,
                gold_context_fallback=gold_context_fallback,
            )
        )

    return tuple(
        sorted(
            jobs,
            key=lambda job: sha256_text(f"{shuffle_seed}:{job.job_id}"),
        )
    )


def _reject_false_claim_credit_collisions(
    attack: AttackCase,
    benign_chunks: Sequence[Chunk],
) -> None:
    if attack.goal is not AttackGoal.FALSE_RULE:
        return
    claim = attack.target_false_claim
    if claim is None:
        raise ValueError(f"{attack.attack_id} G2 attack requires a false claim")
    claim_values = _credit_values(claim)
    if not claim_values:
        return

    collisions: dict[str, list[str]] = {}
    for chunk in benign_chunks:
        chunk_values = _credit_values(chunk.text)
        for value in sorted(claim_values & chunk_values):
            collisions.setdefault(
                _format_credit_value(value),
                [],
            ).append(chunk.chunk_id)
    if collisions:
        details = ", ".join(
            f"{value} in {','.join(chunk_ids)}"
            for value, chunk_ids in sorted(collisions.items())
        )
        raise ValueError(
            f"{attack.attack_id} false-claim credit value collides with benign "
            f"retrieval context: {details}"
        )


def _credit_values(text: str) -> set[Decimal]:
    return {
        Decimal(match.group("value"))
        for match in _CREDIT_VALUE_PATTERN.finditer(text)
    }


def _format_credit_value(value: Decimal) -> str:
    return f"{format(value.normalize(), 'f')}학점"


def write_forced_manifest(
    jobs: Sequence[ExperimentJob],
    *,
    manifest_path: Path,
    summary_path: Path,
    shuffle_seed: int,
) -> ForcedManifestSummary:
    if manifest_path.resolve(strict=False) == summary_path.resolve(strict=False):
        raise ValueError("forced manifest and summary paths must differ")
    if not jobs:
        raise ValueError("forced manifest cannot be empty")

    job_ids = tuple(job.job_id for job in jobs)
    _require_unique(job_ids, "job")
    if any(job.track != "forced" for job in jobs):
        raise ValueError("forced manifest may only contain forced jobs")
    reconstructed = tuple(reconstruct_job_from_research_inputs(job) for job in jobs)
    if reconstructed != tuple(jobs):
        raise ValueError("forced manifest reconstruction mismatch")

    config_hashes = {job.config_hash for job in jobs}
    if len(config_hashes) != 1:
        raise ValueError("forced manifest jobs must share one config hash")
    config_hash = next(iter(config_hashes))

    grouped: dict[str, list[ExperimentJob]] = {}
    for job in jobs:
        if job.attack_id is None:
            raise ValueError("forced manifest job requires attack_id")
        grouped.setdefault(job.attack_id, []).append(job)
    observed_defense_sets = {
        frozenset(job.defense_id for job in group)
        for group in grouped.values()
    }
    if len(observed_defense_sets) != 1:
        raise ValueError("each forced attack must contain the same defenses")
    defense_set = next(iter(observed_defense_sets))
    if any(len(group) != len(defense_set) for group in grouped.values()):
        raise ValueError("forced attack contains duplicate defense jobs")
    defense_ids = tuple(
        defense_id for defense_id in DefenseId if defense_id in defense_set
    )

    attack_representatives = tuple(group[0] for group in grouped.values())
    maximum_rank = max(len(job.ordered_chunks) for job in attack_representatives)
    rank_counts = Counter(
        job.malicious_chunk_rank for job in attack_representatives
    )
    if None in rank_counts:
        raise ValueError("forced manifest requires malicious chunk exposure")

    contents = "".join(f"{canonical_json(job)}\n" for job in jobs)
    _atomic_write_text(manifest_path, contents)
    summary = ForcedManifestSummary(
        schema_version="1.0.0",
        track="forced",
        job_count=len(jobs),
        attack_count=len(grouped),
        defense_ids=defense_ids,
        malicious_rank_counts=tuple(
            rank_counts.get(rank, 0) for rank in range(1, maximum_rank + 1)
        ),
        config_sha256=config_hash,
        shuffle_seed=shuffle_seed,
        manifest_sha256=sha256_text(contents),
        gold_context_fallback_count=sum(
            isinstance(job.research_inputs, ForcedResearchInputs)
            and job.research_inputs.gold_context_fallback
            for job in attack_representatives
        ),
    )
    _atomic_write_text(summary_path, f"{canonical_json(summary)}\n")
    return summary


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


def _require_unique(values: Sequence[str], label: str) -> None:
    if not values:
        raise ValueError(f"forced manifest requires {label}s")
    if len(set(values)) != len(values):
        raise ValueError(f"forced manifest {label} IDs must be unique")


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
