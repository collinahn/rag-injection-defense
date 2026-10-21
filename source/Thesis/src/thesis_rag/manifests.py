import hashlib
from collections.abc import Sequence
from typing import cast

import numpy as np

from thesis_rag.attacks import insert_payload
from thesis_rag.canonical import JsonValue, canonical_json, canonical_sha256
from thesis_rag.domain import (
    AttackCase,
    Chunk,
    DefenseId,
    DirectResearchInputs,
    E2EResearchInputs,
    ExperimentJob,
    ForcedResearchInputs,
    Question,
    ResearchInputs,
    RetrievedChunk,
    expected_experiment_job_id,
)
from thesis_rag.retrieval import exact_cosine_top_k


def _validate_shared_contracts(
    *,
    question: Question,
    base_context_id: str,
    source_chunks: Sequence[Chunk],
    gold_chunk_id: str,
    attack: AttackCase,
    defense_ids: tuple[DefenseId, ...],
) -> None:
    if not defense_ids:
        raise ValueError("defense_ids must be non-empty")
    if len(set(defense_ids)) != len(defense_ids):
        raise ValueError("defense_ids must be unique")
    if not question.answerable:
        raise ValueError("question must be answerable for attack manifests")
    if gold_chunk_id not in question.gold_chunk_ids:
        raise ValueError("gold_chunk_id must be one of question.gold_chunk_ids")
    if question.split != attack.split:
        raise ValueError("question and attack must have the same split")
    if attack.base_context_id != base_context_id:
        raise ValueError("attack base_context_id must match builder base_context_id")

    chunk_ids = tuple(chunk.chunk_id for chunk in source_chunks)
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("source chunk IDs must be unique")
    if chunk_ids.count(gold_chunk_id) != 1:
        raise ValueError("gold chunk must occur exactly once")


def _build_jobs(
    *,
    track: str,
    question: Question,
    base_context_id: str,
    attack: AttackCase,
    defense_ids: tuple[DefenseId, ...],
    config_hash: str,
    ordered_chunks: tuple[RetrievedChunk, ...],
    malicious_chunk_id: str,
    clean_source_chunk: Chunk,
    research_inputs: ResearchInputs,
) -> tuple[ExperimentJob, ...]:
    ordered_chunk_ids = tuple(item.chunk.chunk_id for item in ordered_chunks)
    malicious_chunk_rank = next(
        (item.rank for item in ordered_chunks if item.chunk.chunk_id == malicious_chunk_id), None
    )
    clean_source_commitment = canonical_sha256(clean_source_chunk)
    attack_commitment = canonical_sha256(attack)
    research_input_hash = canonical_sha256(research_inputs)
    first_defense_id = defense_ids[0]
    first_job = ExperimentJob(
        job_id="JOB_PENDING",
        track=track,  # type: ignore[arg-type]
        question_id=question.question_id,
        base_context_id=base_context_id,
        attack_id=attack.attack_id,
        defense_id=first_defense_id,
        question=question.question,
        ordered_chunks=ordered_chunks,
        ordered_chunk_ids=ordered_chunk_ids,
        malicious_chunk_rank=malicious_chunk_rank,
        attack=attack,
        clean_source_chunk=clean_source_chunk,
        clean_source_commitment=clean_source_commitment,
        attack_commitment=attack_commitment,
        research_inputs=research_inputs,
        research_input_hash=research_input_hash,
        config_hash=config_hash,
    )
    first_job = _bind_job_id(first_job)
    return (first_job,) + tuple(
        _bind_job_id(first_job.model_copy(update={"defense_id": defense_id}))
        for defense_id in defense_ids[1:]
    )


def _bind_job_id(job: ExperimentJob) -> ExperimentJob:
    return job.model_copy(update={"job_id": expected_experiment_job_id(job)})


def build_forced_jobs(
    *,
    question: Question,
    base_context_id: str,
    ordered_benign_chunks: tuple[RetrievedChunk, ...],
    gold_chunk_id: str,
    attack: AttackCase,
    malicious_rank: int,
    defense_ids: tuple[DefenseId, ...],
    config_hash: str,
    gold_context_fallback: bool = False,
) -> tuple[ExperimentJob, ...]:
    _validate_shared_contracts(
        question=question,
        base_context_id=base_context_id,
        source_chunks=tuple(item.chunk for item in ordered_benign_chunks),
        gold_chunk_id=gold_chunk_id,
        attack=attack,
        defense_ids=defense_ids,
    )
    expected_ranks = tuple(range(1, len(ordered_benign_chunks) + 1))
    if tuple(item.rank for item in ordered_benign_chunks) != expected_ranks:
        raise ValueError("ordered_benign_chunks ranks must be contiguous and 1-based")
    if not 1 <= malicious_rank <= len(ordered_benign_chunks):
        raise ValueError("malicious_rank must satisfy 1 <= malicious_rank <= context size")

    gold_retrieved = next(
        item for item in ordered_benign_chunks if item.chunk.chunk_id == gold_chunk_id
    )
    malicious_chunk = insert_payload(gold_retrieved.chunk, attack)
    clean_removed = tuple(
        item for item in ordered_benign_chunks if item.chunk.chunk_id != gold_chunk_id
    )
    replacement = RetrievedChunk(chunk=malicious_chunk, score=gold_retrieved.score, rank=1)
    forced_context = (
        clean_removed[: malicious_rank - 1]
        + (replacement,)
        + clean_removed[malicious_rank - 1 :]
    )
    ordered_chunks = tuple(
        item.model_copy(update={"rank": rank}) for rank, item in enumerate(forced_context, start=1)
    )
    research_inputs = ForcedResearchInputs(
        question=question,
        base_context_id=base_context_id,
        ordered_benign_chunks=ordered_benign_chunks,
        gold_chunk_id=gold_chunk_id,
        attack=attack,
        malicious_rank=malicious_rank,
        gold_context_fallback=gold_context_fallback,
        config_hash=config_hash,
    )
    return _build_jobs(
        track="forced",
        question=question,
        base_context_id=base_context_id,
        attack=attack,
        defense_ids=defense_ids,
        config_hash=config_hash,
        ordered_chunks=ordered_chunks,
        malicious_chunk_id=malicious_chunk.chunk_id,
        clean_source_chunk=gold_retrieved.chunk,
        research_inputs=research_inputs,
    )


def build_clean_jobs(
    *,
    question: Question,
    ordered_chunks: tuple[RetrievedChunk, ...],
    defense_ids: tuple[DefenseId, ...],
    config_hash: str,
) -> tuple[ExperimentJob, ...]:
    if not defense_ids:
        raise ValueError("defense_ids must be non-empty")
    if len(set(defense_ids)) != len(defense_ids):
        raise ValueError("defense_ids must be unique")
    if not config_hash.strip():
        raise ValueError("clean job config hash must be nonblank")
    expected_ranks = tuple(range(1, len(ordered_chunks) + 1))
    if tuple(item.rank for item in ordered_chunks) != expected_ranks:
        raise ValueError("ordered_chunks ranks must be contiguous and 1-based")
    chunk_ids = tuple(item.chunk.chunk_id for item in ordered_chunks)
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("clean ordered chunk IDs must be unique")

    research_inputs = DirectResearchInputs(
        track="clean",
        question_id=question.question_id,
        base_context_id=None,
        attack_id=None,
        question=question.question,
        ordered_chunks=ordered_chunks,
        malicious_chunk_rank=None,
        attack=None,
        clean_source_chunk=None,
        config_hash=config_hash,
    )
    research_input_hash = canonical_sha256(research_inputs)
    first = _bind_job_id(
        ExperimentJob(
            job_id="JOB_PENDING",
            track="clean",
            question_id=question.question_id,
            base_context_id=None,
            attack_id=None,
            defense_id=defense_ids[0],
            question=question.question,
            ordered_chunks=ordered_chunks,
            ordered_chunk_ids=chunk_ids,
            malicious_chunk_rank=None,
            attack=None,
            clean_source_chunk=None,
            clean_source_commitment=None,
            attack_commitment=None,
            research_inputs=research_inputs,
            research_input_hash=research_input_hash,
            config_hash=config_hash,
        )
    )
    return (first,) + tuple(
        _bind_job_id(first.model_copy(update={"defense_id": defense_id}))
        for defense_id in defense_ids[1:]
    )


def _as_numeric_array(value: object, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric values") from error
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array


def _numeric_array_commitment(array: np.ndarray) -> str:
    little_endian = np.ascontiguousarray(array, dtype="<f8")
    header = canonical_json(
        cast(
            JsonValue,
            {
                "dtype": "float64-le",
                "shape": list(little_endian.shape),
            },
        )
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(little_endian.tobytes(order="C"))
    return digest.hexdigest()


def _chunk_sequence_commitment(chunks: Sequence[Chunk]) -> str:
    return canonical_sha256(
        cast(
            JsonValue,
            [chunk.model_dump(mode="json") for chunk in chunks],
        )
    )


def build_e2e_jobs(
    *,
    question: Question,
    base_context_id: str,
    corpus_chunks: tuple[Chunk, ...],
    corpus_matrix: object,
    query_vector: object,
    malicious_vector: object,
    gold_chunk_id: str,
    attack: AttackCase,
    top_k: int,
    defense_ids: tuple[DefenseId, ...],
    config_hash: str,
) -> tuple[ExperimentJob, ...]:
    _validate_shared_contracts(
        question=question,
        base_context_id=base_context_id,
        source_chunks=corpus_chunks,
        gold_chunk_id=gold_chunk_id,
        attack=attack,
        defense_ids=defense_ids,
    )
    matrix = _as_numeric_array(corpus_matrix, name="corpus_matrix")
    query = _as_numeric_array(query_vector, name="query_vector")
    malicious = _as_numeric_array(malicious_vector, name="malicious_vector")
    if matrix.ndim != 2:
        raise ValueError("corpus_matrix must be 2-D")
    if matrix.shape[0] != len(corpus_chunks):
        raise ValueError("corpus_matrix row count must match corpus_chunks")
    if query.ndim != 1:
        raise ValueError("query_vector must be 1-D")
    if matrix.shape[1] != query.shape[0]:
        raise ValueError("corpus_matrix and query_vector dimensions must match")
    if malicious.ndim != 1:
        raise ValueError("malicious_vector must be 1-D")
    if matrix.shape[1] != malicious.shape[0]:
        raise ValueError("malicious_vector dimensions must match corpus_matrix rows")
    if not 1 <= top_k <= len(corpus_chunks):
        raise ValueError("top_k must satisfy 1 <= top_k <= corpus size")

    gold_index = next(
        index for index, chunk in enumerate(corpus_chunks) if chunk.chunk_id == gold_chunk_id
    )
    malicious_chunk = insert_payload(corpus_chunks[gold_index], attack)
    overlay_chunks = (
        corpus_chunks[:gold_index] + (malicious_chunk,) + corpus_chunks[gold_index + 1 :]
    )
    overlay_matrix = matrix.copy()
    overlay_matrix[gold_index] = malicious
    ordered_chunks = exact_cosine_top_k(overlay_chunks, overlay_matrix, query, top_k=top_k)
    research_inputs = E2EResearchInputs(
        question=question,
        base_context_id=base_context_id,
        ordered_chunks=ordered_chunks,
        gold_chunk_id=gold_chunk_id,
        attack=attack,
        clean_source_chunk=corpus_chunks[gold_index],
        top_k=top_k,
        corpus_chunks_commitment=_chunk_sequence_commitment(corpus_chunks),
        corpus_matrix_commitment=_numeric_array_commitment(matrix),
        query_vector_commitment=_numeric_array_commitment(query),
        malicious_vector_commitment=_numeric_array_commitment(malicious),
        config_hash=config_hash,
    )
    return _build_jobs(
        track="e2e",
        question=question,
        base_context_id=base_context_id,
        attack=attack,
        defense_ids=defense_ids,
        config_hash=config_hash,
        ordered_chunks=ordered_chunks,
        malicious_chunk_id=malicious_chunk.chunk_id,
        clean_source_chunk=corpus_chunks[gold_index],
        research_inputs=research_inputs,
    )


def reconstruct_job_from_research_inputs(job: ExperimentJob) -> ExperimentJob:
    research_inputs = job.research_inputs
    if isinstance(research_inputs, ForcedResearchInputs):
        reconstructed = build_forced_jobs(
            question=research_inputs.question,
            base_context_id=research_inputs.base_context_id,
            ordered_benign_chunks=research_inputs.ordered_benign_chunks,
            gold_chunk_id=research_inputs.gold_chunk_id,
            attack=research_inputs.attack,
            malicious_rank=research_inputs.malicious_rank,
            defense_ids=(job.defense_id,),
            config_hash=research_inputs.config_hash,
            gold_context_fallback=research_inputs.gold_context_fallback,
        )[0]
    elif isinstance(research_inputs, E2EResearchInputs):
        malicious_chunk = insert_payload(
            research_inputs.clean_source_chunk,
            research_inputs.attack,
        )
        reconstructed = _build_jobs(
            track="e2e",
            question=research_inputs.question,
            base_context_id=research_inputs.base_context_id,
            attack=research_inputs.attack,
            defense_ids=(job.defense_id,),
            config_hash=research_inputs.config_hash,
            ordered_chunks=research_inputs.ordered_chunks,
            malicious_chunk_id=malicious_chunk.chunk_id,
            clean_source_chunk=research_inputs.clean_source_chunk,
            research_inputs=research_inputs,
        )[0]
    else:
        reconstructed = _reconstruct_direct_job(job, research_inputs)

    if reconstructed != job:
        raise ValueError("job must match the reconstructed research manifest")
    return reconstructed


def _reconstruct_direct_job(
    job: ExperimentJob,
    research_inputs: DirectResearchInputs,
) -> ExperimentJob:
    clean_source = research_inputs.clean_source_chunk
    attack = research_inputs.attack
    reconstructed = ExperimentJob(
        job_id="JOB_PENDING",
        track=research_inputs.track,
        question_id=research_inputs.question_id,
        base_context_id=research_inputs.base_context_id,
        attack_id=research_inputs.attack_id,
        defense_id=job.defense_id,
        question=research_inputs.question,
        ordered_chunks=research_inputs.ordered_chunks,
        ordered_chunk_ids=tuple(
            item.chunk.chunk_id for item in research_inputs.ordered_chunks
        ),
        malicious_chunk_rank=research_inputs.malicious_chunk_rank,
        attack=attack,
        clean_source_chunk=clean_source,
        clean_source_commitment=(
            None if clean_source is None else canonical_sha256(clean_source)
        ),
        attack_commitment=(
            None if attack is None else canonical_sha256(attack)
        ),
        research_inputs=research_inputs,
        research_input_hash=canonical_sha256(research_inputs),
        config_hash=research_inputs.config_hash,
    )
    return _bind_job_id(reconstructed)


def verify_e2e_job_from_sources(
    job: ExperimentJob,
    *,
    corpus_chunks: object,
    corpus_matrix: object,
    query_vector: object,
    malicious_vector: object,
) -> ExperimentJob:
    research_inputs = job.research_inputs
    if not isinstance(research_inputs, E2EResearchInputs):
        raise ValueError("external E2E verification requires E2E research inputs")
    if not isinstance(corpus_chunks, tuple) or not all(
        isinstance(chunk, Chunk) for chunk in corpus_chunks
    ):
        raise ValueError("E2E corpus chunks must be a tuple of Chunk records")
    typed_chunks = cast(tuple[Chunk, ...], corpus_chunks)
    if (
        _chunk_sequence_commitment(typed_chunks)
        != research_inputs.corpus_chunks_commitment
    ):
        raise ValueError("corpus chunks commitment mismatch")

    matrix = _as_numeric_array(corpus_matrix, name="corpus_matrix")
    query = _as_numeric_array(query_vector, name="query_vector")
    malicious = _as_numeric_array(malicious_vector, name="malicious_vector")
    for label, observed, expected in (
        (
            "corpus matrix",
            _numeric_array_commitment(matrix),
            research_inputs.corpus_matrix_commitment,
        ),
        (
            "query vector",
            _numeric_array_commitment(query),
            research_inputs.query_vector_commitment,
        ),
        (
            "malicious vector",
            _numeric_array_commitment(malicious),
            research_inputs.malicious_vector_commitment,
        ),
    ):
        if observed != expected:
            raise ValueError(f"{label} commitment mismatch")

    rebuilt = build_e2e_jobs(
        question=research_inputs.question,
        base_context_id=research_inputs.base_context_id,
        corpus_chunks=typed_chunks,
        corpus_matrix=matrix,
        query_vector=query,
        malicious_vector=malicious,
        gold_chunk_id=research_inputs.gold_chunk_id,
        attack=research_inputs.attack,
        top_k=research_inputs.top_k,
        defense_ids=(job.defense_id,),
        config_hash=research_inputs.config_hash,
    )[0]
    if rebuilt != job:
        raise ValueError("E2E job does not match committed external sources")
    return rebuilt
