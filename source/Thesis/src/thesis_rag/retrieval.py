from collections.abc import Sequence

import numpy as np
from pydantic import Field, StrictBool

from thesis_rag.canonical import JsonValue, canonical_sha256
from thesis_rag.domain import Chunk, FiniteFloat, FrozenRecord, RetrievedChunk
from thesis_rag.qa import ResearchQuestion


class RetrievalQuestionResult(FrozenRecord):
    question_id: str
    gold_chunk_ids: tuple[str, ...]
    retrieved_chunk_ids: tuple[str, ...]
    retrieved_gold_chunk_ids: tuple[str, ...]
    missing_gold_chunk_ids: tuple[str, ...]
    first_gold_rank: int | None = Field(default=None, strict=True, ge=1)
    hit: StrictBool
    all_gold_hit: StrictBool


class RetrievalEvaluationSummary(FrozenRecord):
    top_k: int = Field(strict=True, ge=1)
    answerable_question_count: int = Field(strict=True, ge=1)
    hit_count: int = Field(strict=True, ge=0)
    all_gold_hit_count: int = Field(strict=True, ge=0)
    recall_at_k: FiniteFloat
    all_gold_recall_at_k: FiniteFloat
    mean_reciprocal_rank: FiniteFloat
    ranking_sha256: str


def exact_cosine_top_k(
    chunks: Sequence[Chunk],
    matrix: object,
    query: object,
    *,
    top_k: int,
) -> tuple[RetrievedChunk, ...]:
    if not chunks:
        raise ValueError("chunks must be non-empty")
    if not 1 <= top_k <= len(chunks):
        raise ValueError("top_k must satisfy 1 <= top_k <= len(chunks)")

    try:
        matrix_array = np.asarray(matrix, dtype=np.float64)
        query_array = np.asarray(query, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("matrix and query must contain numeric values") from error

    if matrix_array.ndim != 2:
        raise ValueError("matrix must be 2-D")
    if query_array.ndim != 1:
        raise ValueError("query must be 1-D")
    if matrix_array.shape[0] != len(chunks):
        raise ValueError("matrix row count must match chunks")
    if matrix_array.shape[1] != query_array.shape[0]:
        raise ValueError("matrix and query dimensions must match")
    if not np.isfinite(matrix_array).all() or not np.isfinite(query_array).all():
        raise ValueError("matrix and query must contain finite values")

    row_scales = np.max(np.abs(matrix_array), axis=1, initial=0.0)
    query_scale = np.max(np.abs(query_array), initial=0.0)
    if np.any(row_scales == 0.0) or query_scale == 0.0:
        raise ValueError("zero-norm vector")

    scaled_matrix = matrix_array / row_scales[:, np.newaxis]
    scaled_query = query_array / query_scale
    normalized_matrix = scaled_matrix / np.linalg.norm(scaled_matrix, axis=1)[:, np.newaxis]
    normalized_query = scaled_query / np.linalg.norm(scaled_query)
    scores = normalized_matrix @ normalized_query
    order = sorted(
        range(len(chunks)),
        key=lambda index: (-float(scores[index]), chunks[index].chunk_id),
    )[:top_k]

    return tuple(
        RetrievedChunk(chunk=chunks[index], score=float(scores[index]), rank=rank)
        for rank, index in enumerate(order, start=1)
    )


def evaluate_retrieval(
    chunks: Sequence[Chunk],
    questions: Sequence[ResearchQuestion],
    chunk_vectors: object,
    question_vectors: object,
    *,
    top_k: int,
) -> tuple[RetrievalEvaluationSummary, tuple[RetrievalQuestionResult, ...]]:
    answerable_indexes = tuple(
        index for index, question in enumerate(questions) if question.answerable
    )
    if not answerable_indexes:
        raise ValueError("retrieval evaluation requires answerable questions")
    try:
        chunk_matrix = np.asarray(chunk_vectors, dtype=np.float64)
        query_matrix = np.asarray(question_vectors, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("retrieval vectors must be numeric") from error
    if chunk_matrix.shape[0] != len(chunks):
        raise ValueError("chunk vector row count must match chunks")
    if query_matrix.shape[0] != len(questions):
        raise ValueError("question vector row count must match questions")

    results: list[RetrievalQuestionResult] = []
    for index in answerable_indexes:
        question = questions[index]
        retrieved = exact_cosine_top_k(
            chunks,
            chunk_matrix,
            query_matrix[index],
            top_k=top_k,
        )
        retrieved_ids = tuple(item.chunk.chunk_id for item in retrieved)
        retrieved_id_set = set(retrieved_ids)
        retrieved_gold_ids = tuple(
            chunk_id
            for chunk_id in question.gold_chunk_ids
            if chunk_id in retrieved_id_set
        )
        missing_gold_ids = tuple(
            chunk_id
            for chunk_id in question.gold_chunk_ids
            if chunk_id not in retrieved_id_set
        )
        first_gold_rank = next(
            (
                rank
                for rank, chunk_id in enumerate(retrieved_ids, start=1)
                if chunk_id in set(question.gold_chunk_ids)
            ),
            None,
        )
        results.append(
            RetrievalQuestionResult(
                question_id=question.question_id,
                gold_chunk_ids=question.gold_chunk_ids,
                retrieved_chunk_ids=retrieved_ids,
                retrieved_gold_chunk_ids=retrieved_gold_ids,
                missing_gold_chunk_ids=missing_gold_ids,
                first_gold_rank=first_gold_rank,
                hit=bool(retrieved_gold_ids),
                all_gold_hit=not missing_gold_ids,
            )
        )
    frozen_results = tuple(results)
    count = len(frozen_results)
    hit_count = sum(result.hit for result in frozen_results)
    all_gold_hit_count = sum(result.all_gold_hit for result in frozen_results)
    reciprocal_rank_sum = sum(
        0.0 if result.first_gold_rank is None else 1.0 / result.first_gold_rank
        for result in frozen_results
    )
    ranking_value: JsonValue = [
        {
            "question_id": result.question_id,
            "retrieved_chunk_ids": list(result.retrieved_chunk_ids),
        }
        for result in frozen_results
    ]
    summary = RetrievalEvaluationSummary(
        top_k=top_k,
        answerable_question_count=count,
        hit_count=hit_count,
        all_gold_hit_count=all_gold_hit_count,
        recall_at_k=hit_count / count,
        all_gold_recall_at_k=all_gold_hit_count / count,
        mean_reciprocal_rank=reciprocal_rank_sum / count,
        ranking_sha256=canonical_sha256(ranking_value),
    )
    return summary, frozen_results
