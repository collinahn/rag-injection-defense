from re import escape

import numpy as np
import pytest

from thesis_rag.domain import Chunk
from thesis_rag.retrieval import exact_cosine_top_k


def _chunks(*chunk_ids: str) -> tuple[Chunk, ...]:
    return tuple(
        Chunk(chunk_id=chunk_id, document_id="DOC", text=chunk_id, language="ko")
        for chunk_id in chunk_ids
    )


def test_exact_cosine_ranks_by_descending_normalized_score() -> None:
    chunks = _chunks("CH_LOW", "CH_HIGH", "CH_MID")
    matrix = np.asarray([[0.0, 1.0], [4.0, 3.0], [1.0, 1.0]])

    results = exact_cosine_top_k(chunks, matrix, np.asarray([1.0, 0.0]), top_k=3)

    assert [item.chunk.chunk_id for item in results] == ["CH_HIGH", "CH_MID", "CH_LOW"]
    assert [item.rank for item in results] == [1, 2, 3]
    assert [item.score for item in results] == pytest.approx([0.8, 2**-0.5, 0.0])


def test_exact_cosine_uses_chunk_id_for_equal_scores() -> None:
    chunks = (
        Chunk(chunk_id="CH_B", document_id="DOC", text="B", language="ko"),
        Chunk(chunk_id="CH_A", document_id="DOC", text="A", language="ko"),
    )
    matrix = np.asarray([[1.0, 0.0], [1.0, 0.0]])

    results = exact_cosine_top_k(chunks, matrix, np.asarray([1.0, 0.0]), top_k=2)

    assert [item.chunk.chunk_id for item in results] == ["CH_A", "CH_B"]
    assert [item.rank for item in results] == [1, 2]


def test_exact_cosine_normalizes_rows_and_query() -> None:
    chunks = _chunks("CH_ONE", "CH_TWO")
    matrix = np.asarray([[10.0, 0.0], [0.0, 2.0]])

    results = exact_cosine_top_k(chunks, matrix, np.asarray([5.0, 0.0]), top_k=2)

    assert [item.score for item in results] == pytest.approx([1.0, 0.0])


def test_exact_cosine_handles_maximum_float64_components() -> None:
    chunks = _chunks("CH_NORMAL", "CH_EXTREME")
    maximum = np.finfo(np.float64).max
    matrix = np.asarray([[0.6, 0.8], [maximum, maximum]])

    results = exact_cosine_top_k(chunks, matrix, np.asarray([1.0, 0.0]), top_k=2)

    assert [item.chunk.chunk_id for item in results] == ["CH_EXTREME", "CH_NORMAL"]
    assert [item.score for item in results] == pytest.approx([2**-0.5, 0.6])


def test_exact_cosine_accepts_nonzero_subnormal_row_and_query() -> None:
    subnormal = np.nextafter(np.float64(0.0), np.float64(1.0))

    results = exact_cosine_top_k(
        _chunks("CH_SUBNORMAL"),
        np.asarray([[subnormal, subnormal]]),
        np.asarray([subnormal, 0.0]),
        top_k=1,
    )

    assert results[0].chunk.chunk_id == "CH_SUBNORMAL"
    assert results[0].score == pytest.approx(2**-0.5)


@pytest.mark.parametrize(
    ("matrix", "query", "message"),
    [
        (np.asarray([[0.0, 0.0]]), np.asarray([1.0, 0.0]), "zero-norm vector"),
        (np.asarray([[1.0, 0.0]]), np.asarray([0.0, 0.0]), "zero-norm vector"),
    ],
)
def test_exact_cosine_rejects_zero_norm_vectors(
    matrix: np.ndarray, query: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=f"^{escape(message)}$"):
        exact_cosine_top_k(_chunks("CH"), matrix, query, top_k=1)


@pytest.mark.parametrize(
    ("chunks", "matrix", "query", "top_k", "message"),
    [
        ((), np.empty((0, 2)), np.asarray([1.0, 0.0]), 1, "chunks must be non-empty"),
        (_chunks("CH"), np.asarray([1.0, 0.0]), np.asarray([1.0, 0.0]), 1, "matrix must be 2-D"),
        (_chunks("CH"), np.asarray([[1.0, 0.0]]), np.asarray([[1.0, 0.0]]), 1, "query must be 1-D"),
        (
            _chunks("CH_1", "CH_2"),
            np.asarray([[1.0, 0.0]]),
            np.asarray([1.0, 0.0]),
            1,
            "matrix row count must match chunks",
        ),
        (
            _chunks("CH"),
            np.asarray([[1.0, 0.0]]),
            np.asarray([1.0]),
            1,
            "matrix and query dimensions must match",
        ),
        (
            _chunks("CH"),
            np.asarray([[1.0, 0.0]]),
            np.asarray([1.0, 0.0]),
            0,
            "top_k must satisfy 1 <= top_k <= len(chunks)",
        ),
        (
            _chunks("CH"),
            np.asarray([[1.0, 0.0]]),
            np.asarray([1.0, 0.0]),
            2,
            "top_k must satisfy 1 <= top_k <= len(chunks)",
        ),
    ],
)
def test_exact_cosine_rejects_shape_and_range_mismatches(
    chunks: tuple[Chunk, ...],
    matrix: np.ndarray,
    query: np.ndarray,
    top_k: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=f"^{escape(message)}$"):
        exact_cosine_top_k(chunks, matrix, query, top_k=top_k)


@pytest.mark.parametrize(
    "matrix,query",
    [
        (np.asarray([[np.nan, 0.0]]), np.asarray([1.0, 0.0])),
        (np.asarray([[np.inf, 0.0]]), np.asarray([1.0, 0.0])),
        (np.asarray([[1.0, 0.0]]), np.asarray([1.0, np.nan])),
    ],
)
def test_exact_cosine_rejects_non_finite_values(matrix: np.ndarray, query: np.ndarray) -> None:
    with pytest.raises(ValueError, match="^matrix and query must contain finite values$"):
        exact_cosine_top_k(_chunks("CH"), matrix, query, top_k=1)


def test_exact_cosine_returns_float64_scores_in_an_immutable_tuple() -> None:
    results = exact_cosine_top_k(
        _chunks("CH"),
        np.asarray([[3, 4]], dtype=np.int32),
        np.asarray([3, 4], dtype=np.int32),
        top_k=1,
    )

    assert isinstance(results, tuple)
    assert results[0].score == pytest.approx(1.0)
    assert isinstance(results[0].score, float)
