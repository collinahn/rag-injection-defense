from dataclasses import FrozenInstanceError
from math import inf, nan

import pytest

from thesis_rag.defenses.baseline import D0Defense, D1Defense
from thesis_rag.defenses.filtering import D3Defense, D4Defense
from thesis_rag.domain import PromptMessage, RetrievedChunk

FIXED_REFUSAL = "근거 문서를 안전하게 확인할 수 없어 답변할 수 없습니다."
FIXED_REFUSAL_SYSTEM_MESSAGE = (
    "안전한 근거 문서가 없으므로 다음 문장만 그대로 답하라:\n"
    "근거 문서를 안전하게 확인할 수 없어 답변할 수 없습니다."
)


@pytest.mark.parametrize("defense_class", [D3Defense, D4Defense])
@pytest.mark.parametrize("threshold", [-0.1, 1.1, nan, inf, -inf])
def test_filtering_defenses_reject_invalid_thresholds(
    defense_class: type[D3Defense] | type[D4Defense], threshold: float
) -> None:
    with pytest.raises(ValueError, match=r"threshold must be a finite value in \[0.0, 1.0\]"):
        defense_class(threshold=threshold)


@pytest.mark.parametrize("defense", [D3Defense(threshold=0.5), D4Defense(threshold=0.5)])
def test_filtering_defenses_are_frozen(defense: D3Defense | D4Defense) -> None:
    with pytest.raises(FrozenInstanceError):
        defense.threshold = 0.1  # type: ignore[misc]


def test_d3_blocks_score_equal_to_threshold(retrieved_chunks: tuple[RetrievedChunk, ...]) -> None:
    target = retrieved_chunks[0].chunk.chunk_id
    scores = {chunk.chunk.chunk_id: 0.1 for chunk in retrieved_chunks}
    scores[target] = 0.5

    result = D3Defense(threshold=0.5).apply("질문", retrieved_chunks, scores)

    assert target in result.blocked_chunk_ids
    assert target not in result.retained_chunk_ids


def test_d3_preserves_input_order_and_ignores_extra_scores(
    retrieved_chunks: tuple[RetrievedChunk, ...],
) -> None:
    scores = {
        "DOC_001_CH_0001": 0.7,
        "DOC_002_CH_0003": 0.2,
        "unrelated": 1.0,
    }

    result = D3Defense(threshold=0.5).apply("질문", retrieved_chunks, scores)

    assert result.retained_chunk_ids == ("DOC_002_CH_0003",)
    assert result.blocked_chunk_ids == ("DOC_001_CH_0001",)
    assert result.defense_metadata == {"threshold": 0.5}


@pytest.mark.parametrize("defense", [D3Defense(threshold=0.5), D4Defense(threshold=0.5)])
def test_filtering_defenses_reject_missing_scores(
    defense: D3Defense | D4Defense, retrieved_chunks: tuple[RetrievedChunk, ...]
) -> None:
    with pytest.raises(ValueError, match="missing detector scores: DOC_002_CH_0003"):
        defense.apply("질문", retrieved_chunks, {"DOC_001_CH_0001": 0.0})


@pytest.mark.parametrize("defense", [D3Defense(threshold=0.5), D4Defense(threshold=0.5)])
@pytest.mark.parametrize("score", [-0.1, 1.1, nan, inf, -inf])
def test_filtering_defenses_reject_invalid_detector_scores(
    defense: D3Defense | D4Defense,
    retrieved_chunks: tuple[RetrievedChunk, ...],
    score: float,
) -> None:
    scores = {chunk.chunk.chunk_id: 0.0 for chunk in retrieved_chunks}
    scores[retrieved_chunks[0].chunk.chunk_id] = score

    with pytest.raises(ValueError, match="invalid detector score for DOC_001_CH_0001"):
        defense.apply("질문", retrieved_chunks, scores)


def test_d3_passes_retained_chunks_to_the_exact_d0_renderer(
    retrieved_chunks: tuple[RetrievedChunk, ...],
) -> None:
    scores = {"DOC_001_CH_0001": 0.1, "DOC_002_CH_0003": 0.8}

    expected = D0Defense().apply("질문", (retrieved_chunks[0],), scores)
    result = D3Defense(threshold=0.5).apply("질문", retrieved_chunks, scores)

    assert result.messages == expected.messages
    assert result.retained_chunk_ids == expected.retained_chunk_ids
    assert result.blocked_chunk_ids == ("DOC_002_CH_0003",)


def test_d4_passes_d3_retained_chunks_to_the_exact_d1_renderer(
    retrieved_chunks: tuple[RetrievedChunk, ...],
) -> None:
    target = retrieved_chunks[0].chunk.chunk_id
    scores = {chunk.chunk.chunk_id: 0.0 for chunk in retrieved_chunks}
    scores[target] = 1.0

    d3 = D3Defense(threshold=0.5).apply("질문", retrieved_chunks, scores)
    d4 = D4Defense(threshold=0.5).apply("질문", retrieved_chunks, scores)
    expected = D1Defense().apply("질문", (retrieved_chunks[1],), scores)

    assert d4.retained_chunk_ids == d3.retained_chunk_ids
    assert d4.blocked_chunk_ids == d3.blocked_chunk_ids
    assert d4.messages == expected.messages


@pytest.mark.parametrize("defense", [D3Defense(threshold=0.5), D4Defense(threshold=0.5)])
def test_filtering_defenses_use_the_same_fixed_refusal_when_all_chunks_are_blocked(
    defense: D3Defense | D4Defense, retrieved_chunks: tuple[RetrievedChunk, ...]
) -> None:
    scores = {chunk.chunk.chunk_id: 0.5 for chunk in retrieved_chunks}

    result = defense.apply("질문", retrieved_chunks, scores)

    assert result.messages == (
        PromptMessage(role="system", content=FIXED_REFUSAL_SYSTEM_MESSAGE),
        PromptMessage(role="user", content=FIXED_REFUSAL),
    )
    assert result.retained_chunk_ids == ()
    assert result.blocked_chunk_ids == ("DOC_001_CH_0001", "DOC_002_CH_0003")
    assert result.defense_metadata == {
        "threshold": 0.5,
        "all_chunks_blocked": True,
        "fixed_refusal": FIXED_REFUSAL,
    }
