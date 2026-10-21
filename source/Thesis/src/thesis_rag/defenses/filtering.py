from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import ClassVar

from thesis_rag.defenses.base import render_d0_messages, render_d1_messages
from thesis_rag.domain import DefenseId, DefenseResult, PromptMessage, RetrievedChunk

FIXED_REFUSAL = "근거 문서를 안전하게 확인할 수 없어 답변할 수 없습니다."
FIXED_REFUSAL_SYSTEM_MESSAGE = (
    "안전한 근거 문서가 없으므로 다음 문장만 그대로 답하라:\n" + FIXED_REFUSAL
)


def _validate_probability(value: float, *, label: str) -> None:
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be a finite value in [0.0, 1.0]")


def _partition_chunks(
    chunks: tuple[RetrievedChunk, ...],
    detector_scores: Mapping[str, float],
    threshold: float,
) -> tuple[tuple[RetrievedChunk, ...], tuple[str, ...]]:
    missing = [item.chunk.chunk_id for item in chunks if item.chunk.chunk_id not in detector_scores]
    if missing:
        raise ValueError(f"missing detector scores: {','.join(sorted(missing))}")

    for item in chunks:
        _validate_probability(
            detector_scores[item.chunk.chunk_id],
            label=f"invalid detector score for {item.chunk.chunk_id}",
        )

    retained = tuple(
        item for item in chunks if detector_scores[item.chunk.chunk_id] < threshold
    )
    blocked = tuple(
        item.chunk.chunk_id
        for item in chunks
        if detector_scores[item.chunk.chunk_id] >= threshold
    )
    return retained, blocked


def _fixed_refusal_result(
    defense_id: DefenseId, threshold: float, blocked_chunk_ids: tuple[str, ...]
) -> DefenseResult:
    return DefenseResult(
        defense_id=defense_id,
        messages=(
            PromptMessage(role="system", content=FIXED_REFUSAL_SYSTEM_MESSAGE),
            PromptMessage(role="user", content=FIXED_REFUSAL),
        ),
        retained_chunk_ids=(),
        blocked_chunk_ids=blocked_chunk_ids,
        defense_metadata={
            "threshold": threshold,
            "all_chunks_blocked": True,
            "fixed_refusal": FIXED_REFUSAL,
        },
    )


@dataclass(frozen=True)
class D3Defense:
    threshold: float

    defense_id: ClassVar[DefenseId] = DefenseId.D3

    def __post_init__(self) -> None:
        _validate_probability(self.threshold, label="threshold")

    def apply(
        self,
        question: str,
        chunks: tuple[RetrievedChunk, ...],
        detector_scores: Mapping[str, float],
    ) -> DefenseResult:
        retained, blocked = _partition_chunks(chunks, detector_scores, self.threshold)
        if not retained:
            return _fixed_refusal_result(self.defense_id, self.threshold, blocked)
        return DefenseResult(
            defense_id=self.defense_id,
            messages=render_d0_messages(question, retained),
            retained_chunk_ids=tuple(item.chunk.chunk_id for item in retained),
            blocked_chunk_ids=blocked,
            defense_metadata={"threshold": self.threshold},
        )


@dataclass(frozen=True)
class D4Defense:
    threshold: float

    defense_id: ClassVar[DefenseId] = DefenseId.D4

    def __post_init__(self) -> None:
        _validate_probability(self.threshold, label="threshold")

    def apply(
        self,
        question: str,
        chunks: tuple[RetrievedChunk, ...],
        detector_scores: Mapping[str, float],
    ) -> DefenseResult:
        retained, blocked = _partition_chunks(chunks, detector_scores, self.threshold)
        if not retained:
            return _fixed_refusal_result(self.defense_id, self.threshold, blocked)
        return DefenseResult(
            defense_id=self.defense_id,
            messages=render_d1_messages(question, retained),
            retained_chunk_ids=tuple(item.chunk.chunk_id for item in retained),
            blocked_chunk_ids=blocked,
            defense_metadata={"threshold": self.threshold},
        )
