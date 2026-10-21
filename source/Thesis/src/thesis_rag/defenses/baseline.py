from collections.abc import Mapping

from thesis_rag.defenses.base import render_d0_messages, render_d1_messages
from thesis_rag.domain import DefenseId, DefenseResult, RetrievedChunk


class D0Defense:
    defense_id = DefenseId.D0

    def apply(
        self,
        question: str,
        chunks: tuple[RetrievedChunk, ...],
        detector_scores: Mapping[str, float],
    ) -> DefenseResult:
        return DefenseResult(
            defense_id=self.defense_id,
            messages=render_d0_messages(question, chunks),
            retained_chunk_ids=tuple(item.chunk.chunk_id for item in chunks),
            blocked_chunk_ids=(),
        )


class D1Defense:
    defense_id = DefenseId.D1

    def apply(
        self,
        question: str,
        chunks: tuple[RetrievedChunk, ...],
        detector_scores: Mapping[str, float],
    ) -> DefenseResult:
        return DefenseResult(
            defense_id=self.defense_id,
            messages=render_d1_messages(question, chunks),
            retained_chunk_ids=tuple(item.chunk.chunk_id for item in chunks),
            blocked_chunk_ids=(),
        )
