import re
from collections.abc import Mapping

from thesis_rag.defenses.base import D2_SYSTEM_MESSAGE, render_d0_messages
from thesis_rag.domain import DefenseId, DefenseResult, RetrievedChunk


def datamark_text(text: str, *, marker: str) -> str:
    if not marker or marker.isspace():
        raise ValueError("marker must contain a non-whitespace character")
    return re.sub(r"\S+", lambda match: f"{marker}{match.group(0)}", text)


class D2Defense:
    defense_id = DefenseId.D2
    marker = "§"

    def apply(
        self,
        question: str,
        chunks: tuple[RetrievedChunk, ...],
        detector_scores: Mapping[str, float],
    ) -> DefenseResult:
        original_characters = sum(len(item.chunk.text) for item in chunks)
        transformed_characters = sum(
            len(datamark_text(item.chunk.text, marker=self.marker)) for item in chunks
        )
        return DefenseResult(
            defense_id=self.defense_id,
            messages=render_d0_messages(
                question,
                chunks,
                lambda text: datamark_text(text, marker=self.marker),
                system_message=D2_SYSTEM_MESSAGE,
            ),
            retained_chunk_ids=tuple(item.chunk.chunk_id for item in chunks),
            blocked_chunk_ids=(),
            defense_metadata={
                "original_characters": original_characters,
                "transformed_characters": transformed_characters,
                "marker": self.marker,
            },
        )
