from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from thesis_rag.domain import Chunk, DetectorScore, GenerationResult, PromptMessage

_GROUNDED_ANSWER = "The answer is grounded in the retrieved documents."


class GeneratorAdapter(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def model_revision(self) -> str: ...

    def generate(self, messages: tuple[PromptMessage, ...]) -> GenerationResult: ...


class DetectorAdapter(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def model_revision(self) -> str: ...

    def score(self, chunk: Chunk) -> float: ...


@dataclass(frozen=True)
class MockGenerator:
    success_markers: tuple[str, ...]
    follow_malicious_instructions: bool = False
    model_id: str = "mock-generator"
    model_revision: str = "v1"

    def generate(self, messages: tuple[PromptMessage, ...]) -> GenerationResult:
        if self.follow_malicious_instructions:
            for marker in self.success_markers:
                if any(marker in message.content for message in messages):
                    return GenerationResult(raw_response=marker)
        return GenerationResult(raw_response=_GROUNDED_ANSWER)


@dataclass(frozen=True)
class MockDetector:
    scores: Mapping[str, float]
    model_id: str = "mock-detector"
    model_revision: str = "v1"

    def score(self, chunk: Chunk) -> float:
        try:
            score = self.scores[chunk.chunk_id]
        except KeyError as error:
            raise ValueError(f"missing detector score for {chunk.chunk_id}") from error
        return DetectorScore(chunk_id=chunk.chunk_id, score=score).score


@dataclass(frozen=True)
class BypassDetector:
    model_id: str = "detector-not-applied"
    model_revision: str = "d0-d2-development-pilot-v1"

    def score(self, chunk: Chunk) -> float:
        del chunk
        return 0.0
