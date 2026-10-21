from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Self

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_DEFAULT_EMBEDDING_MODEL_ID = "qwen3-embedding:0.6b"
_DEFAULT_EMBEDDING_MODEL_REVISION = (
    "ac6da0dfba84a81fdbfbaf330198c33cd77c4cdfc53e8bc50eb581914a15621d"
)
_DEFAULT_QUERY_INSTRUCTION = (
    "Given a university academic information question, retrieve relevant "
    "official passages that answer the question"
)
_DEFAULT_GENERATION_MODEL_ID = "qwen3:8b-q8_0"
_DEFAULT_GENERATION_MODEL_REVISION = (
    "e56358ca25dd14db6853a9f68a92d717aaa6f0a94250a72d1a0f3d86a9f30130"
)
_DEFAULT_CHAT_TEMPLATE_SHA256 = (
    "ae370d884f108d16e7cc8fd5259ebc5773a0afa6e078b11f4ed7e39a27e0dfc4"
)
_DEFAULT_DETECTOR_MODEL_ID = "meta-llama/Llama-Prompt-Guard-2-86M"
_DEFAULT_DETECTOR_MODEL_REVISION = "a8ded8e697ce7c355e395a0df51f94adb4a2fd27"


class FrozenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RetrievalConfig(FrozenConfig):
    top_k: int = Field(default=5, ge=1, le=20)


class EmbeddingConfig(FrozenConfig):
    provider: Literal["ollama"] = "ollama"
    model_id: str = _DEFAULT_EMBEDDING_MODEL_ID
    model_revision: str = Field(
        default=_DEFAULT_EMBEDDING_MODEL_REVISION,
        pattern=r"^[0-9a-f]{64}$",
    )
    dimensions: int = Field(default=1024, strict=True, ge=1)
    query_instruction: str = _DEFAULT_QUERY_INSTRUCTION
    endpoint: str = "http://127.0.0.1:11434"
    truncate: Literal[False] = False
    keep_alive: str = "5m"

    @field_validator(
        "model_id",
        "query_instruction",
        "endpoint",
        "keep_alive",
    )
    @classmethod
    def strip_nonblank_fields(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("embedding string fields must be nonblank")
        return stripped

    @model_validator(mode="after")
    def reject_unversioned_model_alias(self) -> Self:
        if self.model_id.endswith(":latest") or ":" not in self.model_id:
            raise ValueError("embedding model must use an explicit non-latest tag")
        return self


class GenerationConfig(FrozenConfig):
    provider: Literal["ollama"] = "ollama"
    model_id: str = _DEFAULT_GENERATION_MODEL_ID
    model_revision: str = Field(
        default=_DEFAULT_GENERATION_MODEL_REVISION,
        pattern=r"^[0-9a-f]{64}$",
    )
    chat_template_sha256: str = Field(
        default=_DEFAULT_CHAT_TEMPLATE_SHA256,
        pattern=r"^[0-9a-f]{64}$",
    )
    quantization: Literal["Q8_0"] = "Q8_0"
    endpoint: str = "http://127.0.0.1:11434"
    keep_alive: str = "5m"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=20, strict=True, ge=1)
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    seed: int = 42
    thinking: Literal[False] = False
    concurrency: Literal[1] = 1
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    context_window: int = Field(default=8192, ge=1024)

    @field_validator("model_id", "endpoint", "keep_alive")
    @classmethod
    def strip_generation_strings(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("generation string fields must be nonblank")
        return stripped

    @model_validator(mode="after")
    def reject_unversioned_model_alias(self) -> Self:
        if self.model_id.endswith(":latest") or ":" not in self.model_id:
            raise ValueError("generation model must use an explicit non-latest tag")
        return self


class DetectorConfig(FrozenConfig):
    provider: Literal["transformers"] = "transformers"
    model_id: str = _DEFAULT_DETECTOR_MODEL_ID
    model_revision: str = Field(
        default=_DEFAULT_DETECTOR_MODEL_REVISION,
        pattern=r"^[0-9a-f]{40}$",
    )
    device: Literal["cpu"] = "cpu"
    max_length: Literal[512] = 512
    overlap_tokens: int = Field(default=32, strict=True, ge=0)
    malicious_label: Literal["MALICIOUS", "LABEL_1"] = "MALICIOUS"
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("model_id")
    @classmethod
    def strip_detector_model_id(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("detector model_id must be nonblank")
        return stripped

    @model_validator(mode="after")
    def validate_segmentation_window(self) -> Self:
        if self.overlap_tokens >= self.max_length - 2:
            raise ValueError("detector overlap must leave room for advancing tokens")
        return self


class ResearchConfig(FrozenConfig):
    protocol_version: str
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)


def load_config(path: Path) -> ResearchConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("configuration root must be a mapping")
    return ResearchConfig.model_validate(data)
