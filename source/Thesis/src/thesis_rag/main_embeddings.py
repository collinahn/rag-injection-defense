import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
from pydantic import Field, field_validator, model_validator

from thesis_rag.canonical import JsonValue, canonical_json
from thesis_rag.domain import FrozenRecord, Sha256Hex


class MainEmbeddingCacheMetadata(FrozenRecord):
    schema_version: Literal["1.0.0"] = "1.0.0"
    corpus_sha256: Sha256Hex
    questions_sha256: Sha256Hex
    detector_cases_sha256: Sha256Hex
    model_id: str
    model_revision: Sha256Hex
    runtime_version: str
    dimensions: int = Field(strict=True, ge=1)
    query_instruction: str
    normalization_policy: Literal["cosine_at_retrieval"]
    chunk_ids: tuple[str, ...]
    chunk_text_sha256: tuple[Sha256Hex, ...]
    question_ids: tuple[str, ...]
    question_text_sha256: tuple[Sha256Hex, ...]
    malicious_attack_ids: tuple[str, ...]
    malicious_chunk_ids: tuple[str, ...]
    malicious_text_sha256: tuple[Sha256Hex, ...]

    @field_validator(
        "model_id",
        "runtime_version",
        "query_instruction",
    )
    @classmethod
    def strip_nonblank_strings(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("main embedding metadata strings must be nonblank")
        return stripped

    @field_validator(
        "chunk_ids",
        "question_ids",
        "malicious_attack_ids",
        "malicious_chunk_ids",
    )
    @classmethod
    def validate_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not item.strip() for item in value):
            raise ValueError("main embedding IDs must be nonblank and non-empty")
        if len(set(value)) != len(value):
            raise ValueError("main embedding IDs must be unique")
        return value

    @model_validator(mode="after")
    def validate_cardinality(self) -> "MainEmbeddingCacheMetadata":
        if len(self.chunk_ids) != len(self.chunk_text_sha256):
            raise ValueError("chunk IDs and hashes must have equal length")
        if len(self.question_ids) != len(self.question_text_sha256):
            raise ValueError("question IDs and hashes must have equal length")
        malicious_lengths = {
            len(self.malicious_attack_ids),
            len(self.malicious_chunk_ids),
            len(self.malicious_text_sha256),
        }
        if len(malicious_lengths) != 1:
            raise ValueError(
                "malicious IDs and hashes must have equal length"
            )
        return self


@dataclass(frozen=True)
class MainEmbeddingCache:
    metadata: MainEmbeddingCacheMetadata
    chunk_vectors: np.ndarray
    question_vectors: np.ndarray
    malicious_vectors: np.ndarray


def build_main_embedding_cache(
    cache_path: Path,
    metadata_path: Path,
    *,
    metadata: MainEmbeddingCacheMetadata,
    chunk_vectors: object,
    question_vectors: object,
    malicious_vectors: object,
) -> MainEmbeddingCache:
    arrays = _validated_arrays(
        metadata,
        chunk_vectors,
        question_vectors,
        malicious_vectors,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    cache_temporary = _temporary_path(cache_path)
    metadata_temporary = _temporary_path(metadata_path)
    try:
        with cache_temporary.open("wb") as stream:
            np.savez(
                stream,
                chunk_vectors=arrays[0],
                question_vectors=arrays[1],
                malicious_vectors=arrays[2],
            )
            stream.flush()
            os.fsync(stream.fileno())
        metadata_contents = (
            canonical_json(
                cast(JsonValue, metadata.model_dump(mode="json"))
            )
            + "\n"
        )
        with metadata_temporary.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as stream:
            stream.write(metadata_contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(cache_temporary, cache_path)
        os.replace(metadata_temporary, metadata_path)
    finally:
        cache_temporary.unlink(missing_ok=True)
        metadata_temporary.unlink(missing_ok=True)
    return MainEmbeddingCache(
        metadata=metadata,
        chunk_vectors=arrays[0],
        question_vectors=arrays[1],
        malicious_vectors=arrays[2],
    )


def load_main_embedding_cache(
    cache_path: Path,
    metadata_path: Path,
    *,
    expected_metadata: MainEmbeddingCacheMetadata,
) -> MainEmbeddingCache:
    try:
        metadata = MainEmbeddingCacheMetadata.model_validate_json(
            metadata_path.read_text(encoding="utf-8"),
            strict=True,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("invalid main embedding cache metadata") from error
    if metadata != expected_metadata:
        raise ValueError("main embedding cache metadata mismatch")
    try:
        with np.load(cache_path, allow_pickle=False) as cache:
            expected_arrays = {
                "chunk_vectors",
                "question_vectors",
                "malicious_vectors",
            }
            if set(cache.files) != expected_arrays:
                raise ValueError(
                    "main embedding cache contains unexpected arrays"
                )
            arrays = _validated_arrays(
                metadata,
                cache["chunk_vectors"],
                cache["question_vectors"],
                cache["malicious_vectors"],
            )
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith(
            "main embedding "
        ):
            raise
        raise ValueError("invalid main embedding cache") from error
    return MainEmbeddingCache(
        metadata=metadata,
        chunk_vectors=arrays[0],
        question_vectors=arrays[1],
        malicious_vectors=arrays[2],
    )


def _validated_arrays(
    metadata: MainEmbeddingCacheMetadata,
    chunk_vectors: object,
    question_vectors: object,
    malicious_vectors: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        chunks = np.asarray(chunk_vectors, dtype=np.float64)
        questions = np.asarray(question_vectors, dtype=np.float64)
        malicious = np.asarray(malicious_vectors, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("main embedding arrays must be numeric") from error
    expected_shapes = (
        (
            "chunk",
            chunks,
            (len(metadata.chunk_ids), metadata.dimensions),
        ),
        (
            "question",
            questions,
            (len(metadata.question_ids), metadata.dimensions),
        ),
        (
            "malicious",
            malicious,
            (len(metadata.malicious_attack_ids), metadata.dimensions),
        ),
    )
    for label, array, expected_shape in expected_shapes:
        if array.shape != expected_shape:
            raise ValueError(f"main embedding {label} shape mismatch")
        if not np.isfinite(array).all():
            raise ValueError(f"main embedding {label} vectors must be finite")
        if np.any(np.linalg.norm(array, axis=1) == 0.0):
            raise ValueError(
                f"main embedding {label} vectors must have nonzero norm"
            )
        array.setflags(write=False)
    return chunks, questions, malicious


def _temporary_path(target: Path) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    return Path(raw_path)
