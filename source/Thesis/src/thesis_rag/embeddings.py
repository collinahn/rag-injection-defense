import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from pydantic import Field, StrictBool, field_validator, model_validator

from thesis_rag.canonical import JsonValue, canonical_json, sha256_text
from thesis_rag.domain import Chunk, FrozenRecord, Sha256Hex
from thesis_rag.qa import ResearchQuestion, validate_research_questions
from thesis_rag.retrieval import RetrievalEvaluationSummary, evaluate_retrieval

EmbeddingTransport = Callable[
    [str, dict[str, object] | None],
    dict[str, object],
]


class EmbeddingAdapter(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def model_revision(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    @property
    def query_instruction(self) -> str: ...

    def verify_model(self) -> None: ...

    def runtime_version(self) -> str: ...

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray: ...


def format_embedding_query(question: str, instruction: str) -> str:
    stripped_question = question.strip()
    stripped_instruction = instruction.strip()
    if not stripped_question:
        raise ValueError("embedding question must be nonblank")
    if not stripped_instruction:
        raise ValueError("embedding query instruction must be nonblank")
    return f"Instruct: {stripped_instruction}\nQuery:{stripped_question}"


@dataclass(frozen=True)
class OllamaEmbeddingAdapter:
    model_id: str
    model_revision: str
    dimensions: int
    query_instruction: str
    endpoint: str = "http://127.0.0.1:11434"
    truncate: bool = False
    keep_alive: str = "5m"
    timeout_seconds: float = 120.0
    transport: EmbeddingTransport | None = None

    def verify_model(self) -> None:
        response = self._request("/api/tags", None)
        raw_models = response.get("models")
        if not isinstance(raw_models, list):
            raise ValueError("Ollama tags response missing models")
        matching = [
            model
            for model in raw_models
            if isinstance(model, dict) and model.get("name") == self.model_id
        ]
        if len(matching) != 1:
            raise ValueError(f"expected one installed model named {self.model_id}")
        model = matching[0]
        if model.get("digest") != self.model_revision:
            raise ValueError("Ollama embedding model revision mismatch")
        details = model.get("details")
        if not isinstance(details, dict):
            raise ValueError("Ollama model details are missing")
        if details.get("embedding_length") != self.dimensions:
            raise ValueError("Ollama embedding dimensions mismatch")

    def runtime_version(self) -> str:
        response = self._request("/api/version", None)
        version = response.get("version")
        if not isinstance(version, str) or not version.strip():
            raise ValueError("Ollama version response is invalid")
        return f"ollama {version.strip()}"

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(tuple(texts))

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        formatted = tuple(
            format_embedding_query(text, self.query_instruction) for text in texts
        )
        return self._embed(formatted)

    def _embed(self, texts: tuple[str, ...]) -> np.ndarray:
        if not texts:
            raise ValueError("embedding input must be non-empty")
        if any(not text.strip() for text in texts):
            raise ValueError("embedding input must contain nonblank text")
        response = self._request(
            "/api/embed",
            {
                "model": self.model_id,
                "input": list(texts),
                "dimensions": self.dimensions,
                "truncate": self.truncate,
                "keep_alive": self.keep_alive,
            },
        )
        if response.get("model") != self.model_id:
            raise ValueError("Ollama embedding response model mismatch")
        raw_embeddings = response.get("embeddings")
        try:
            embeddings = np.asarray(raw_embeddings, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("Ollama embeddings must be numeric") from error
        if embeddings.shape != (len(texts), self.dimensions):
            raise ValueError("Ollama embedding response shape mismatch")
        if not np.isfinite(embeddings).all():
            raise ValueError("Ollama embeddings must be finite")
        if np.any(np.linalg.norm(embeddings, axis=1) == 0.0):
            raise ValueError("Ollama embeddings must have nonzero norm")
        return embeddings

    def _request(
        self,
        path: str,
        payload: dict[str, object] | None,
    ) -> dict[str, object]:
        if self.transport is not None:
            return self.transport(path, payload)
        method = "GET" if payload is None else "POST"
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.endpoint.rstrip('/')}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except (
            HTTPError,
            URLError,
            TimeoutError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as error:
            raise ValueError(f"Ollama request failed for {path}") from error
        if not isinstance(decoded, dict):
            raise ValueError(f"Ollama response must be an object for {path}")
        return cast(dict[str, object], decoded)


class EmbeddingCacheMetadata(FrozenRecord):
    schema_version: str
    corpus_sha256: Sha256Hex
    questions_sha256: Sha256Hex
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

    @field_validator(
        "schema_version",
        "model_id",
        "runtime_version",
        "query_instruction",
    )
    @classmethod
    def reject_blank_strings(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("embedding cache metadata strings must be nonblank")
        return stripped

    @field_validator("chunk_ids", "question_ids")
    @classmethod
    def reject_blank_or_duplicate_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not item.strip() for item in value):
            raise ValueError("embedding cache IDs must be nonblank and non-empty")
        if len(set(value)) != len(value):
            raise ValueError("embedding cache IDs must be unique")
        return value

    @model_validator(mode="after")
    def validate_hash_cardinality(self) -> "EmbeddingCacheMetadata":
        if len(self.chunk_ids) != len(self.chunk_text_sha256):
            raise ValueError("chunk IDs and hashes must have equal length")
        if len(self.question_ids) != len(self.question_text_sha256):
            raise ValueError("question IDs and hashes must have equal length")
        return self


@dataclass(frozen=True)
class EmbeddingCache:
    metadata: EmbeddingCacheMetadata
    chunk_vectors: np.ndarray
    question_vectors: np.ndarray


class RetrievalPilotSummary(FrozenRecord):
    schema_version: str
    corpus_sha256: Sha256Hex
    questions_sha256: Sha256Hex
    model_id: str
    model_revision: Sha256Hex
    runtime_version: str
    dimensions: int = Field(strict=True, ge=1)
    query_instruction: str
    chunk_count: int = Field(strict=True, ge=1)
    question_count: int = Field(strict=True, ge=1)
    draft_question_count: int = Field(strict=True, ge=0)
    cache_reused: StrictBool
    cache_file_sha256: Sha256Hex
    metadata_file_sha256: Sha256Hex
    results_file_sha256: Sha256Hex
    retrieval: RetrievalEvaluationSummary


def build_embedding_cache(
    cache_path: Path,
    metadata_path: Path,
    *,
    metadata: EmbeddingCacheMetadata,
    chunk_vectors: object,
    question_vectors: object,
) -> EmbeddingCache:
    chunk_array, question_array = _validate_cache_arrays(
        metadata,
        chunk_vectors,
        question_vectors,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    cache_temporary = _temporary_path(cache_path)
    metadata_temporary = _temporary_path(metadata_path)
    try:
        with cache_temporary.open("wb") as stream:
            np.savez(
                stream,
                chunk_vectors=chunk_array,
                question_vectors=question_array,
            )
            stream.flush()
            os.fsync(stream.fileno())
        metadata_contents = (
            canonical_json(cast(JsonValue, metadata.model_dump(mode="json"))) + "\n"
        )
        with metadata_temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(metadata_contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(cache_temporary, cache_path)
        os.replace(metadata_temporary, metadata_path)
    finally:
        cache_temporary.unlink(missing_ok=True)
        metadata_temporary.unlink(missing_ok=True)
    return EmbeddingCache(
        metadata=metadata,
        chunk_vectors=chunk_array,
        question_vectors=question_array,
    )


def load_embedding_cache(
    cache_path: Path,
    metadata_path: Path,
    *,
    expected_metadata: EmbeddingCacheMetadata,
) -> EmbeddingCache:
    try:
        metadata = EmbeddingCacheMetadata.model_validate_json(
            metadata_path.read_text(encoding="utf-8"),
            strict=True,
        )
    except (OSError, ValueError) as error:
        raise ValueError("invalid embedding cache metadata") from error
    if metadata != expected_metadata:
        raise ValueError("embedding cache metadata mismatch")
    try:
        with np.load(cache_path, allow_pickle=False) as cache:
            if set(cache.files) != {"chunk_vectors", "question_vectors"}:
                raise ValueError("embedding cache contains unexpected arrays")
            chunk_vectors = np.asarray(cache["chunk_vectors"], dtype=np.float64)
            question_vectors = np.asarray(cache["question_vectors"], dtype=np.float64)
    except (OSError, ValueError) as error:
        raise ValueError("invalid embedding cache") from error
    chunk_vectors, question_vectors = _validate_cache_arrays(
        metadata,
        chunk_vectors,
        question_vectors,
    )
    return EmbeddingCache(
        metadata=metadata,
        chunk_vectors=chunk_vectors,
        question_vectors=question_vectors,
    )


def run_retrieval_pilot(
    chunks: Sequence[Chunk],
    questions: Sequence[ResearchQuestion],
    adapter: EmbeddingAdapter,
    *,
    corpus_sha256: str,
    questions_sha256: str,
    top_k: int,
    cache_path: Path,
    metadata_path: Path,
    results_path: Path,
    summary_path: Path,
) -> RetrievalPilotSummary:
    validate_research_questions(questions, chunks)
    adapter.verify_model()
    runtime_version = adapter.runtime_version()
    metadata = EmbeddingCacheMetadata(
        schema_version="1.0.0",
        corpus_sha256=corpus_sha256,
        questions_sha256=questions_sha256,
        model_id=adapter.model_id,
        model_revision=adapter.model_revision,
        runtime_version=runtime_version,
        dimensions=adapter.dimensions,
        query_instruction=adapter.query_instruction,
        normalization_policy="cosine_at_retrieval",
        chunk_ids=tuple(chunk.chunk_id for chunk in chunks),
        chunk_text_sha256=tuple(sha256_text(chunk.text) for chunk in chunks),
        question_ids=tuple(question.question_id for question in questions),
        question_text_sha256=tuple(
            sha256_text(question.question) for question in questions
        ),
    )
    cache_exists = cache_path.exists()
    metadata_exists = metadata_path.exists()
    if cache_exists != metadata_exists:
        raise ValueError("embedding cache and metadata must exist together")
    if cache_exists:
        cache = load_embedding_cache(
            cache_path,
            metadata_path,
            expected_metadata=metadata,
        )
        cache_reused = True
    else:
        chunk_vectors = adapter.embed_documents(
            tuple(chunk.text for chunk in chunks)
        )
        question_vectors = adapter.embed_queries(
            tuple(question.question for question in questions)
        )
        cache = build_embedding_cache(
            cache_path,
            metadata_path,
            metadata=metadata,
            chunk_vectors=chunk_vectors,
            question_vectors=question_vectors,
        )
        cache_reused = False

    retrieval_summary, retrieval_results = evaluate_retrieval(
        chunks,
        questions,
        cache.chunk_vectors,
        cache.question_vectors,
        top_k=top_k,
    )
    result_contents = "".join(
        f"{canonical_json(result)}\n" for result in retrieval_results
    )
    _atomic_write_text(results_path, result_contents)
    summary = RetrievalPilotSummary(
        schema_version="1.0.0",
        corpus_sha256=corpus_sha256,
        questions_sha256=questions_sha256,
        model_id=adapter.model_id,
        model_revision=adapter.model_revision,
        runtime_version=runtime_version,
        dimensions=adapter.dimensions,
        query_instruction=adapter.query_instruction,
        chunk_count=len(chunks),
        question_count=len(questions),
        draft_question_count=sum(
            question.review_status.value == "draft" for question in questions
        ),
        cache_reused=cache_reused,
        cache_file_sha256=_sha256_file(cache_path),
        metadata_file_sha256=_sha256_file(metadata_path),
        results_file_sha256=_sha256_file(results_path),
        retrieval=retrieval_summary,
    )
    _atomic_write_text(summary_path, f"{canonical_json(summary)}\n")
    return summary


def _validate_cache_arrays(
    metadata: EmbeddingCacheMetadata,
    chunk_vectors: object,
    question_vectors: object,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        chunk_array = np.asarray(chunk_vectors, dtype=np.float64)
        question_array = np.asarray(question_vectors, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("embedding cache arrays must be numeric") from error
    expected_chunk_shape = (len(metadata.chunk_ids), metadata.dimensions)
    expected_question_shape = (len(metadata.question_ids), metadata.dimensions)
    if chunk_array.shape != expected_chunk_shape:
        raise ValueError("embedding cache chunk shape mismatch")
    if question_array.shape != expected_question_shape:
        raise ValueError("embedding cache question shape mismatch")
    if not np.isfinite(chunk_array).all() or not np.isfinite(question_array).all():
        raise ValueError("embedding cache arrays must be finite")
    if np.any(np.linalg.norm(chunk_array, axis=1) == 0.0):
        raise ValueError("embedding cache chunk vectors must have nonzero norm")
    if np.any(np.linalg.norm(question_array, axis=1) == 0.0):
        raise ValueError("embedding cache question vectors must have nonzero norm")
    chunk_array.setflags(write=False)
    question_array.setflags(write=False)
    return chunk_array, question_array


def _temporary_path(target: Path) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    return Path(raw_path)


def _atomic_write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
