from pathlib import Path

import numpy as np
import pytest

from thesis_rag.canonical import sha256_text
from thesis_rag.main_embeddings import (
    MainEmbeddingCacheMetadata,
    build_main_embedding_cache,
    load_main_embedding_cache,
)


def _metadata() -> MainEmbeddingCacheMetadata:
    return MainEmbeddingCacheMetadata(
        schema_version="1.0.0",
        corpus_sha256="1" * 64,
        questions_sha256="2" * 64,
        detector_cases_sha256="3" * 64,
        model_id="qwen3-embedding:0.6b",
        model_revision="4" * 64,
        runtime_version="ollama test",
        dimensions=2,
        query_instruction="retrieve university rules",
        normalization_policy="cosine_at_retrieval",
        chunk_ids=("CH_1", "CH_2"),
        chunk_text_sha256=(sha256_text("문서 1"), sha256_text("문서 2")),
        question_ids=("Q_TEST_001",),
        question_text_sha256=(sha256_text("질문"),),
        malicious_attack_ids=("ATK_TEST_001",),
        malicious_chunk_ids=("MCH_1",),
        malicious_text_sha256=(sha256_text("악성 문서"),),
    )


def test_main_embedding_cache_round_trips_all_three_vector_sets(
    tmp_path: Path,
) -> None:
    metadata = _metadata()
    cache_path = tmp_path / "main.npz"
    metadata_path = tmp_path / "main.metadata.json"

    built = build_main_embedding_cache(
        cache_path,
        metadata_path,
        metadata=metadata,
        chunk_vectors=((1.0, 0.0), (0.0, 1.0)),
        question_vectors=((0.9, 0.1),),
        malicious_vectors=((0.8, 0.2),),
    )
    loaded = load_main_embedding_cache(
        cache_path,
        metadata_path,
        expected_metadata=metadata,
    )

    assert built.metadata == metadata
    assert loaded.metadata == metadata
    assert loaded.chunk_vectors.tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert loaded.question_vectors.tolist() == [[0.9, 0.1]]
    assert loaded.malicious_vectors.tolist() == [[0.8, 0.2]]
    assert loaded.chunk_vectors.flags.writeable is False
    assert loaded.question_vectors.flags.writeable is False
    assert loaded.malicious_vectors.flags.writeable is False


def test_main_embedding_cache_rejects_metadata_or_array_tampering(
    tmp_path: Path,
) -> None:
    metadata = _metadata()
    cache_path = tmp_path / "main.npz"
    metadata_path = tmp_path / "main.metadata.json"
    build_main_embedding_cache(
        cache_path,
        metadata_path,
        metadata=metadata,
        chunk_vectors=((1.0, 0.0), (0.0, 1.0)),
        question_vectors=((0.9, 0.1),),
        malicious_vectors=((0.8, 0.2),),
    )

    changed = metadata.model_copy(update={"questions_sha256": "5" * 64})
    with pytest.raises(ValueError, match="main embedding cache metadata mismatch"):
        load_main_embedding_cache(
            cache_path,
            metadata_path,
            expected_metadata=changed,
        )

    with cache_path.open("wb") as stream:
        np.savez(
            stream,
            chunk_vectors=np.asarray(((1.0, 0.0),)),
            question_vectors=np.asarray(((0.9, 0.1),)),
            malicious_vectors=np.asarray(((0.8, 0.2),)),
        )
    with pytest.raises(ValueError, match="main embedding chunk shape mismatch"):
        load_main_embedding_cache(
            cache_path,
            metadata_path,
            expected_metadata=metadata,
        )


def test_metadata_requires_one_to_one_malicious_attack_and_chunk_ids() -> None:
    with pytest.raises(
        ValueError,
        match="malicious IDs and hashes must have equal length",
    ):
        _metadata().model_copy(
            update={"malicious_chunk_ids": ("MCH_1", "MCH_2")}
        ).model_validate(
            {
                **_metadata().model_dump(mode="python"),
                "malicious_chunk_ids": ("MCH_1", "MCH_2"),
            }
        )
