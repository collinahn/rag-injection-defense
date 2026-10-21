import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from thesis_rag.canonical import sha256_text
from thesis_rag.domain import Chunk, Language, Split
from thesis_rag.embeddings import (
    EmbeddingCacheMetadata,
    OllamaEmbeddingAdapter,
    build_embedding_cache,
    format_embedding_query,
    load_embedding_cache,
    run_retrieval_pilot,
)
from thesis_rag.qa import QuestionReviewStatus, QuestionType, ResearchQuestion
from thesis_rag.retrieval import evaluate_retrieval

_MODEL_ID = "qwen3-embedding:0.6b"
_MODEL_REVISION = "a" * 64
_QUERY_INSTRUCTION = (
    "Given a university academic information question, retrieve relevant "
    "official passages that answer the question"
)


class RecordingTransport:
    def __init__(
        self,
        responses: Mapping[str, list[dict[str, object]]],
    ) -> None:
        self._responses = {path: list(values) for path, values in responses.items()}
        self.calls: list[tuple[str, dict[str, object] | None]] = []

    def __call__(
        self,
        path: str,
        payload: dict[str, object] | None,
    ) -> dict[str, object]:
        self.calls.append((path, payload))
        try:
            return self._responses[path].pop(0)
        except (KeyError, IndexError) as error:
            raise AssertionError(f"unexpected request: {path}") from error


def _adapter(transport: RecordingTransport) -> OllamaEmbeddingAdapter:
    return OllamaEmbeddingAdapter(
        model_id=_MODEL_ID,
        model_revision=_MODEL_REVISION,
        dimensions=2,
        query_instruction=_QUERY_INSTRUCTION,
        transport=transport,
    )


def test_format_embedding_query_uses_frozen_qwen_instruction_template() -> None:
    assert format_embedding_query("  졸업 요건은 무엇인가요?  ", _QUERY_INSTRUCTION) == (
        "Instruct: Given a university academic information question, retrieve "
        "relevant official passages that answer the question\n"
        "Query:졸업 요건은 무엇인가요?"
    )


def test_ollama_adapter_verifies_digest_and_embeds_documents_and_queries() -> None:
    transport = RecordingTransport(
        {
            "/api/tags": [
                {
                    "models": [
                        {
                            "name": _MODEL_ID,
                            "digest": _MODEL_REVISION,
                            "details": {"embedding_length": 2},
                        }
                    ]
                }
            ],
            "/api/embed": [
                {"model": _MODEL_ID, "embeddings": [[1.0, 0.0], [0.0, 1.0]]},
                {"model": _MODEL_ID, "embeddings": [[0.5, 0.5]]},
            ],
        }
    )
    adapter = _adapter(transport)

    adapter.verify_model()
    document_vectors = adapter.embed_documents(("문서 1", "문서 2"))
    query_vectors = adapter.embed_queries(("질문",))

    assert document_vectors.tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert query_vectors.tolist() == [[0.5, 0.5]]
    assert transport.calls == [
        ("/api/tags", None),
        (
            "/api/embed",
            {
                "model": _MODEL_ID,
                "input": ["문서 1", "문서 2"],
                "dimensions": 2,
                "truncate": False,
                "keep_alive": "5m",
            },
        ),
        (
            "/api/embed",
            {
                "model": _MODEL_ID,
                "input": [
                    "Instruct: Given a university academic information question, "
                    "retrieve relevant official passages that answer the question\n"
                    "Query:질문"
                ],
                "dimensions": 2,
                "truncate": False,
                "keep_alive": "5m",
            },
        ),
    ]


def test_ollama_adapter_rejects_model_digest_mismatch() -> None:
    transport = RecordingTransport(
        {
            "/api/tags": [
                {
                    "models": [
                        {
                            "name": _MODEL_ID,
                            "digest": "b" * 64,
                            "details": {"embedding_length": 2},
                        }
                    ]
                }
            ]
        }
    )

    with pytest.raises(ValueError, match="model revision mismatch"):
        _adapter(transport).verify_model()


@pytest.mark.parametrize(
    "response",
    [
        {"model": "other-model:1", "embeddings": [[1.0, 0.0]]},
        {"model": _MODEL_ID, "embeddings": []},
        {"model": _MODEL_ID, "embeddings": [[1.0]]},
        {"model": _MODEL_ID, "embeddings": [[float("nan"), 0.0]]},
    ],
)
def test_ollama_adapter_rejects_invalid_embedding_response(
    response: dict[str, object],
) -> None:
    transport = RecordingTransport({"/api/embed": [response]})

    with pytest.raises(ValueError):
        _adapter(transport).embed_documents(("문서",))


def test_embedding_cache_is_bound_to_inputs_and_rejects_metadata_mismatch(
    tmp_path: Path,
) -> None:
    metadata = EmbeddingCacheMetadata(
        schema_version="1.0.0",
        corpus_sha256="1" * 64,
        questions_sha256="2" * 64,
        model_id=_MODEL_ID,
        model_revision=_MODEL_REVISION,
        runtime_version="ollama version 1",
        dimensions=2,
        query_instruction=_QUERY_INSTRUCTION,
        normalization_policy="cosine_at_retrieval",
        chunk_ids=("CH_1", "CH_2"),
        chunk_text_sha256=(sha256_text("문서 1"), sha256_text("문서 2")),
        question_ids=("Q_DEV_001",),
        question_text_sha256=(sha256_text("질문"),),
    )
    cache_path = tmp_path / "embeddings.npz"
    metadata_path = tmp_path / "embeddings.metadata.json"

    build_embedding_cache(
        cache_path,
        metadata_path,
        metadata=metadata,
        chunk_vectors=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        question_vectors=np.asarray([[1.0, 0.0]]),
    )

    loaded = load_embedding_cache(
        cache_path,
        metadata_path,
        expected_metadata=metadata,
    )
    assert loaded.chunk_vectors.tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert loaded.question_vectors.tolist() == [[1.0, 0.0]]

    changed = metadata.model_copy(update={"questions_sha256": "3" * 64})
    with pytest.raises(ValueError, match="embedding cache metadata mismatch"):
        load_embedding_cache(
            cache_path,
            metadata_path,
            expected_metadata=changed,
        )


def _question(
    question_id: str,
    *gold_chunk_ids: str,
    question_type: QuestionType = QuestionType.SINGLE_SOURCE,
) -> ResearchQuestion:
    document_ids = tuple(f"DOC_{chunk_id[-1]}" for chunk_id in gold_chunk_ids)
    return ResearchQuestion(
        question_id=question_id,
        split=Split.DEV,
        language=Language.KO,
        question=f"{question_id} 질문",
        answerable=True,
        required_facts=("사실",),
        forbidden_claims=(),
        gold_chunk_ids=gold_chunk_ids,
        question_type=question_type,
        reference_answer="답변",
        gold_document_ids=document_ids,
        source_citation="공식 문서",
        review_status=QuestionReviewStatus.DRAFT,
    )


def test_retrieval_evaluation_reports_any_and_all_gold_recall() -> None:
    chunks = tuple(
        Chunk(
            chunk_id=f"CH_{index}",
            document_id=f"DOC_{index}",
            text=f"문서 {index}",
            language=Language.KO,
        )
        for index in range(1, 7)
    )
    questions = (
        _question("Q_DEV_001", "CH_1"),
        _question(
            "Q_DEV_002",
            "CH_2",
            "CH_6",
            question_type=QuestionType.MULTI_SOURCE,
        ),
    )
    chunk_vectors = np.eye(6)
    question_vectors = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    summary, results = evaluate_retrieval(
        chunks,
        questions,
        chunk_vectors,
        question_vectors,
        top_k=5,
    )

    assert summary.answerable_question_count == 2
    assert summary.hit_count == 2
    assert summary.all_gold_hit_count == 1
    assert summary.recall_at_k == 1.0
    assert summary.all_gold_recall_at_k == 0.5
    assert summary.mean_reciprocal_rank == 1.0
    assert results[1].retrieved_gold_chunk_ids == ("CH_2",)
    assert results[1].missing_gold_chunk_ids == ("CH_6",)
    assert results[1].all_gold_hit is False


class FixedEmbeddingAdapter:
    model_id = _MODEL_ID
    model_revision = _MODEL_REVISION
    dimensions = 2
    query_instruction = _QUERY_INSTRUCTION

    def __init__(self, *, fail_if_embedding_requested: bool = False) -> None:
        self.fail_if_embedding_requested = fail_if_embedding_requested
        self.verified = False

    def verify_model(self) -> None:
        self.verified = True

    def runtime_version(self) -> str:
        return "ollama test"

    def embed_documents(self, texts: tuple[str, ...]) -> np.ndarray:
        if self.fail_if_embedding_requested:
            raise AssertionError("existing valid cache should be reused")
        assert texts == ("근거 문서", "비관련 문서")
        return np.asarray([[1.0, 0.0], [0.0, 1.0]])

    def embed_queries(self, texts: tuple[str, ...]) -> np.ndarray:
        if self.fail_if_embedding_requested:
            raise AssertionError("existing valid cache should be reused")
        assert texts == ("근거는 무엇인가?",)
        return np.asarray([[1.0, 0.0]])


def test_retrieval_pilot_writes_canonical_results_and_reuses_bound_cache(
    tmp_path: Path,
) -> None:
    chunks = (
        Chunk(
            chunk_id="CH_1",
            document_id="DOC_1",
            text="근거 문서",
            language=Language.KO,
        ),
        Chunk(
            chunk_id="CH_2",
            document_id="DOC_2",
            text="비관련 문서",
            language=Language.KO,
        ),
    )
    questions = (
        ResearchQuestion(
            question_id="Q_DEV_001",
            split=Split.DEV,
            language=Language.KO,
            question="근거는 무엇인가?",
            answerable=True,
            required_facts=("근거",),
            forbidden_claims=(),
            gold_chunk_ids=("CH_1",),
            question_type=QuestionType.SINGLE_SOURCE,
            reference_answer="근거",
            gold_document_ids=("DOC_1",),
            source_citation="공식 문서",
            review_status=QuestionReviewStatus.DRAFT,
        ),
    )
    paths = {
        "cache_path": tmp_path / "embeddings.npz",
        "metadata_path": tmp_path / "embeddings.metadata.json",
        "results_path": tmp_path / "results.jsonl",
        "summary_path": tmp_path / "summary.json",
    }
    adapter = FixedEmbeddingAdapter()

    first = run_retrieval_pilot(
        chunks,
        questions,
        adapter,
        corpus_sha256="1" * 64,
        questions_sha256="2" * 64,
        top_k=1,
        **paths,
    )
    second = run_retrieval_pilot(
        chunks,
        questions,
        FixedEmbeddingAdapter(fail_if_embedding_requested=True),
        corpus_sha256="1" * 64,
        questions_sha256="2" * 64,
        top_k=1,
        **paths,
    )

    assert adapter.verified is True
    assert first.cache_reused is False
    assert second.cache_reused is True
    assert first.retrieval.recall_at_k == 1.0
    assert second.retrieval == first.retrieval
    assert paths["results_path"].read_text(encoding="utf-8").endswith("\n")
    assert paths["summary_path"].read_text(encoding="utf-8").endswith("\n")
    result_line = paths["results_path"].read_text(encoding="utf-8").strip()
    assert json.loads(result_line)["retrieved_chunk_ids"] == ["CH_1"]
