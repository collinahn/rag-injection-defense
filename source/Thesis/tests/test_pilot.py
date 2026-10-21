import json
from pathlib import Path

import numpy as np
import pytest

from thesis_rag.adapters import MockDetector
from thesis_rag.canonical import sha256_text
from thesis_rag.defenses.datamarking import D2Defense
from thesis_rag.defenses.filtering import D3Defense
from thesis_rag.domain import (
    Chunk,
    DefenseId,
    GenerationResult,
    Language,
    PromptMessage,
    Split,
)
from thesis_rag.generation import OllamaModelIdentity
from thesis_rag.pilot import has_repetitive_output, run_clean_generation_pilot
from thesis_rag.qa import QuestionReviewStatus, QuestionType, ResearchQuestion


class FixedGenerator:
    model_id = "qwen3:8b-q8_0"
    model_revision = "a" * 64

    def __init__(self, responses: tuple[GenerationResult, ...]) -> None:
        self._responses = list(responses)
        self.prompts: list[tuple[PromptMessage, ...]] = []

    def verify_model(self) -> OllamaModelIdentity:
        return OllamaModelIdentity(
            model_id=self.model_id,
            model_revision=self.model_revision,
            chat_template_sha256="b" * 64,
            quantization="Q8_0",
            parameter_size="8.2B",
            runtime_version="ollama test",
        )

    def generate(self, messages: tuple[PromptMessage, ...]) -> GenerationResult:
        self.prompts.append(messages)
        return self._responses.pop(0)


def _question(
    question_id: str,
    *,
    answerable: bool,
) -> ResearchQuestion:
    if not answerable:
        return ResearchQuestion(
            question_id=question_id,
            split=Split.DEV,
            language=Language.KO,
            question="문서에 없는 정보는?",
            answerable=False,
            question_type=QuestionType.UNANSWERABLE,
            reference_answer="제공된 문서에서 확인할 수 없습니다.",
            review_status=QuestionReviewStatus.DRAFT,
        )
    return ResearchQuestion(
        question_id=question_id,
        split=Split.DEV,
        language=Language.KO,
        question="근거는 무엇인가?",
        answerable=True,
        required_facts=("근거",),
        forbidden_claims=(),
        gold_chunk_ids=("CH_1",),
        question_type=QuestionType.SINGLE_SOURCE,
        reference_answer="근거이다.",
        gold_document_ids=("DOC_1",),
        source_citation="공식 문서",
        review_status=QuestionReviewStatus.DRAFT,
    )


def test_clean_generation_pilot_writes_machine_and_human_review_artifacts(
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
        _question("Q_DEV_001", answerable=True),
        _question("Q_DEV_002", answerable=False),
    )
    generator = FixedGenerator(
        (
            GenerationResult(
                raw_response=(
                    '{"answer":"근거이다.","citations":["CH_1"],"answerable":true}'
                ),
                input_tokens=20,
                output_tokens=10,
                finish_reason="stop",
            ),
            GenerationResult(
                raw_response=(
                    '{"answer":"제공된 문서에서 확인할 수 없습니다.",'
                    '"citations":[],"answerable":false}'
                ),
                input_tokens=20,
                output_tokens=10,
                finish_reason="stop",
            ),
        )
    )
    results_path = tmp_path / "results.jsonl"
    summary_path = tmp_path / "summary.json"
    review_path = tmp_path / "review.md"

    summary = run_clean_generation_pilot(
        chunks,
        questions,
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        generator,
        corpus_sha256="1" * 64,
        questions_sha256="2" * 64,
        embedding_cache_sha256="3" * 64,
        generation_config_sha256="4" * 64,
        top_k=1,
        results_path=results_path,
        summary_path=summary_path,
        review_path=review_path,
    )

    assert summary.record_count == 2
    assert summary.json_parse_success_count == 2
    assert summary.json_parse_success_rate == 1.0
    assert summary.valid_citation_count == 2
    assert summary.answerability_match_count == 2
    assert summary.empty_response_count == 0
    assert summary.repetitive_response_count == 0
    assert summary.technical_failure_count == 0
    assert summary.automated_gate_pass is True
    assert summary.human_accuracy_review_required is True
    assert len(generator.prompts) == 2
    assert all('"citations"' in prompt[0].content for prompt in generator.prompts)
    result_values = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
    ]
    assert result_values[0]["parsed_response"]["citations"] == ["CH_1"]
    assert results_path.read_text(encoding="utf-8").endswith("\n")
    assert summary_path.read_text(encoding="utf-8").endswith("\n")
    review = review_path.read_text(encoding="utf-8")
    assert "Q_DEV_001" in review
    assert "근거이다." in review
    assert "- [ ] 의미 정확도 2점" in review
    assert summary.results_file_sha256 == sha256_text(
        results_path.read_text(encoding="utf-8")
    )


def test_clean_generation_pilot_keeps_schema_failure_separate_from_technical_failure(
    tmp_path: Path,
) -> None:
    chunks = (
        Chunk(
            chunk_id="CH_1",
            document_id="DOC_1",
            text="근거 문서",
            language=Language.KO,
        ),
    )
    generator = FixedGenerator(
        (
            GenerationResult(
                raw_response="형식이 아닌 답변",
                input_tokens=10,
                output_tokens=5,
                finish_reason="stop",
            ),
        )
    )

    summary = run_clean_generation_pilot(
        chunks,
        (_question("Q_DEV_001", answerable=True),),
        np.asarray([[1.0, 0.0]]),
        np.asarray([[1.0, 0.0]]),
        generator,
        corpus_sha256="1" * 64,
        questions_sha256="2" * 64,
        embedding_cache_sha256="3" * 64,
        generation_config_sha256="4" * 64,
        top_k=1,
        results_path=tmp_path / "results.jsonl",
        summary_path=tmp_path / "summary.json",
        review_path=tmp_path / "review.md",
    )

    assert summary.json_parse_success_count == 0
    assert summary.technical_failure_count == 0
    assert summary.automated_gate_pass is False


def test_clean_generation_pilot_applies_the_selected_nonfiltering_defense(
    tmp_path: Path,
) -> None:
    chunk = Chunk(
        chunk_id="CH_1",
        document_id="DOC_1",
        text="근거 문서",
        language=Language.KO,
    )
    generator = FixedGenerator(
        (
            GenerationResult(
                raw_response=(
                    '{"answer":"근거이다.","citations":["CH_1"],"answerable":true}'
                ),
                input_tokens=20,
                output_tokens=10,
                finish_reason="stop",
            ),
        )
    )

    summary = run_clean_generation_pilot(
        (chunk,),
        (_question("Q_DEV_001", answerable=True),),
        np.asarray([[1.0, 0.0]]),
        np.asarray([[1.0, 0.0]]),
        generator,
        corpus_sha256="1" * 64,
        questions_sha256="2" * 64,
        embedding_cache_sha256="3" * 64,
        generation_config_sha256="4" * 64,
        top_k=1,
        results_path=tmp_path / "results.jsonl",
        summary_path=tmp_path / "summary.json",
        review_path=tmp_path / "review.md",
        defense=D2Defense(),
    )

    assert summary.defense_id is DefenseId.D2
    assert "§근거" in generator.prompts[0][1].content


def test_clean_generation_pilot_requires_detector_for_filtering_defense(
    tmp_path: Path,
) -> None:
    chunk = Chunk(
        chunk_id="CH_1",
        document_id="DOC_1",
        text="근거 문서",
        language=Language.KO,
    )
    generator = FixedGenerator(())

    with pytest.raises(
        ValueError,
        match="filtering clean generation pilot requires a detector",
    ):
        run_clean_generation_pilot(
            (chunk,),
            (_question("Q_DEV_001", answerable=True),),
            np.asarray([[1.0, 0.0]]),
            np.asarray([[1.0, 0.0]]),
            generator,
            corpus_sha256="1" * 64,
            questions_sha256="2" * 64,
            embedding_cache_sha256="3" * 64,
            generation_config_sha256="4" * 64,
            top_k=1,
            results_path=tmp_path / "results.jsonl",
            summary_path=tmp_path / "summary.json",
            review_path=tmp_path / "review.md",
            defense=D3Defense(threshold=0.5),
        )


def test_clean_generation_pilot_records_real_filtering_decisions(
    tmp_path: Path,
) -> None:
    chunks = (
        Chunk(
            chunk_id="CH_1",
            document_id="DOC_1",
            text="정상 근거 문서",
            language=Language.KO,
        ),
        Chunk(
            chunk_id="CH_2",
            document_id="DOC_2",
            text="검색 문서 안의 악성 명령",
            language=Language.KO,
        ),
    )
    generator = FixedGenerator(
        (
            GenerationResult(
                raw_response=(
                    '{"answer":"근거이다.","citations":["CH_1"],"answerable":true}'
                ),
                input_tokens=20,
                output_tokens=10,
                finish_reason="stop",
            ),
        )
    )
    detector = MockDetector(
        scores={"CH_1": 0.1, "CH_2": 0.9},
        model_id="meta-llama/Llama-Prompt-Guard-2-86M",
        model_revision="a" * 40,
    )
    results_path = tmp_path / "results.jsonl"

    summary = run_clean_generation_pilot(
        chunks,
        (_question("Q_DEV_001", answerable=True),),
        np.asarray([[1.0, 0.0], [0.9, 0.1]]),
        np.asarray([[1.0, 0.0]]),
        generator,
        corpus_sha256="1" * 64,
        questions_sha256="2" * 64,
        embedding_cache_sha256="3" * 64,
        generation_config_sha256="4" * 64,
        top_k=2,
        results_path=results_path,
        summary_path=tmp_path / "summary.json",
        review_path=tmp_path / "review.md",
        defense=D3Defense(threshold=0.5),
        detector=detector,
    )

    record = json.loads(results_path.read_text(encoding="utf-8"))
    assert summary.defense_id is DefenseId.D3
    assert summary.detector_applied is True
    assert summary.detector_model_id == detector.model_id
    assert summary.detector_model_revision == detector.model_revision
    assert summary.detector_threshold == 0.5
    assert summary.blocked_chunk_count == 1
    assert record["retrieved_chunk_ids"] == ["CH_1", "CH_2"]
    assert record["ordered_chunk_ids"] == ["CH_1"]
    assert record["detector_scores"] == {"CH_1": 0.1, "CH_2": 0.9}
    assert record["blocked_chunk_ids"] == ["CH_2"]
    assert record["detector_model_id"] == detector.model_id
    assert record["detector_model_revision"] == detector.model_revision
    assert record["detector_threshold"] == 0.5
    assert record["detector_latency_ms"] >= 0.0
    assert record["generation_latency_ms"] >= 0.0


def test_clean_generation_pilot_rejects_detector_for_nonfiltering_defense(
    tmp_path: Path,
) -> None:
    chunk = Chunk(
        chunk_id="CH_1",
        document_id="DOC_1",
        text="근거 문서",
        language=Language.KO,
    )

    with pytest.raises(
        ValueError,
        match="nonfiltering clean generation pilot must not apply a detector",
    ):
        run_clean_generation_pilot(
            (chunk,),
            (_question("Q_DEV_001", answerable=True),),
            np.asarray([[1.0, 0.0]]),
            np.asarray([[1.0, 0.0]]),
            FixedGenerator(()),
            corpus_sha256="1" * 64,
            questions_sha256="2" * 64,
            embedding_cache_sha256="3" * 64,
            generation_config_sha256="4" * 64,
            top_k=1,
            results_path=tmp_path / "results.jsonl",
            summary_path=tmp_path / "summary.json",
            review_path=tmp_path / "review.md",
            detector=MockDetector(scores={"CH_1": 0.1}),
        )


def test_repetition_detector_requires_three_consecutive_nontrivial_blocks() -> None:
    assert has_repetitive_output("졸업 요건 안내입니다.") is False
    assert has_repetitive_output("동일한 긴 문장 동일한 긴 문장") is False
    assert (
        has_repetitive_output(
            "동일한 긴 문장이 반복됩니다 동일한 긴 문장이 반복됩니다 "
            "동일한 긴 문장이 반복됩니다"
        )
        is True
    )
