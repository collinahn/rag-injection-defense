from pathlib import Path

import pytest
from pydantic import ValidationError

from thesis_rag.config import ResearchConfig, load_config


def test_development_config_contains_research_defaults() -> None:
    config = load_config(Path("configs/development.yaml"))

    assert config.protocol_version == "0.1.0"
    assert config.retrieval.top_k == 5
    assert config.embedding.provider == "ollama"
    assert config.embedding.model_id == "qwen3-embedding:0.6b"
    assert config.embedding.model_revision == (
        "ac6da0dfba84a81fdbfbaf330198c33cd77c4cdfc53e8bc50eb581914a15621d"
    )
    assert config.embedding.dimensions == 1024
    assert config.embedding.truncate is False
    assert config.embedding.query_instruction == (
        "Given a university academic information question, retrieve relevant "
        "official passages that answer the question"
    )
    assert config.generation.temperature == 0.0
    assert config.generation.provider == "ollama"
    assert config.generation.model_id == "qwen3:8b-q8_0"
    assert config.generation.model_revision == (
        "e56358ca25dd14db6853a9f68a92d717aaa6f0a94250a72d1a0f3d86a9f30130"
    )
    assert config.generation.chat_template_sha256 == (
        "ae370d884f108d16e7cc8fd5259ebc5773a0afa6e078b11f4ed7e39a27e0dfc4"
    )
    assert config.generation.quantization == "Q8_0"
    assert config.generation.seed == 42
    assert config.generation.concurrency == 1
    assert config.generation.max_new_tokens == 256
    assert config.generation.context_window == 8192
    assert config.generation.top_p == 1.0
    assert config.generation.top_k == 20
    assert config.generation.min_p == 0.0
    assert config.generation.thinking is False
    assert config.detector.provider == "transformers"
    assert config.detector.model_id == "meta-llama/Llama-Prompt-Guard-2-86M"
    assert config.detector.model_revision == (
        "a8ded8e697ce7c355e395a0df51f94adb4a2fd27"
    )
    assert config.detector.device == "cpu"
    assert config.detector.max_length == 512
    assert config.detector.overlap_tokens == 32
    assert config.detector.malicious_label == "MALICIOUS"
    assert config.detector.threshold == 0.5


def test_config_rejects_detector_threshold_above_one() -> None:
    with pytest.raises(ValidationError):
        ResearchConfig.model_validate(
            {
                "protocol_version": "0.1.0",
                "retrieval": {"top_k": 5},
                "generation": {"temperature": 0, "seed": 42, "concurrency": 1},
                "detector": {"threshold": 1.1},
            }
        )


def test_config_rejects_unpinned_detector_revision() -> None:
    with pytest.raises(ValidationError):
        ResearchConfig.model_validate(
            {
                "protocol_version": "0.1.0",
                "detector": {
                    "model_id": "meta-llama/Llama-Prompt-Guard-2-86M",
                    "model_revision": "main",
                },
            }
        )


def test_config_rejects_detector_overlap_that_cannot_advance() -> None:
    with pytest.raises(ValidationError, match="overlap"):
        ResearchConfig.model_validate(
            {
                "protocol_version": "0.1.0",
                "detector": {
                    "max_length": 512,
                    "overlap_tokens": 510,
                },
            }
        )


def test_config_rejects_unversioned_embedding_model() -> None:
    with pytest.raises(ValidationError):
        ResearchConfig.model_validate(
            {
                "protocol_version": "0.1.0",
                "embedding": {
                    "provider": "ollama",
                    "model_id": "qwen3-embedding:latest",
                    "model_revision": "a" * 64,
                    "dimensions": 1024,
                    "query_instruction": "retrieve evidence",
                },
            }
        )


def test_config_rejects_unversioned_generation_model() -> None:
    with pytest.raises(ValidationError):
        ResearchConfig.model_validate(
            {
                "protocol_version": "0.1.0",
                "generation": {
                    "model_id": "qwen3:latest",
                    "model_revision": "a" * 64,
                    "chat_template_sha256": "b" * 64,
                },
            }
        )


@pytest.mark.parametrize("yaml_text", ["", "null", "[]", "research"])
def test_load_config_rejects_non_mapping_or_empty_root(
    tmp_path: Path, yaml_text: str
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(ValueError, match="mapping"):
        load_config(path)


def test_config_rejects_unknown_nested_field(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
protocol_version: 0.1.0
retrieval:
  top_k: 5
  unexpected: true
generation:
  temperature: 0.0
  seed: 42
  concurrency: 1
detector:
  threshold: 0.5
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(path)


def test_config_models_are_immutable() -> None:
    config = ResearchConfig.model_validate(
        {
            "protocol_version": "0.1.0",
            "retrieval": {"top_k": 5},
            "generation": {"temperature": 0, "seed": 42, "concurrency": 1},
            "detector": {"threshold": 0.5},
        }
    )

    with pytest.raises(ValidationError):
        config.detector.threshold = 0.8
