from collections.abc import Mapping

import pytest

from thesis_rag.canonical import sha256_text
from thesis_rag.domain import PromptMessage
from thesis_rag.generation import OllamaGeneratorAdapter

_MODEL_ID = "qwen3:8b-q8_0"
_MODEL_REVISION = "a" * 64
_TEMPLATE = "{{ if .System }}system: {{ .System }}{{ end }}"
_TEMPLATE_SHA256 = sha256_text(_TEMPLATE)


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


def _adapter(transport: RecordingTransport) -> OllamaGeneratorAdapter:
    return OllamaGeneratorAdapter(
        model_id=_MODEL_ID,
        model_revision=_MODEL_REVISION,
        expected_template_sha256=_TEMPLATE_SHA256,
        expected_quantization="Q8_0",
        context_window=8192,
        max_new_tokens=256,
        temperature=0.0,
        top_p=1.0,
        top_k=20,
        min_p=0.0,
        seed=42,
        thinking=False,
        keep_alive="5m",
        transport=transport,
    )


def test_ollama_generator_verifies_revision_template_runtime_and_quantization() -> None:
    transport = RecordingTransport(
        {
            "/api/tags": [
                {
                    "models": [
                        {
                            "name": _MODEL_ID,
                            "digest": _MODEL_REVISION,
                            "details": {
                                "family": "qwen3",
                                "parameter_size": "8.19B",
                                "quantization_level": "Q8_0",
                            },
                        }
                    ]
                }
            ],
            "/api/show": [
                {
                    "template": _TEMPLATE,
                    "details": {
                        "family": "qwen3",
                        "parameter_size": "8.19B",
                        "quantization_level": "Q8_0",
                    },
                }
            ],
            "/api/version": [{"version": "0.32.4"}],
        }
    )

    identity = _adapter(transport).verify_model()

    assert identity.model_id == _MODEL_ID
    assert identity.model_revision == _MODEL_REVISION
    assert identity.chat_template_sha256 == _TEMPLATE_SHA256
    assert identity.quantization == "Q8_0"
    assert identity.parameter_size == "8.19B"
    assert identity.runtime_version == "ollama 0.32.4"
    assert transport.calls == [
        ("/api/tags", None),
        ("/api/show", {"model": _MODEL_ID, "verbose": False}),
        ("/api/version", None),
    ]


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("digest", "b" * 64, "model revision mismatch"),
        ("quantization_level", "Q4_K_M", "quantization mismatch"),
    ],
)
def test_ollama_generator_rejects_model_identity_mismatch(
    field: str,
    replacement: str,
    message: str,
) -> None:
    details = {
        "family": "qwen3",
        "parameter_size": "8.19B",
        "quantization_level": "Q8_0",
    }
    model = {
        "name": _MODEL_ID,
        "digest": _MODEL_REVISION,
        "details": details,
    }
    if field == "digest":
        model[field] = replacement
    else:
        details[field] = replacement
    transport = RecordingTransport({"/api/tags": [{"models": [model]}]})

    with pytest.raises(ValueError, match=message):
        _adapter(transport).verify_model()


def test_ollama_generator_rejects_chat_template_mismatch() -> None:
    transport = RecordingTransport(
        {
            "/api/tags": [
                {
                    "models": [
                        {
                            "name": _MODEL_ID,
                            "digest": _MODEL_REVISION,
                            "details": {
                                "family": "qwen3",
                                "parameter_size": "8.19B",
                                "quantization_level": "Q8_0",
                            },
                        }
                    ]
                }
            ],
            "/api/show": [{"template": "changed template"}],
        }
    )

    with pytest.raises(ValueError, match="chat template mismatch"):
        _adapter(transport).verify_model()


def test_ollama_generator_sends_frozen_nonthinking_options_and_preserves_output() -> None:
    transport = RecordingTransport(
        {
            "/api/chat": [
                {
                    "model": _MODEL_ID,
                    "message": {
                        "role": "assistant",
                        "content": "  근거 기반 답변\n",
                        "thinking": "",
                        "tool_calls": [],
                    },
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 120,
                    "eval_count": 18,
                }
            ]
        }
    )
    adapter = _adapter(transport)
    messages = (
        PromptMessage(role="system", content="시스템"),
        PromptMessage(role="user", content="질문"),
    )

    result = adapter.generate(messages)

    assert result.raw_response == "  근거 기반 답변\n"
    assert result.input_tokens == 120
    assert result.output_tokens == 18
    assert result.finish_reason == "stop"
    assert transport.calls == [
        (
            "/api/chat",
            {
                "model": _MODEL_ID,
                "messages": [
                    {"role": "system", "content": "시스템"},
                    {"role": "user", "content": "질문"},
                ],
                "stream": False,
                "think": False,
                "keep_alive": "5m",
                "options": {
                    "num_ctx": 8192,
                    "num_predict": 256,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": 20,
                    "min_p": 0.0,
                    "seed": 42,
                },
            },
        )
    ]


@pytest.mark.parametrize(
    "response",
    [
        {
            "model": "other:1",
            "message": {"role": "assistant", "content": "답변"},
            "done": True,
            "prompt_eval_count": 1,
            "eval_count": 1,
        },
        {
            "model": _MODEL_ID,
            "message": {"role": "assistant", "content": "답변", "thinking": "숨은 추론"},
            "done": True,
            "prompt_eval_count": 1,
            "eval_count": 1,
        },
        {
            "model": _MODEL_ID,
            "message": {
                "role": "assistant",
                "content": "답변",
                "tool_calls": [{"function": {"name": "unexpected"}}],
            },
            "done": True,
            "prompt_eval_count": 1,
            "eval_count": 1,
        },
        {
            "model": _MODEL_ID,
            "message": {"role": "assistant", "content": "답변"},
            "done": False,
            "prompt_eval_count": 1,
            "eval_count": 1,
        },
        {
            "model": _MODEL_ID,
            "message": {"role": "assistant", "content": "답변"},
            "done": True,
            "prompt_eval_count": -1,
            "eval_count": 1,
        },
    ],
)
def test_ollama_generator_rejects_invalid_chat_response(
    response: dict[str, object],
) -> None:
    adapter = _adapter(RecordingTransport({"/api/chat": [response]}))

    with pytest.raises(ValueError):
        adapter.generate((PromptMessage(role="user", content="질문"),))
