import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from thesis_rag.canonical import sha256_text
from thesis_rag.domain import (
    FrozenRecord,
    GenerationResult,
    PromptMessage,
    Sha256Hex,
)

GenerationTransport = Callable[
    [str, dict[str, object] | None],
    dict[str, object],
]


class OllamaModelIdentity(FrozenRecord):
    model_id: str
    model_revision: Sha256Hex
    chat_template_sha256: Sha256Hex
    quantization: str
    parameter_size: str
    runtime_version: str


@dataclass(frozen=True)
class OllamaGeneratorAdapter:
    model_id: str
    model_revision: str
    expected_template_sha256: str
    expected_quantization: str
    context_window: int
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    seed: int
    thinking: bool = False
    endpoint: str = "http://127.0.0.1:11434"
    keep_alive: str = "5m"
    timeout_seconds: float = 300.0
    transport: GenerationTransport | None = None

    def __post_init__(self) -> None:
        if self.thinking:
            raise ValueError("thesis generation adapter requires non-thinking mode")
        if self.model_id.endswith(":latest") or ":" not in self.model_id:
            raise ValueError("generation model must use an explicit non-latest tag")

    def verify_model(self) -> OllamaModelIdentity:
        tags_response = self._request("/api/tags", None)
        raw_models = tags_response.get("models")
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
            raise ValueError("Ollama generation model revision mismatch")
        details = model.get("details")
        if not isinstance(details, dict):
            raise ValueError("Ollama generation model details are missing")
        quantization = details.get("quantization_level")
        if quantization != self.expected_quantization:
            raise ValueError("Ollama generation model quantization mismatch")
        parameter_size = details.get("parameter_size")
        if not isinstance(parameter_size, str) or not parameter_size.strip():
            raise ValueError("Ollama generation model parameter size is missing")

        show_response = self._request(
            "/api/show",
            {"model": self.model_id, "verbose": False},
        )
        template = show_response.get("template")
        if not isinstance(template, str) or not template:
            raise ValueError("Ollama generation model chat template is missing")
        template_sha256 = sha256_text(template)
        if template_sha256 != self.expected_template_sha256:
            raise ValueError("Ollama generation model chat template mismatch")

        version_response = self._request("/api/version", None)
        version = version_response.get("version")
        if not isinstance(version, str) or not version.strip():
            raise ValueError("Ollama version response is invalid")
        return OllamaModelIdentity(
            model_id=self.model_id,
            model_revision=self.model_revision,
            chat_template_sha256=template_sha256,
            quantization=quantization,
            parameter_size=parameter_size.strip(),
            runtime_version=f"ollama {version.strip()}",
        )

    def generate(self, messages: tuple[PromptMessage, ...]) -> GenerationResult:
        if not messages:
            raise ValueError("generation messages must be non-empty")
        response = self._request(
            "/api/chat",
            {
                "model": self.model_id,
                "messages": [
                    {"role": message.role, "content": message.content}
                    for message in messages
                ],
                "stream": False,
                "think": False,
                "keep_alive": self.keep_alive,
                "options": {
                    "num_ctx": self.context_window,
                    "num_predict": self.max_new_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "min_p": self.min_p,
                    "seed": self.seed,
                },
            },
        )
        if response.get("model") != self.model_id:
            raise ValueError("Ollama generation response model mismatch")
        if response.get("done") is not True:
            raise ValueError("Ollama generation response is incomplete")
        message = response.get("message")
        if not isinstance(message, dict):
            raise ValueError("Ollama generation response message is missing")
        if message.get("role") != "assistant":
            raise ValueError("Ollama generation response role must be assistant")
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("Ollama generation response content must be text")
        thinking = message.get("thinking")
        if thinking not in (None, ""):
            raise ValueError("Ollama returned thinking content in non-thinking mode")
        tool_calls = message.get("tool_calls")
        if tool_calls not in (None, []):
            raise ValueError("Ollama returned an unexpected tool call")
        finish_reason = response.get("done_reason")
        if not isinstance(finish_reason, str) or not finish_reason.strip():
            raise ValueError("Ollama generation response done_reason is invalid")
        input_tokens = _token_count(response, "prompt_eval_count")
        output_tokens = _token_count(response, "eval_count")
        return GenerationResult(
            raw_response=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason.strip(),
        )

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


def _token_count(response: dict[str, object], field: str) -> int:
    value = response.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Ollama generation response {field} is invalid")
    return value
