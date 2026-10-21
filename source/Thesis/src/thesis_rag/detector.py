from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from math import isfinite
from typing import Annotated, Any, Literal, Protocol, cast

from pydantic import Field, StringConstraints

from thesis_rag.domain import Chunk, FiniteFloat, FrozenRecord, Probability

GitCommitSha = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]


class PromptGuardModelIdentity(FrozenRecord):
    model_id: str = Field(min_length=1)
    model_revision: GitCommitSha
    labels: tuple[str, ...]
    max_position_embeddings: int = Field(strict=True, ge=1)
    transformers_version: str = Field(min_length=1)
    torch_version: str = Field(min_length=1)


class PromptGuardScoringPolicy(FrozenRecord):
    schema_version: Literal["1.0.0"]
    max_length: int = Field(strict=True, ge=1)
    overlap_tokens: int = Field(strict=True, ge=0)
    configured_malicious_label: str = Field(min_length=1)
    resolved_malicious_label_index: int = Field(strict=True, ge=0)
    windowing_policy: Literal["fixed_token_windows_with_overlap"]
    segment_aggregation: Literal["maximum_malicious_probability"]


@dataclass(frozen=True)
class PromptGuardWindowOutput:
    logits: tuple[float, ...]
    probabilities: tuple[float, ...]


class PromptGuardSegmentScore(FrozenRecord):
    segment_index: int = Field(strict=True, ge=0)
    content_token_count: int = Field(strict=True, ge=1)
    logits: tuple[FiniteFloat, ...] = Field(min_length=1)
    probabilities: tuple[Probability, ...] = Field(min_length=1)


class PromptGuardScore(FrozenRecord):
    chunk_id: str = Field(min_length=1)
    score: Probability
    token_count: int = Field(strict=True, ge=1)
    segment_count: int = Field(strict=True, ge=1)
    segments: tuple[PromptGuardSegmentScore, ...] = Field(min_length=1)


class PromptGuardRuntime(Protocol):
    identity: PromptGuardModelIdentity

    def encode(self, text: str) -> tuple[int, ...]: ...

    def num_special_tokens_to_add(self) -> int: ...

    def classify(
        self,
        windows: tuple[tuple[int, ...], ...],
    ) -> tuple[PromptGuardWindowOutput, ...]: ...


def segment_token_ids(
    token_ids: Sequence[int],
    *,
    content_capacity: int,
    overlap_tokens: int,
) -> tuple[tuple[int, ...], ...]:
    if content_capacity < 1:
        raise ValueError("content capacity must be positive")
    if overlap_tokens < 0 or overlap_tokens >= content_capacity:
        raise ValueError("overlap must be smaller than content capacity")
    if not token_ids:
        return ()

    step = content_capacity - overlap_tokens
    windows: list[tuple[int, ...]] = []
    start = 0
    while start < len(token_ids):
        window = tuple(token_ids[start : start + content_capacity])
        windows.append(window)
        if start + content_capacity >= len(token_ids):
            break
        start += step
    return tuple(windows)


def build_prompt_guard_window_features(
    window: Sequence[int],
    *,
    bos_token_id: int,
    eos_token_id: int,
    token_type_id: int,
) -> dict[str, list[int]]:
    if not window:
        raise ValueError("Prompt Guard token window must not be empty")
    if any(
        not isinstance(token_id, int) or token_id < 0
        for token_id in (*window, bos_token_id, eos_token_id, token_type_id)
    ):
        raise ValueError("Prompt Guard token IDs must be nonnegative integers")
    input_ids = [bos_token_id, *window, eos_token_id]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "token_type_ids": [token_type_id] * len(input_ids),
    }


def resolve_prompt_guard_malicious_label_index(
    labels: Sequence[str],
    malicious_label: str,
) -> int:
    normalized_labels = tuple(labels)
    if malicious_label in normalized_labels:
        return normalized_labels.index(malicious_label)
    if (
        malicious_label == "MALICIOUS"
        and normalized_labels == ("LABEL_0", "LABEL_1")
    ):
        return 1
    raise ValueError("Prompt Guard model has no configured malicious label")


def build_prompt_guard_scoring_policy(
    *,
    max_length: int,
    overlap_tokens: int,
    malicious_label: str,
    labels: Sequence[str],
) -> PromptGuardScoringPolicy:
    return PromptGuardScoringPolicy(
        schema_version="1.0.0",
        max_length=max_length,
        overlap_tokens=overlap_tokens,
        configured_malicious_label=malicious_label,
        resolved_malicious_label_index=(
            resolve_prompt_guard_malicious_label_index(
                labels,
                malicious_label,
            )
        ),
        windowing_policy="fixed_token_windows_with_overlap",
        segment_aggregation="maximum_malicious_probability",
    )


class PromptGuardDetectorAdapter:
    def __init__(
        self,
        *,
        model_id: str,
        model_revision: str,
        max_length: int,
        overlap_tokens: int,
        malicious_label: str,
        runtime: PromptGuardRuntime | None = None,
    ) -> None:
        self.model_id = model_id.strip()
        self.model_revision = model_revision.strip()
        self.max_length = max_length
        self.overlap_tokens = overlap_tokens
        self.malicious_label = malicious_label.strip()
        if not self.model_id or not self.model_revision or not self.malicious_label:
            raise ValueError("Prompt Guard identity fields must be nonblank")
        if max_length < 1:
            raise ValueError("Prompt Guard max_length must be positive")
        if overlap_tokens < 0:
            raise ValueError("Prompt Guard overlap must be nonnegative")
        self._runtime = runtime or TransformersPromptGuardRuntime(
            model_id=self.model_id,
            model_revision=self.model_revision,
            max_length=self.max_length,
            malicious_label=self.malicious_label,
        )

    @property
    def scoring_policy(self) -> PromptGuardScoringPolicy:
        return build_prompt_guard_scoring_policy(
            max_length=self.max_length,
            overlap_tokens=self.overlap_tokens,
            malicious_label=self.malicious_label,
            labels=self.verify_model().labels,
        )

    def verify_model(self) -> PromptGuardModelIdentity:
        identity = self._runtime.identity
        if (
            identity.model_id != self.model_id
            or identity.model_revision != self.model_revision
        ):
            raise ValueError("Prompt Guard model identity mismatch")
        resolve_prompt_guard_malicious_label_index(
            identity.labels,
            self.malicious_label,
        )
        if identity.max_position_embeddings < self.max_length:
            raise ValueError("Prompt Guard context window is smaller than configured")
        return identity

    def score(self, chunk: Chunk) -> float:
        return self.score_with_metadata(chunk).score

    def score_with_metadata(self, chunk: Chunk) -> PromptGuardScore:
        identity = self.verify_model()
        token_ids = self._runtime.encode(chunk.text)
        if not token_ids:
            raise ValueError("Prompt Guard tokenizer returned no content tokens")
        special_token_count = self._runtime.num_special_tokens_to_add()
        content_capacity = self.max_length - special_token_count
        if content_capacity < 1:
            raise ValueError("Prompt Guard special tokens consume the full window")
        windows = segment_token_ids(
            token_ids,
            content_capacity=content_capacity,
            overlap_tokens=self.overlap_tokens,
        )
        outputs = self._runtime.classify(windows)
        if len(outputs) != len(windows) or not outputs:
            raise ValueError("invalid Prompt Guard malicious probabilities count")
        malicious_index = resolve_prompt_guard_malicious_label_index(
            identity.labels,
            self.malicious_label,
        )
        segments: list[PromptGuardSegmentScore] = []
        for index, (window, output) in enumerate(zip(windows, outputs, strict=True)):
            if (
                len(output.logits) != len(identity.labels)
                or len(output.probabilities) != len(identity.labels)
            ):
                raise ValueError("invalid Prompt Guard output label count")
            if any(not isfinite(logit) for logit in output.logits):
                raise ValueError("invalid Prompt Guard raw logits")
            if any(
                not isfinite(probability) or not 0.0 <= probability <= 1.0
                for probability in output.probabilities
            ):
                raise ValueError("invalid Prompt Guard malicious probabilities values")
            segments.append(
                PromptGuardSegmentScore(
                    segment_index=index,
                    content_token_count=len(window),
                    logits=output.logits,
                    probabilities=output.probabilities,
                )
            )
        return PromptGuardScore(
            chunk_id=chunk.chunk_id,
            score=max(
                segment.probabilities[malicious_index]
                for segment in segments
            ),
            token_count=len(token_ids),
            segment_count=len(windows),
            segments=tuple(segments),
        )


class TransformersPromptGuardRuntime:
    def __init__(
        self,
        *,
        model_id: str,
        model_revision: str,
        max_length: int,
        malicious_label: str,
    ) -> None:
        try:
            torch = import_module("torch")
            transformers = import_module("transformers")
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Prompt Guard requires the optional local-model dependencies"
            ) from error

        self._torch: Any = torch
        self._tokenizer: Any = transformers.AutoTokenizer.from_pretrained(
            model_id,
            revision=model_revision,
            trust_remote_code=False,
        )
        self._model: Any = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_id,
            revision=model_revision,
            trust_remote_code=False,
        )
        self._model.eval()

        config = self._model.config
        loaded_revision = getattr(config, "_commit_hash", None)
        if loaded_revision != model_revision:
            raise ValueError("loaded Prompt Guard revision does not match requested commit")
        raw_id2label = getattr(config, "id2label", None)
        if not isinstance(raw_id2label, dict) or not raw_id2label:
            raise ValueError("Prompt Guard model has no label mapping")
        try:
            labels_by_id = {
                int(label_id): str(label)
                for label_id, label in raw_id2label.items()
            }
        except (TypeError, ValueError) as error:
            raise ValueError("Prompt Guard label mapping is invalid") from error
        labels = tuple(labels_by_id[index] for index in sorted(labels_by_id))
        self._malicious_label_index = (
            resolve_prompt_guard_malicious_label_index(
                labels,
                malicious_label,
            )
        )
        max_position_embeddings = getattr(config, "max_position_embeddings", None)
        if (
            not isinstance(max_position_embeddings, int)
            or max_position_embeddings < max_length
        ):
            raise ValueError("Prompt Guard model context window is incompatible")
        self.identity = PromptGuardModelIdentity(
            model_id=model_id,
            model_revision=model_revision,
            labels=labels,
            max_position_embeddings=max_position_embeddings,
            transformers_version=str(transformers.__version__),
            torch_version=str(torch.__version__),
        )

    def encode(self, text: str) -> tuple[int, ...]:
        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        if not isinstance(token_ids, list) or any(
            not isinstance(token_id, int) for token_id in token_ids
        ):
            raise ValueError("Prompt Guard tokenizer returned invalid token IDs")
        return tuple(token_ids)

    def num_special_tokens_to_add(self) -> int:
        value = self._tokenizer.num_special_tokens_to_add(pair=False)
        if not isinstance(value, int) or value < 0:
            raise ValueError("Prompt Guard tokenizer returned invalid special-token count")
        return value

    def classify(
        self,
        windows: tuple[tuple[int, ...], ...],
    ) -> tuple[PromptGuardWindowOutput, ...]:
        if not windows:
            raise ValueError("Prompt Guard requires at least one token window")
        bos_token_id = getattr(self._tokenizer, "bos_token_id", None)
        eos_token_id = getattr(self._tokenizer, "eos_token_id", None)
        token_type_id = getattr(self._tokenizer, "pad_token_type_id", None)
        if (
            not isinstance(bos_token_id, int)
            or not isinstance(eos_token_id, int)
            or not isinstance(token_type_id, int)
        ):
            raise ValueError("Prompt Guard tokenizer special-token IDs are invalid")
        features = [
            build_prompt_guard_window_features(
                window,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                token_type_id=token_type_id,
            )
            for window in windows
        ]
        batch = self._tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        with self._torch.inference_mode():
            logits_tensor = self._model(**batch).logits.float()
            probability_tensor = self._torch.softmax(
                logits_tensor,
                dim=-1,
            )
        logits = cast(
            list[list[float]],
            logits_tensor.detach().cpu().tolist(),
        )
        probabilities = cast(
            list[list[float]],
            probability_tensor.detach().cpu().tolist(),
        )
        return tuple(
            PromptGuardWindowOutput(
                logits=tuple(float(value) for value in raw_logits),
                probabilities=tuple(float(value) for value in raw_probabilities),
            )
            for raw_logits, raw_probabilities in zip(
                logits,
                probabilities,
                strict=True,
            )
        )
