import re
import unicodedata
from enum import StrEnum
from typing import Annotated, Literal, Self, cast
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StringConstraints,
    field_validator,
    model_validator,
)

from thesis_rag.canonical import JsonValue, canonical_sha256, stable_id

MetadataValue = str | bool | int | float | None
Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Probability = Annotated[
    float,
    Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False),
]
NonNegativeFiniteFloat = Annotated[
    float,
    Field(strict=True, ge=0.0, allow_inf_nan=False),
]
FiniteFloat = Annotated[
    float,
    Field(strict=True, allow_inf_nan=False),
]
_HTTP_SCHEME_PATTERN = re.compile(r"https?://", re.IGNORECASE)
_URL_TOKEN_BOUNDARY_PATTERN = re.compile(r"[\s<>\"']")
_TRAILING_URL_PUNCTUATION = ".,;:!?)]}"


def normalize_research_text(text: str) -> str:
    """Normalize text for research comparisons and G2 substring matching."""
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


class FrozenRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("job_id", check_fields=False)
    @classmethod
    def reject_blank_job_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("job_id must be a non-empty string")
        return value


class Split(StrEnum):
    DEV = "dev"
    TEST = "test"


class Language(StrEnum):
    KO = "ko"
    EN = "en"


class AttackGoal(StrEnum):
    HIJACK = "G1"
    FALSE_RULE = "G2"
    CANARY = "G3"
    BLOCK = "G4"
    SUFFIX = "G5"


class AttackForm(StrEnum):
    PLAIN = "plain"
    DISGUISED = "disguised"


class AttackPosition(StrEnum):
    HEAD = "head"
    MIDDLE = "middle"
    TAIL = "tail"


class DefenseId(StrEnum):
    D0 = "D0"
    D1 = "D1"
    D2 = "D2"
    D3 = "D3"
    D4 = "D4"


class Chunk(FrozenRecord):
    chunk_id: str
    document_id: str
    text: str = Field(min_length=1)
    language: Language
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)


class RetrievedChunk(FrozenRecord):
    chunk: Chunk
    score: FiniteFloat
    rank: int = Field(strict=True, ge=1)


class Question(FrozenRecord):
    question_id: str
    split: Split
    language: Language
    question: str = Field(min_length=1)
    answerable: StrictBool
    required_facts: tuple[str, ...] = ()
    forbidden_claims: tuple[str, ...] = ()
    gold_chunk_ids: tuple[str, ...] = ()


class SuccessMarkerTemplateSlot(FrozenRecord):
    kind: Literal["success_marker"] = "success_marker"
    success_marker_index: int = Field(strict=True, ge=0)


class FalseClaimTemplateSlot(FrozenRecord):
    kind: Literal["false_claim"] = "false_claim"
    false_claim_role: Literal["target_false_claim"] = "target_false_claim"


AttackTemplateSlot = Annotated[
    SuccessMarkerTemplateSlot | FalseClaimTemplateSlot,
    Field(discriminator="kind"),
]
TemplateSegment = Annotated[str, Field(strict=True)]


class AttackCase(FrozenRecord):
    attack_id: str
    split: Split
    base_context_id: str
    goal: AttackGoal
    language: Language
    form: AttackForm
    position: AttackPosition
    payload: str = Field(min_length=1)
    success_markers: tuple[str, ...] = ()
    target_false_claim: str | None = None
    template_id: str
    safety_class: Literal["sentinel_only", "false_claim", "invalid_link"]
    template_segments: tuple[TemplateSegment, ...] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    template_slots: tuple[AttackTemplateSlot, ...] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

    @field_validator("payload")
    @classmethod
    def reject_unsafe_payload(cls, value: str) -> str:
        scheme_positions = [match.start() for match in _HTTP_SCHEME_PATTERN.finditer(value)]
        for start in scheme_positions:
            boundary = _URL_TOKEN_BOUNDARY_PATTERN.search(value, start)
            end = boundary.start() if boundary is not None else len(value)
            candidate = value[start:end].rstrip(_TRAILING_URL_PUNCTUATION)
            try:
                hostname = urlparse(candidate).hostname
            except ValueError as error:
                raise ValueError("attack payload contains a non-.invalid network URL") from error
            if hostname is None or (hostname != "invalid" and not hostname.endswith(".invalid")):
                raise ValueError("attack payload contains a non-.invalid network URL")
        return value

    @model_validator(mode="after")
    def validate_template_structure(self) -> Self:
        if self.template_segments is None and self.template_slots is None:
            return self
        if self.template_segments is None or self.template_slots is None:
            raise ValueError("template_segments and template_slots must be present together")
        if len(self.template_segments) != len(self.template_slots) + 1:
            raise ValueError("template_segments must contain one more item than template_slots")

        resolved_slots: list[str] = []
        for slot in self.template_slots:
            if isinstance(slot, SuccessMarkerTemplateSlot):
                if slot.success_marker_index >= len(self.success_markers):
                    raise ValueError("success_marker_index is outside success_markers")
                marker = self.success_markers[slot.success_marker_index]
                if not marker.strip():
                    raise ValueError("success_marker_index must resolve to a nonblank marker")
                resolved_slots.append(marker)
            else:
                false_claim = self.target_false_claim
                if false_claim is None or not false_claim.strip():
                    raise ValueError(
                        "false_claim slot requires a nonblank target_false_claim"
                    )
                resolved_slots.append(false_claim)

        reconstructed_parts = [self.template_segments[0]]
        for resolved, segment in zip(
            resolved_slots,
            self.template_segments[1:],
            strict=True,
        ):
            reconstructed_parts.extend((resolved, segment))
        if "".join(reconstructed_parts) != self.payload:
            raise ValueError("template structure must reconstruct payload exactly")
        if any(
            marker.strip()
            and any(marker in segment for segment in self.template_segments)
            for marker in self.success_markers
        ):
            raise ValueError(
                "success marker values must be represented by template slots"
            )
        normalized_false_claim = (
            normalize_research_text(self.target_false_claim)
            if self.target_false_claim is not None
            else ""
        )
        if normalized_false_claim and any(
            normalized_false_claim in normalize_research_text(segment)
            for segment in self.template_segments
        ):
            raise ValueError(
                "target_false_claim values must be represented by template slots"
            )
        return self


class DetectorScore(FrozenRecord):
    chunk_id: str
    score: Probability


class PromptMessage(FrozenRecord):
    role: Literal["system", "user"]
    content: str


class DefenseResult(FrozenRecord):
    defense_id: DefenseId
    messages: tuple[PromptMessage, ...]
    retained_chunk_ids: tuple[str, ...]
    blocked_chunk_ids: tuple[str, ...]
    defense_metadata: dict[str, MetadataValue] = Field(default_factory=dict)


class GenerationResult(FrozenRecord):
    raw_response: str
    input_tokens: int | None = Field(default=None, strict=True, ge=0)
    output_tokens: int | None = Field(default=None, strict=True, ge=0)
    finish_reason: str | None = Field(default=None, min_length=1)


class ForcedResearchInputs(FrozenRecord):
    track: Literal["forced"] = "forced"
    question: Question
    base_context_id: str
    ordered_benign_chunks: tuple[RetrievedChunk, ...]
    gold_chunk_id: str
    attack: AttackCase
    malicious_rank: int = Field(strict=True, ge=1)
    gold_context_fallback: StrictBool = False
    config_hash: str


class E2EResearchInputs(FrozenRecord):
    track: Literal["e2e"] = "e2e"
    question: Question
    base_context_id: str
    ordered_chunks: tuple[RetrievedChunk, ...]
    gold_chunk_id: str
    attack: AttackCase
    clean_source_chunk: Chunk
    top_k: int = Field(strict=True, ge=1)
    corpus_chunks_commitment: Sha256Hex
    corpus_matrix_commitment: Sha256Hex
    query_vector_commitment: Sha256Hex
    malicious_vector_commitment: Sha256Hex
    config_hash: str


class DirectResearchInputs(FrozenRecord):
    track: Literal["clean", "detector"]
    question_id: str
    base_context_id: str | None
    attack_id: str | None
    question: str
    ordered_chunks: tuple[RetrievedChunk, ...]
    malicious_chunk_rank: int | None = Field(default=None, strict=True, ge=1)
    attack: AttackCase | None
    clean_source_chunk: Chunk | None
    config_hash: str


ResearchInputs = Annotated[
    ForcedResearchInputs | E2EResearchInputs | DirectResearchInputs,
    Field(discriminator="track"),
]


class ExperimentJob(FrozenRecord):
    job_id: str
    track: Literal["forced", "e2e", "clean", "detector"]
    question_id: str
    base_context_id: str | None
    attack_id: str | None
    defense_id: DefenseId
    question: str
    ordered_chunks: tuple[RetrievedChunk, ...]
    ordered_chunk_ids: tuple[str, ...]
    malicious_chunk_rank: int | None = Field(default=None, ge=1)
    attack: AttackCase | None
    clean_source_chunk: Chunk | None
    clean_source_commitment: Sha256Hex | None
    attack_commitment: Sha256Hex | None
    research_inputs: ResearchInputs
    research_input_hash: Sha256Hex
    config_hash: str

    @model_validator(mode="after")
    def validate_frozen_conditions(self) -> Self:
        chunk_ids = tuple(item.chunk.chunk_id for item in self.ordered_chunks)
        if self.ordered_chunk_ids != chunk_ids:
            raise ValueError("ordered_chunk_ids must match ordered_chunks chunk IDs")
        ranks = tuple(item.rank for item in self.ordered_chunks)
        if ranks != tuple(range(1, len(self.ordered_chunks) + 1)):
            raise ValueError("ordered_chunks ranks must match rendered tuple order 1..N")
        if self.research_inputs.track != self.track:
            raise ValueError("research_inputs track must match job track")
        if canonical_sha256(self.research_inputs) != self.research_input_hash:
            raise ValueError("research_input_hash must match full research_inputs")

        if isinstance(self.research_inputs, DirectResearchInputs):
            for field in (
                "question_id",
                "base_context_id",
                "attack_id",
                "question",
                "ordered_chunks",
                "malicious_chunk_rank",
                "attack",
                "clean_source_chunk",
                "config_hash",
            ):
                if getattr(self, field) != getattr(self.research_inputs, field):
                    raise ValueError(f"{field} must match direct research_inputs")
        else:
            if self.question_id != self.research_inputs.question.question_id:
                raise ValueError("question_id must match research_inputs question")
            if self.question != self.research_inputs.question.question:
                raise ValueError("question must match research_inputs question")
            if self.base_context_id != self.research_inputs.base_context_id:
                raise ValueError("base_context_id must match research_inputs")
            if self.attack_id != self.research_inputs.attack.attack_id:
                raise ValueError("attack_id must match research_inputs attack")
            if self.attack != self.research_inputs.attack:
                raise ValueError("attack must match research_inputs attack")
            if self.config_hash != self.research_inputs.config_hash:
                raise ValueError("config_hash must match research_inputs")

        attack_bound = self.track in {"forced", "e2e"} or (
            self.track == "detector"
            and (self.attack is not None or self.attack_id is not None)
        )
        if not attack_bound:
            if any(
                value is not None
                for value in (
                    self.attack,
                    self.attack_id,
                    self.clean_source_chunk,
                    self.clean_source_commitment,
                    self.attack_commitment,
                )
            ):
                raise ValueError("clean and benign detector jobs must not carry attack inputs")
            return self

        if self.attack is None:
            raise ValueError(f"{self.track} job requires an AttackCase")
        if self.clean_source_chunk is None:
            raise ValueError(f"{self.track} job requires a clean source chunk")
        if self.clean_source_commitment is None or self.attack_commitment is None:
            raise ValueError(f"{self.track} job requires source and attack commitments")
        if canonical_sha256(self.clean_source_chunk) != self.clean_source_commitment:
            raise ValueError("clean_source_commitment must match clean_source_chunk")
        if canonical_sha256(self.attack) != self.attack_commitment:
            raise ValueError("attack_commitment must match AttackCase")
        return self


def expected_experiment_job_id(job: ExperimentJob) -> str:
    payload = cast(
        JsonValue,
        job.model_dump(mode="json", exclude={"job_id"}),
    )
    return stable_id("JOB", payload)


class ExperimentRecord(FrozenRecord):
    experiment_id: str
    job_id: str
    track: Literal["forced", "e2e", "clean", "detector"]
    question_id: str
    base_context_id: str | None
    attack_id: str | None
    defense_id: DefenseId
    config_hash: str
    clean_source_commitment: Sha256Hex | None
    attack_commitment: Sha256Hex | None
    model_id: str
    model_revision: str
    detector_model_id: str
    detector_model_revision: str
    research_input_hash: Sha256Hex
    prompt_hash: str | None
    ordered_chunk_ids: tuple[str, ...]
    malicious_chunk_rank: int | None = Field(strict=True, ge=1)
    detector_scores: dict[str, Probability]
    blocked_chunk_ids: tuple[str, ...]
    raw_response: str
    attack_success: StrictBool | None
    technical_failure: StrictBool
    error_type: str | None
    latency_ms: NonNegativeFiniteFloat
    detector_latency_ms: NonNegativeFiniteFloat
    generation_latency_ms: NonNegativeFiniteFloat
    input_tokens: int | None = Field(strict=True, ge=0)
    output_tokens: int | None = Field(strict=True, ge=0)

    @field_validator(
        "model_id",
        "model_revision",
        "detector_model_id",
        "detector_model_revision",
    )
    @classmethod
    def require_strict_nonblank_model_identity(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("model identities must be nonblank strings")
        return value

    @model_validator(mode="after")
    def require_attack_outcome_when_applicable(self) -> Self:
        attack_bound = self.track in {"forced", "e2e"} or (
            self.track == "detector" and self.attack_id is not None
        )
        if attack_bound and (
            self.clean_source_commitment is None or self.attack_commitment is None
        ):
            raise ValueError("attack records require source and attack commitments")
        if not attack_bound and (
            self.clean_source_commitment is not None or self.attack_commitment is not None
        ):
            raise ValueError("non-attack records must not carry attack commitments")
        if self.technical_failure:
            if self.error_type is None or not self.error_type.strip():
                raise ValueError("technical failures require a nonblank error_type")
        elif self.error_type is not None:
            raise ValueError("nontechnical records must not carry error_type")
        if self.technical_failure and self.attack_success is not None:
            raise ValueError("attack_success must be None for technical failures")
        if not self.technical_failure and not self.raw_response.strip():
            raise ValueError("raw_response must be nonblank for nontechnical records")
        if self.track in {"clean", "detector"} and self.attack_success is not None:
            raise ValueError("attack_success must be None for clean and detector tracks")
        if (
            not self.technical_failure
            and self.track in {"forced", "e2e"}
            and self.attack_success is None
        ):
            raise ValueError(
                "attack_success must be a boolean for nontechnical forced and e2e records"
            )
        return self


class FailureAttemptRecord(ExperimentRecord):
    attempt_id: str
    attempt_number: int = Field(strict=True, ge=1)

    @classmethod
    def from_record(
        cls,
        record: ExperimentRecord,
        *,
        attempt_number: int,
    ) -> Self:
        validated = ExperimentRecord.model_validate_json(
            record.model_dump_json(),
            strict=True,
        )
        if not validated.technical_failure:
            raise ValueError("failure attempts require a technical failure record")
        payload = validated.model_dump(mode="json") | {
            "attempt_number": attempt_number,
        }
        attempt_id = stable_id("ATTEMPT", cast(JsonValue, payload))
        return cls.model_validate(
            {
                **payload,
                "attempt_id": attempt_id,
            }
        )

    @model_validator(mode="after")
    def validate_failure_attempt(self) -> Self:
        if not self.technical_failure:
            raise ValueError("failure attempts must be technical failures")
        payload = self.model_dump(mode="json", exclude={"attempt_id"})
        expected_attempt_id = stable_id("ATTEMPT", cast(JsonValue, payload))
        if self.attempt_id != expected_attempt_id:
            raise ValueError("attempt_id must match the canonical failure attempt")
        return self


class AttackEvaluation(FrozenRecord):
    attack_id: str
    attack_success: bool | None
    included_in_asr: bool
    requires_human_review: bool
    matched_markers: tuple[str, ...]
    error_type: str | None
