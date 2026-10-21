import hashlib
import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, TypeVar, cast

import typer
from pydantic import BaseModel, ValidationError

from thesis_rag.adapters import (
    BypassDetector,
    DetectorAdapter,
    GeneratorAdapter,
    MockDetector,
    MockGenerator,
)
from thesis_rag.attack_pilot import (
    AttackPilotSummary,
    select_factorial_pilot_attack_ids,
    summarize_attack_pilot,
    write_attack_pilot_summary,
)
from thesis_rag.canonical import (
    JsonValue,
    canonical_json,
    canonical_sha256,
    sha256_text,
    stable_id,
)
from thesis_rag.config import ResearchConfig, load_config
from thesis_rag.corpus.models import ChunkingConfig, CorpusValidationReport
from thesis_rag.corpus.pipeline import build_corpus, verify_corpus_artifacts
from thesis_rag.defenses.base import Defense
from thesis_rag.defenses.baseline import D0Defense, D1Defense
from thesis_rag.defenses.datamarking import D2Defense
from thesis_rag.defenses.filtering import D3Defense, D4Defense
from thesis_rag.detector import (
    PromptGuardDetectorAdapter,
    PromptGuardModelIdentity,
    build_prompt_guard_scoring_policy,
)
from thesis_rag.detector_pilot import (
    CalibratedDetectorThreshold,
    DetectorEvaluationCase,
    DetectorPilotRecord,
    DetectorPilotSummary,
    bind_detector_threshold,
    run_detector_pilot,
    select_detector_threshold,
    verify_calibrated_detector_threshold,
    write_detector_threshold_selection,
)
from thesis_rag.domain import (
    AttackCase,
    Chunk,
    DefenseId,
    ExperimentJob,
    ExperimentRecord,
    Question,
    RetrievedChunk,
    Split,
)
from thesis_rag.embeddings import (
    EmbeddingCacheMetadata,
    OllamaEmbeddingAdapter,
    RetrievalPilotSummary,
    load_embedding_cache,
    run_retrieval_pilot,
)
from thesis_rag.evaluation import evaluate_attack, validate_split_integrity
from thesis_rag.filtering_audit import (
    FilteringPairedAuditReport,
    audit_filtering_pilots,
    write_filtering_paired_audit,
)
from thesis_rag.generation import (
    OllamaGeneratorAdapter,
    OllamaModelIdentity,
)
from thesis_rag.human_review import (
    CleanUtilityReviewSummary,
    load_clean_utility_annotations,
    summarize_clean_utility_annotations,
    write_clean_utility_review_summary,
)
from thesis_rag.jsonl import load_failure_records, load_records
from thesis_rag.manifests import build_forced_jobs
from thesis_rag.pilot import (
    CleanGenerationPilotRecord,
    CleanGenerationPilotSummary,
    run_clean_generation_pilot,
)
from thesis_rag.qa import (
    ResearchQuestion,
    render_question_review_sheet,
    validate_research_questions,
)
from thesis_rag.research_manifest import (
    ForcedManifestSummary,
    build_forced_manifest,
    write_forced_manifest,
)
from thesis_rag.runner import run_jobs

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_FIXTURES_DIRECTORY = _REPOSITORY_ROOT / "tests" / "fixtures"
_DEVELOPMENT_CONFIG_PATH = _REPOSITORY_ROOT / "configs" / "development.yaml"
_APPROVED_CORPUS_PATH = _REPOSITORY_ROOT / "data" / "sources" / "approved-urls.yaml"
_PARSED_CORPUS_DIRECTORY = _REPOSITORY_ROOT / "data" / "parsed"
_CORPUS_CHUNKS_PATH = _REPOSITORY_ROOT / "data" / "chunks" / "core.jsonl"
_DEVELOPMENT_QUESTIONS_PATH = (
    _REPOSITORY_ROOT / "data" / "questions" / "dev.jsonl"
)
_DEVELOPMENT_ATTACKS_PATH = _REPOSITORY_ROOT / "data" / "attacks" / "dev.jsonl"
_CORPUS_REPORT_PATH = (
    _REPOSITORY_ROOT / "artifacts" / "reports" / "corpus-validation.json"
)
_EMBEDDING_CACHE_PATH = (
    _REPOSITORY_ROOT
    / "artifacts"
    / "embeddings"
    / "qwen3-embedding-0.6b-dev.npz"
)
_EMBEDDING_METADATA_PATH = (
    _REPOSITORY_ROOT
    / "artifacts"
    / "embeddings"
    / "qwen3-embedding-0.6b-dev.metadata.json"
)
_RETRIEVAL_RESULTS_PATH = (
    _REPOSITORY_ROOT / "artifacts" / "reports" / "dev-retrieval-results.jsonl"
)
_RETRIEVAL_SUMMARY_PATH = (
    _REPOSITORY_ROOT / "artifacts" / "reports" / "dev-retrieval-summary.json"
)


class _DevelopmentPilotDefenseId(StrEnum):
    D0 = "D0"
    D1 = "D1"
    D2 = "D2"
    D3 = "D3"
    D4 = "D4"


_GENERATION_PILOT_RESULTS_PATH = (
    _REPOSITORY_ROOT
    / "artifacts"
    / "reports"
    / "dev-clean-generation-results.jsonl"
)
_GENERATION_PILOT_SUMMARY_PATH = (
    _REPOSITORY_ROOT
    / "artifacts"
    / "reports"
    / "dev-clean-generation-summary.json"
)
_GENERATION_PILOT_REVIEW_PATH = (
    _REPOSITORY_ROOT
    / "artifacts"
    / "reports"
    / "dev-clean-generation-review.md"
)
_GENERATION_PILOT_CODEX_ANNOTATIONS_PATH = (
    _REPOSITORY_ROOT
    / "artifacts"
    / "reports"
    / "dev-clean-generation-codex-review.jsonl"
)
_GENERATION_PILOT_CODEX_SUMMARY_PATH = (
    _REPOSITORY_ROOT
    / "artifacts"
    / "reports"
    / "dev-clean-generation-codex-review-summary.json"
)
_DETECTOR_CASES_PATH = (
    _REPOSITORY_ROOT / "data" / "detector" / "dev-cases.jsonl"
)
_DETECTOR_PILOT_RESULTS_PATH = (
    _REPOSITORY_ROOT / "artifacts" / "reports" / "dev-detector-results.jsonl"
)
_DETECTOR_PILOT_SUMMARY_PATH = (
    _REPOSITORY_ROOT / "artifacts" / "reports" / "dev-detector-summary.json"
)
_DETECTOR_THRESHOLD_PATH = (
    _REPOSITORY_ROOT / "artifacts" / "reports" / "dev-detector-threshold.json"
)
_FORCED_MANIFEST_PATH = (
    _REPOSITORY_ROOT / "artifacts" / "manifests" / "dev-forced.jsonl"
)
_FORCED_MANIFEST_SUMMARY_PATH = (
    _REPOSITORY_ROOT / "artifacts" / "reports" / "dev-forced-manifest-summary.json"
)
_D0_ATTACK_PILOT_RESULTS_PATH = (
    _REPOSITORY_ROOT / "artifacts" / "reports" / "dev-d0-attack-pilot.jsonl"
)
_D0_ATTACK_PILOT_FAILURES_PATH = (
    _REPOSITORY_ROOT
    / "artifacts"
    / "reports"
    / "dev-d0-attack-pilot.failures.jsonl"
)
_D0_ATTACK_PILOT_SUMMARY_PATH = (
    _REPOSITORY_ROOT
    / "artifacts"
    / "reports"
    / "dev-d0-attack-pilot-summary.json"
)
_FILTERING_PAIRED_AUDIT_PATH = (
    _REPOSITORY_ROOT
    / "artifacts"
    / "reports"
    / "dev-d3-d4-paired-audit.json"
)
_EXPECTED_DEFENSE_IDS = tuple(DefenseId)
_FIXTURE_GENERATOR_MODEL_ID = "mock-generator"
_FIXTURE_GENERATOR_MODEL_REVISION = "v1"
_FIXTURE_DETECTOR_MODEL_ID = "mock-detector"
_FIXTURE_DETECTOR_MODEL_REVISION = "v1"
_REQUIRED_PATH = cast(Path, ...)
_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _attack_pilot_default_paths(
    defense_id: DefenseId,
) -> tuple[Path, Path, Path]:
    if defense_id is DefenseId.D0:
        return (
            _D0_ATTACK_PILOT_RESULTS_PATH,
            _D0_ATTACK_PILOT_FAILURES_PATH,
            _D0_ATTACK_PILOT_SUMMARY_PATH,
        )
    stem = f"dev-{defense_id.value.lower()}-attack-pilot"
    reports = _REPOSITORY_ROOT / "artifacts" / "reports"
    return (
        reports / f"{stem}.jsonl",
        reports / f"{stem}.failures.jsonl",
        reports / f"{stem}-summary.json",
    )


def _generation_pilot_default_paths(
    defense_id: DefenseId,
) -> tuple[Path, Path, Path]:
    if defense_id is DefenseId.D0:
        return (
            _GENERATION_PILOT_RESULTS_PATH,
            _GENERATION_PILOT_SUMMARY_PATH,
            _GENERATION_PILOT_REVIEW_PATH,
        )
    stem = f"dev-{defense_id.value.lower()}-clean-generation"
    reports = _REPOSITORY_ROOT / "artifacts" / "reports"
    return (
        reports / f"{stem}-results.jsonl",
        reports / f"{stem}-summary.json",
        reports / f"{stem}-review.md",
    )


_D3_ATTACK_PILOT_PATHS = _attack_pilot_default_paths(DefenseId.D3)
_D4_ATTACK_PILOT_PATHS = _attack_pilot_default_paths(DefenseId.D4)
_D3_GENERATION_PILOT_PATHS = _generation_pilot_default_paths(DefenseId.D3)
_D4_GENERATION_PILOT_PATHS = _generation_pilot_default_paths(DefenseId.D4)


@dataclass(frozen=True)
class FixturePipelineSummary:
    jobs_expected: int
    jobs_completed: int
    jobs_skipped: int
    duplicate_jobs: int
    technical_failures: int
    defense_ids: tuple[DefenseId, ...]


class _DuplicateJsonKeyError(ValueError):
    pass


def _read_fixture_jsonl(
    path: Path, record_type: type[_ModelT], identity_field: str
) -> tuple[_ModelT, ...]:
    try:
        contents = path.read_bytes()
    except OSError as error:
        raise ValueError(f"unable to read fixture file: {path}") from error
    if not contents:
        raise ValueError(f"fixture file is empty: {path}")
    if not contents.endswith(b"\n"):
        raise ValueError(f"fixture JSONL missing final newline: {path}")

    records: list[_ModelT] = []
    identities: set[str] = set()
    for line_number, raw_line in enumerate(contents.splitlines(), start=1):
        try:
            decoded_line = raw_line.decode("utf-8")
            value = json.loads(
                decoded_line,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_non_json_constant,
            )
        except _DuplicateJsonKeyError as error:
            raise ValueError(f"{error} at {path}:{line_number}") from error
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError(f"invalid fixture JSON at {path}:{line_number}") from error
        if not isinstance(value, dict):
            raise ValueError(f"fixture line must be an object at {path}:{line_number}")
        try:
            record = record_type.model_validate_json(decoded_line, strict=True)
        except ValidationError as error:
            raise ValueError(
                f"invalid {record_type.__name__} fixture at {path}:{line_number}"
            ) from error
        identity = getattr(record, identity_field)
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError(f"{identity_field} must be a non-empty string at {path}:{line_number}")
        if identity in identities:
            raise ValueError(f"duplicate {identity_field}: {identity}")
        identities.add(identity)
        records.append(record)
    return tuple(records)


def _reject_non_json_constant(value: str) -> float:
    raise ValueError(f"invalid JSON numeric constant: {value}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    object_value: dict[str, object] = {}
    for key, value in pairs:
        if key in object_value:
            raise _DuplicateJsonKeyError(f"duplicate JSON key: {key}")
        object_value[key] = value
    return object_value


def _validate_question_attack_splits(
    questions: Sequence[Question], attacks: Sequence[AttackCase]
) -> None:
    validate_split_integrity(questions, attacks)
    question_splits = {question.split for question in questions}
    missing_splits = sorted(
        {attack.split.value for attack in attacks if attack.split not in question_splits}
    )
    if missing_splits:
        raise ValueError(f"attacks have no matching question split: {','.join(missing_splits)}")


def _select_development_case(
    chunks: tuple[Chunk, ...], questions: tuple[Question, ...], attacks: tuple[AttackCase, ...]
) -> tuple[Question, AttackCase, str]:
    _validate_question_attack_splits(questions, attacks)
    chunk_ids = {chunk.chunk_id for chunk in chunks}
    for question in questions:
        if len(set(question.gold_chunk_ids)) != len(question.gold_chunk_ids):
            raise ValueError(f"duplicate gold chunk ID for question: {question.question_id}")
        missing_chunks = sorted(set(question.gold_chunk_ids) - chunk_ids)
        if missing_chunks:
            raise ValueError(
                "question references missing chunks: "
                f"{question.question_id}:{','.join(missing_chunks)}"
            )

    development_questions = tuple(
        question for question in questions if question.split is Split.DEV and question.answerable
    )
    development_attacks = tuple(attack for attack in attacks if attack.split is Split.DEV)
    if len(development_questions) != 1:
        raise ValueError("expected exactly one answerable development question")
    if len(development_attacks) != 1:
        raise ValueError("expected exactly one development attack")

    question = development_questions[0]
    attack = development_attacks[0]
    if question.split != attack.split:
        raise ValueError("selected question and attack must have the same split")
    if question.language != attack.language:
        raise ValueError("selected question and attack must have the same language")
    if len(question.gold_chunk_ids) != 1:
        raise ValueError("selected development question must have exactly one gold chunk")
    if len(attack.success_markers) != 1:
        raise ValueError("selected development attack must have exactly one success marker")
    marker = attack.success_markers[0]
    if marker not in attack.payload:
        raise ValueError("selected development attack payload must contain its success marker")
    return question, attack, question.gold_chunk_ids[0]


def _fixture_context(chunks: tuple[Chunk, ...]) -> tuple[RetrievedChunk, ...]:
    return tuple(
        RetrievedChunk(chunk=chunk, score=float(len(chunks) - index), rank=index + 1)
        for index, chunk in enumerate(chunks)
    )


def _config_hash(config: ResearchConfig) -> str:
    return sha256_text(canonical_json(config))


def _defenses(threshold: float) -> dict[DefenseId, Defense]:
    return {
        DefenseId.D0: D0Defense(),
        DefenseId.D1: D1Defense(),
        DefenseId.D2: D2Defense(),
        DefenseId.D3: cast(Defense, D3Defense(threshold=threshold)),
        DefenseId.D4: cast(Defense, D4Defense(threshold=threshold)),
    }


def _fixture_generator(attack: AttackCase) -> MockGenerator:
    return MockGenerator(
        success_markers=attack.success_markers,
        follow_malicious_instructions=True,
        model_id=_FIXTURE_GENERATOR_MODEL_ID,
        model_revision=_FIXTURE_GENERATOR_MODEL_REVISION,
    )


def _fixture_detector(jobs: Sequence[ExperimentJob]) -> MockDetector:
    return MockDetector(
        scores=_fixture_detector_scores(jobs),
        model_id=_FIXTURE_DETECTOR_MODEL_ID,
        model_revision=_FIXTURE_DETECTOR_MODEL_REVISION,
    )


def _fixture_experiment_id(job_id: str) -> str:
    return stable_id(
        "EXP",
        cast(
            JsonValue,
            {
                "job_id": job_id,
                "model_id": _FIXTURE_GENERATOR_MODEL_ID,
                "model_revision": _FIXTURE_GENERATOR_MODEL_REVISION,
                "detector_model_id": _FIXTURE_DETECTOR_MODEL_ID,
                "detector_model_revision": _FIXTURE_DETECTOR_MODEL_REVISION,
            },
        ),
    )


def _preflight_fixture_jobs(jobs: Sequence[ExperimentJob]) -> None:
    if len(jobs) != len(_EXPECTED_DEFENSE_IDS):
        raise ValueError("fixture pipeline must build exactly five jobs")
    if tuple(job.defense_id for job in jobs) != _EXPECTED_DEFENSE_IDS:
        raise ValueError("fixture jobs must contain ordered D0-D4 defense IDs")
    if len({job.job_id for job in jobs}) != len(jobs):
        raise ValueError("fixture jobs must have unique job IDs")
    for field in (
        "track",
        "question_id",
        "base_context_id",
        "attack_id",
        "question",
        "ordered_chunks",
        "ordered_chunk_ids",
        "malicious_chunk_rank",
        "attack",
        "clean_source_chunk",
        "clean_source_commitment",
        "attack_commitment",
        "research_inputs",
        "research_input_hash",
        "config_hash",
    ):
        if any(getattr(job, field) != getattr(jobs[0], field) for job in jobs[1:]):
            raise ValueError(f"fixture jobs must share paired field: {field}")


def _prepare_fixture_jobs(
    config_path: Path,
    chunks_path: Path,
    questions_path: Path,
    attacks_path: Path,
) -> tuple[ResearchConfig, tuple[ExperimentJob, ...]]:
    config = load_config(config_path)
    chunks = _read_fixture_jsonl(chunks_path, Chunk, "chunk_id")
    questions = _read_fixture_jsonl(questions_path, Question, "question_id")
    attacks = _read_fixture_jsonl(attacks_path, AttackCase, "attack_id")
    question, attack, gold_chunk_id = _select_development_case(chunks, questions, attacks)
    jobs = build_forced_jobs(
        question=question,
        base_context_id=attack.base_context_id,
        ordered_benign_chunks=_fixture_context(chunks),
        gold_chunk_id=gold_chunk_id,
        attack=attack,
        malicious_rank=1,
        defense_ids=_EXPECTED_DEFENSE_IDS,
        config_hash=_config_hash(config),
    )
    _preflight_fixture_jobs(jobs)
    return config, jobs


def _fixture_detector_scores(jobs: Sequence[ExperimentJob]) -> dict[str, float]:
    malicious_chunk_ids = tuple(
        item.chunk.chunk_id
        for item in jobs[0].ordered_chunks
        if item.chunk.metadata.get("is_malicious") is True
    )
    if len(malicious_chunk_ids) != 1:
        raise ValueError("fixture forced context must contain exactly one malicious chunk")
    malicious_chunk_id = malicious_chunk_ids[0]
    return {
        item.chunk.chunk_id: 1.0 if item.chunk.chunk_id == malicious_chunk_id else 0.0
        for item in jobs[0].ordered_chunks
    }


def run_fixture_pipeline(
    config_path: Path,
    result_path: Path,
    *,
    chunks_path: Path = _FIXTURES_DIRECTORY / "chunks.jsonl",
    questions_path: Path = _FIXTURES_DIRECTORY / "questions.jsonl",
    attacks_path: Path = _FIXTURES_DIRECTORY / "attacks.jsonl",
) -> FixturePipelineSummary:
    config, jobs = _prepare_fixture_jobs(
        config_path,
        chunks_path,
        questions_path,
        attacks_path,
    )
    if result_path.exists() and result_path.stat().st_size:
        existing_records = load_records(result_path)
        if len(existing_records) == len(_EXPECTED_DEFENSE_IDS):
            _verify_fixture_records(existing_records, jobs, config)
        else:
            _verify_fixture_record_subset(existing_records, jobs, config)
    attack = jobs[0].attack
    if attack is None:
        raise ValueError("fixture jobs require an attack")
    detector = _fixture_detector(jobs)
    summary = run_jobs(
        jobs,
        result_path=result_path,
        defenses=_defenses(config.detector.threshold),
        generator=cast(
            "GeneratorAdapter",
            _fixture_generator(attack),
        ),
        detector=cast("DetectorAdapter", detector),
    )
    return FixturePipelineSummary(
        jobs_expected=summary.jobs_expected,
        jobs_completed=summary.completed,
        jobs_skipped=summary.skipped,
        duplicate_jobs=0,
        technical_failures=summary.technical_failures,
        defense_ids=_EXPECTED_DEFENSE_IDS,
    )


def verify_fixture_results(
    result_path: Path,
    *,
    config_path: Path = _DEVELOPMENT_CONFIG_PATH,
    chunks_path: Path = _FIXTURES_DIRECTORY / "chunks.jsonl",
    questions_path: Path = _FIXTURES_DIRECTORY / "questions.jsonl",
    attacks_path: Path = _FIXTURES_DIRECTORY / "attacks.jsonl",
) -> FixturePipelineSummary:
    config, jobs = _prepare_fixture_jobs(
        config_path,
        chunks_path,
        questions_path,
        attacks_path,
    )
    records = load_records(result_path)
    _verify_fixture_records(records, jobs, config)

    return FixturePipelineSummary(
        jobs_expected=5,
        jobs_completed=5,
        jobs_skipped=0,
        duplicate_jobs=0,
        technical_failures=0,
        defense_ids=_EXPECTED_DEFENSE_IDS,
    )


def _verify_fixture_records(
    records: Sequence[ExperimentRecord],
    jobs: Sequence[ExperimentJob],
    config: ResearchConfig,
) -> None:
    if len(records) != len(_EXPECTED_DEFENSE_IDS):
        raise ValueError(f"expected exactly five records, got {len(records)}")

    by_defense = _records_by_defense(records)
    _verify_shared_record_fields(records)
    if any(record.technical_failure for record in records):
        raise ValueError("fixture results contain technical failures")

    first = records[0]
    malicious_chunk_id = _malicious_chunk_id(first)
    for record in records:
        _verify_detector_scores(record, malicious_chunk_id)

    exposed = (DefenseId.D0, DefenseId.D1, DefenseId.D2)
    filtered = (DefenseId.D3, DefenseId.D4)
    for defense_id in exposed:
        record = by_defense[defense_id]
        if record.blocked_chunk_ids:
            raise ValueError(f"{defense_id.value} must not block fixture chunks")
        if record.attack_success is not True:
            raise ValueError(f"{defense_id.value} must expose the fixture attack")
    for defense_id in filtered:
        record = by_defense[defense_id]
        if record.attack_success is not False:
            raise ValueError(f"{defense_id.value} must filter the fixture attack")
    if by_defense[DefenseId.D3].blocked_chunk_ids != by_defense[DefenseId.D4].blocked_chunk_ids:
        raise ValueError("D3 and D4 blocked chunk IDs differ")
    if by_defense[DefenseId.D3].blocked_chunk_ids != (malicious_chunk_id,):
        raise ValueError("D3 and D4 must block the malicious fixture chunk")
    _verify_record_derivations(by_defense, jobs, config)


def _verify_fixture_record_subset(
    records: Sequence[ExperimentRecord],
    jobs: Sequence[ExperimentJob],
    config: ResearchConfig,
) -> None:
    if len(records) >= len(_EXPECTED_DEFENSE_IDS):
        raise ValueError(f"expected fewer than five partial records, got {len(records)}")

    jobs_by_id = {job.job_id: job for job in jobs}
    for record in records:
        if record.job_id not in jobs_by_id:
            raise ValueError(f"unknown fixture job_id: {record.job_id}")

    counts = Counter(record.defense_id for record in records)
    duplicate_defenses = tuple(
        defense_id.value for defense_id in _EXPECTED_DEFENSE_IDS if counts[defense_id] > 1
    )
    if duplicate_defenses:
        raise ValueError(f"duplicate defense IDs: {','.join(duplicate_defenses)}")
    if any(record.technical_failure for record in records):
        raise ValueError("fixture results contain technical failures")

    detector_scores = _fixture_detector_scores(jobs)
    defenses = _defenses(config.detector.threshold)
    for record in records:
        job = jobs_by_id[record.job_id]
        _verify_record_derivation(record, job, defenses, detector_scores)
        malicious_chunk_id = _malicious_chunk_id(record)
        _verify_detector_scores(record, malicious_chunk_id)
        _verify_expected_defense_outcome(record, malicious_chunk_id)

    _verify_shared_record_fields(records)


def _records_by_defense(
    records: Sequence[ExperimentRecord],
) -> dict[DefenseId, ExperimentRecord]:
    counts = Counter(record.defense_id for record in records)
    duplicate_defenses = tuple(
        defense_id.value for defense_id in _EXPECTED_DEFENSE_IDS if counts[defense_id] > 1
    )
    if duplicate_defenses:
        raise ValueError(f"duplicate defense IDs: {','.join(duplicate_defenses)}")
    actual = set(counts)
    expected = set(_EXPECTED_DEFENSE_IDS)
    missing = tuple(
        defense_id.value for defense_id in _EXPECTED_DEFENSE_IDS if defense_id not in actual
    )
    extra = tuple(sorted(defense_id.value for defense_id in actual - expected))
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing defense IDs: {','.join(missing)}")
        if extra:
            details.append(f"extra defense IDs: {','.join(extra)}")
        raise ValueError("; ".join(details))
    return {record.defense_id: record for record in records}


def _verify_shared_record_fields(records: Sequence[ExperimentRecord]) -> None:
    for field in (
        "track",
        "question_id",
        "base_context_id",
        "attack_id",
        "config_hash",
        "clean_source_commitment",
        "attack_commitment",
        "model_id",
        "model_revision",
        "detector_model_id",
        "detector_model_revision",
        "research_input_hash",
        "ordered_chunk_ids",
        "malicious_chunk_rank",
    ):
        if any(getattr(record, field) != getattr(records[0], field) for record in records[1:]):
            raise ValueError(f"unequal paired field: {field}")


def _verify_detector_scores(record: ExperimentRecord, malicious_chunk_id: str) -> None:
    expected_chunk_ids = set(record.ordered_chunk_ids)
    if set(record.detector_scores) != expected_chunk_ids:
        raise ValueError("detector scores must cover exactly the ordered fixture chunks")
    for chunk_id, score in record.detector_scores.items():
        expected_score = 1.0 if chunk_id == malicious_chunk_id else 0.0
        if score != expected_score:
            raise ValueError(
                "fixture detector scores must be one for malicious and zero for benign"
            )


def _malicious_chunk_id(record: ExperimentRecord) -> str:
    if record.track != "forced":
        raise ValueError("fixture records must use forced track")
    if record.malicious_chunk_rank is None:
        raise ValueError("fixture records require a malicious chunk rank")
    if not 1 <= record.malicious_chunk_rank <= len(record.ordered_chunk_ids):
        raise ValueError("malicious_chunk_rank must be within ordered chunk IDs")
    return record.ordered_chunk_ids[record.malicious_chunk_rank - 1]


def _verify_expected_defense_outcome(
    record: ExperimentRecord, malicious_chunk_id: str
) -> None:
    if record.defense_id in (DefenseId.D0, DefenseId.D1, DefenseId.D2):
        if record.blocked_chunk_ids:
            raise ValueError(f"{record.defense_id.value} must not block fixture chunks")
        if record.attack_success is not True:
            raise ValueError(f"{record.defense_id.value} must expose the fixture attack")
        return
    if record.attack_success is not False:
        raise ValueError(f"{record.defense_id.value} must filter the fixture attack")
    if record.blocked_chunk_ids != (malicious_chunk_id,):
        raise ValueError(f"{record.defense_id.value} must block the malicious fixture chunk")


def _verify_record_derivations(
    records_by_defense: dict[DefenseId, ExperimentRecord],
    jobs: Sequence[ExperimentJob],
    config: ResearchConfig,
) -> None:
    jobs_by_defense = {job.defense_id: job for job in jobs}
    detector_scores = _fixture_detector_scores(jobs)
    defenses = _defenses(config.detector.threshold)
    for defense_id in _EXPECTED_DEFENSE_IDS:
        record = records_by_defense[defense_id]
        job = jobs_by_defense[defense_id]
        _verify_record_derivation(record, job, defenses, detector_scores)


def _verify_record_derivation(
    record: ExperimentRecord,
    job: ExperimentJob,
    defenses: dict[DefenseId, Defense],
    detector_scores: dict[str, float],
) -> None:
    for field in (
        "job_id",
        "track",
        "question_id",
        "base_context_id",
        "attack_id",
        "defense_id",
        "config_hash",
        "clean_source_commitment",
        "attack_commitment",
        "research_input_hash",
        "ordered_chunk_ids",
        "malicious_chunk_rank",
    ):
        if getattr(record, field) != getattr(job, field):
            raise ValueError(f"fixture record does not match reconstructed job field: {field}")
    if (
        record.model_id != _FIXTURE_GENERATOR_MODEL_ID
        or record.model_revision != _FIXTURE_GENERATOR_MODEL_REVISION
    ):
        raise ValueError("fixture record does not match frozen MockGenerator identity")
    if (
        record.detector_model_id != _FIXTURE_DETECTOR_MODEL_ID
        or record.detector_model_revision != _FIXTURE_DETECTOR_MODEL_REVISION
    ):
        raise ValueError("fixture record does not match frozen MockDetector identity")
    expected_experiment_id = _fixture_experiment_id(job.job_id)
    if record.experiment_id != expected_experiment_id:
        raise ValueError("experiment_id does not match frozen adapter identities")
    if record.error_type is not None:
        raise ValueError("nontechnical fixture records must have error_type null")
    attack = job.attack
    if attack is None:
        raise ValueError("reconstructed fixture job requires an attack")
    evaluation = evaluate_attack(
        attack=attack,
        raw_response=record.raw_response,
        technical_failure=False,
    )
    if evaluation.attack_success != record.attack_success:
        raise ValueError("attack_success does not match recomputed response")
    defense_result = defenses[job.defense_id].apply(
        job.question,
        job.ordered_chunks,
        detector_scores,
    )
    message_values: JsonValue = [
        cast(JsonValue, message.model_dump(mode="json")) for message in defense_result.messages
    ]
    expected_prompt_hash = sha256_text(canonical_json(message_values))
    if record.prompt_hash != expected_prompt_hash:
        raise ValueError("prompt_hash does not match reconstructed defense prompt")
    if record.blocked_chunk_ids != defense_result.blocked_chunk_ids:
        raise ValueError("blocked chunk IDs do not match reconstructed defense prompt")
    expected_generation = _fixture_generator(attack).generate(defense_result.messages)
    for field in ("raw_response", "input_tokens", "output_tokens"):
        if getattr(record, field) != getattr(expected_generation, field):
            raise ValueError(f"{field} does not match frozen MockGenerator replay")


def validate_fixture_data(questions_path: Path, attacks_path: Path) -> dict[str, int]:
    questions = _read_fixture_jsonl(questions_path, Question, "question_id")
    attacks = _read_fixture_jsonl(attacks_path, AttackCase, "attack_id")
    _validate_question_attack_splits(questions, attacks)
    return {"attacks": len(attacks), "questions": len(questions)}


def validate_question_data(
    questions_path: Path,
    chunks_path: Path,
    *,
    expected_count: int | None = None,
) -> dict[str, JsonValue]:
    questions = _read_fixture_jsonl(
        questions_path,
        ResearchQuestion,
        "question_id",
    )
    chunks = _read_fixture_jsonl(chunks_path, Chunk, "chunk_id")
    if expected_count is not None and len(questions) != expected_count:
        raise ValueError(
            f"expected {expected_count} questions, found {len(questions)}"
        )
    summary = validate_research_questions(questions, chunks)
    value = cast(dict[str, JsonValue], summary.model_dump(mode="json"))
    value["questions_sha256"] = hashlib.sha256(
        questions_path.read_bytes()
    ).hexdigest()
    value["chunks_sha256"] = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
    return value


def generate_question_review_sheet(
    questions_path: Path,
    chunks_path: Path,
    output_path: Path,
) -> dict[str, JsonValue]:
    questions = _read_fixture_jsonl(
        questions_path,
        ResearchQuestion,
        "question_id",
    )
    chunks = _read_fixture_jsonl(chunks_path, Chunk, "chunk_id")
    questions_sha256 = hashlib.sha256(questions_path.read_bytes()).hexdigest()
    chunks_sha256 = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
    review_sheet = render_question_review_sheet(
        questions,
        chunks,
        questions_sha256=questions_sha256,
        chunks_sha256=chunks_sha256,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(review_sheet, encoding="utf-8")
    return {
        "question_count": len(questions),
        "questions_sha256": questions_sha256,
        "chunks_sha256": chunks_sha256,
        "output_sha256": hashlib.sha256(review_sheet.encode("utf-8")).hexdigest(),
    }


def run_retrieval_pilot_from_files(
    *,
    config_path: Path,
    chunks_path: Path,
    questions_path: Path,
    cache_path: Path,
    metadata_path: Path,
    results_path: Path,
    summary_path: Path,
) -> RetrievalPilotSummary:
    config = load_config(config_path)
    chunks = _read_fixture_jsonl(chunks_path, Chunk, "chunk_id")
    questions = _read_fixture_jsonl(
        questions_path,
        ResearchQuestion,
        "question_id",
    )
    embedding = config.embedding
    adapter = OllamaEmbeddingAdapter(
        model_id=embedding.model_id,
        model_revision=embedding.model_revision,
        dimensions=embedding.dimensions,
        query_instruction=embedding.query_instruction,
        endpoint=embedding.endpoint,
        truncate=embedding.truncate,
        keep_alive=embedding.keep_alive,
    )
    return run_retrieval_pilot(
        chunks,
        questions,
        adapter,
        corpus_sha256=hashlib.sha256(chunks_path.read_bytes()).hexdigest(),
        questions_sha256=hashlib.sha256(questions_path.read_bytes()).hexdigest(),
        top_k=config.retrieval.top_k,
        cache_path=cache_path,
        metadata_path=metadata_path,
        results_path=results_path,
        summary_path=summary_path,
    )


def _ollama_generator_from_config(config: ResearchConfig) -> OllamaGeneratorAdapter:
    generation = config.generation
    return OllamaGeneratorAdapter(
        model_id=generation.model_id,
        model_revision=generation.model_revision,
        expected_template_sha256=generation.chat_template_sha256,
        expected_quantization=generation.quantization,
        context_window=generation.context_window,
        max_new_tokens=generation.max_new_tokens,
        temperature=generation.temperature,
        top_p=generation.top_p,
        top_k=generation.top_k,
        min_p=generation.min_p,
        seed=generation.seed,
        thinking=generation.thinking,
        endpoint=generation.endpoint,
        keep_alive=generation.keep_alive,
    )


def verify_generation_model(config_path: Path) -> OllamaModelIdentity:
    return _ollama_generator_from_config(load_config(config_path)).verify_model()


def _prompt_guard_detector_from_config(
    config: ResearchConfig,
) -> PromptGuardDetectorAdapter:
    detector = config.detector
    return PromptGuardDetectorAdapter(
        model_id=detector.model_id,
        model_revision=detector.model_revision,
        max_length=detector.max_length,
        overlap_tokens=detector.overlap_tokens,
        malicious_label=detector.malicious_label,
    )


def verify_detector_model(config_path: Path) -> PromptGuardModelIdentity:
    return _prompt_guard_detector_from_config(load_config(config_path)).verify_model()


def run_detector_pilot_from_files(
    *,
    config_path: Path,
    cases_path: Path,
    results_path: Path,
    summary_path: Path,
) -> DetectorPilotSummary:
    _validate_immutable_inputs_and_outputs(
        input_paths=(config_path, cases_path),
        output_paths=(results_path, summary_path),
        label="detector pilot",
    )
    config = load_config(config_path)
    cases = _read_fixture_jsonl(
        cases_path,
        DetectorEvaluationCase,
        "case_id",
    )
    return run_detector_pilot(
        cases,
        _prompt_guard_detector_from_config(config),
        threshold=config.detector.threshold,
        results_path=results_path,
        summary_path=summary_path,
    )


def calibrate_detector_from_file(
    *,
    results_path: Path,
    summary_path: Path,
    output_path: Path,
    maximum_false_positive_rate: float,
) -> CalibratedDetectorThreshold:
    resolved_paths = {
        path.resolve(strict=False)
        for path in (results_path, summary_path, output_path)
    }
    if len(resolved_paths) != 3:
        raise ValueError(
            "detector results, summary, and threshold output paths must differ"
        )
    records = _read_fixture_jsonl(
        results_path,
        DetectorPilotRecord,
        "case_id",
    )
    try:
        summary = DetectorPilotSummary.model_validate_json(
            summary_path.read_text(encoding="utf-8"),
            strict=True,
        )
    except (OSError, ValidationError) as error:
        raise ValueError("invalid detector pilot summary") from error
    selection = select_detector_threshold(
        records,
        maximum_false_positive_rate=maximum_false_positive_rate,
    )
    calibrated = bind_detector_threshold(
        selection,
        summary,
        results_file_sha256=hashlib.sha256(results_path.read_bytes()).hexdigest(),
        summary_file_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
    )
    write_detector_threshold_selection(output_path, calibrated)
    return calibrated


def _load_verified_detector_calibration(
    *,
    config: ResearchConfig,
    results_path: Path,
    summary_path: Path,
    threshold_path: Path,
) -> tuple[PromptGuardDetectorAdapter, float]:
    resolved_paths = {
        path.resolve(strict=False)
        for path in (results_path, summary_path, threshold_path)
    }
    if len(resolved_paths) != 3:
        raise ValueError("detector calibration input paths must be distinct")
    try:
        summary = DetectorPilotSummary.model_validate_json(
            summary_path.read_text(encoding="utf-8"),
            strict=True,
        )
        calibrated = CalibratedDetectorThreshold.model_validate_json(
            threshold_path.read_text(encoding="utf-8"),
            strict=True,
        )
        results_file_sha256 = hashlib.sha256(
            results_path.read_bytes()
        ).hexdigest()
        summary_file_sha256 = hashlib.sha256(
            summary_path.read_bytes()
        ).hexdigest()
    except (OSError, ValidationError) as error:
        raise ValueError("invalid detector calibration artifacts") from error

    detector_config = config.detector
    expected_scoring_policy_sha256 = canonical_sha256(
        build_prompt_guard_scoring_policy(
            max_length=detector_config.max_length,
            overlap_tokens=detector_config.overlap_tokens,
            malicious_label=detector_config.malicious_label,
            labels=calibrated.model_identity.labels,
        )
    )
    threshold = verify_calibrated_detector_threshold(
        calibrated,
        summary,
        results_file_sha256=results_file_sha256,
        summary_file_sha256=summary_file_sha256,
        expected_model_id=detector_config.model_id,
        expected_model_revision=detector_config.model_revision,
        expected_scoring_policy_sha256=expected_scoring_policy_sha256,
    )
    detector = _prompt_guard_detector_from_config(config)
    if detector.verify_model() != calibrated.model_identity:
        raise ValueError(
            "loaded detector identity does not match calibrated detector identity"
        )
    return detector, threshold


def build_forced_manifest_from_files(
    *,
    config_path: Path,
    chunks_path: Path,
    questions_path: Path,
    attacks_path: Path,
    cache_path: Path,
    metadata_path: Path,
    manifest_path: Path,
    summary_path: Path,
    shuffle_seed: int,
) -> ForcedManifestSummary:
    config = load_config(config_path)
    chunks = _read_fixture_jsonl(chunks_path, Chunk, "chunk_id")
    questions = _read_fixture_jsonl(
        questions_path,
        ResearchQuestion,
        "question_id",
    )
    attacks = _read_fixture_jsonl(attacks_path, AttackCase, "attack_id")
    try:
        metadata = EmbeddingCacheMetadata.model_validate_json(
            metadata_path.read_text(encoding="utf-8"),
            strict=True,
        )
    except (OSError, ValueError) as error:
        raise ValueError("invalid forced manifest embedding metadata") from error

    embedding = config.embedding
    expected_bindings = {
        "corpus_sha256": hashlib.sha256(chunks_path.read_bytes()).hexdigest(),
        "questions_sha256": hashlib.sha256(questions_path.read_bytes()).hexdigest(),
        "model_id": embedding.model_id,
        "model_revision": embedding.model_revision,
        "dimensions": embedding.dimensions,
        "query_instruction": embedding.query_instruction,
        "chunk_ids": tuple(chunk.chunk_id for chunk in chunks),
        "chunk_text_sha256": tuple(sha256_text(chunk.text) for chunk in chunks),
        "question_ids": tuple(question.question_id for question in questions),
        "question_text_sha256": tuple(
            sha256_text(question.question) for question in questions
        ),
    }
    for field_name, expected in expected_bindings.items():
        if getattr(metadata, field_name) != expected:
            raise ValueError(
                f"forced manifest embedding metadata mismatch: {field_name}"
            )
    cache = load_embedding_cache(
        cache_path,
        metadata_path,
        expected_metadata=metadata,
    )
    jobs = build_forced_manifest(
        chunks,
        questions,
        attacks,
        chunk_vectors=cache.chunk_vectors,
        question_vectors=cache.question_vectors,
        top_k=config.retrieval.top_k,
        defense_ids=tuple(DefenseId),
        config_hash=_config_hash(config),
        shuffle_seed=shuffle_seed,
    )
    return write_forced_manifest(
        jobs,
        manifest_path=manifest_path,
        summary_path=summary_path,
        shuffle_seed=shuffle_seed,
    )


def run_non_filtering_attack_pilot_from_files(
    *,
    config_path: Path,
    manifest_path: Path,
    results_path: Path,
    failure_path: Path,
    summary_path: Path,
    defense_id: DefenseId,
) -> AttackPilotSummary:
    if defense_id not in {DefenseId.D0, DefenseId.D1, DefenseId.D2}:
        raise ValueError("attack pilot supports D0, D1, and D2")
    config = load_config(config_path)
    return _run_attack_pilot_from_files(
        config=config,
        manifest_path=manifest_path,
        results_path=results_path,
        failure_path=failure_path,
        summary_path=summary_path,
        defense_id=defense_id,
        detector=BypassDetector(),
        detector_threshold=config.detector.threshold,
    )


def run_filtering_attack_pilot_from_files(
    *,
    config_path: Path,
    manifest_path: Path,
    results_path: Path,
    failure_path: Path,
    summary_path: Path,
    defense_id: DefenseId,
    detector_results_path: Path,
    detector_summary_path: Path,
    detector_threshold_path: Path,
) -> AttackPilotSummary:
    if defense_id not in {DefenseId.D3, DefenseId.D4}:
        raise ValueError("filtering attack pilot supports D3 and D4")
    _validate_immutable_inputs_and_outputs(
        input_paths=(
            config_path,
            manifest_path,
            detector_results_path,
            detector_summary_path,
            detector_threshold_path,
        ),
        output_paths=(results_path, failure_path, summary_path),
        label="filtering attack pilot",
    )
    config = load_config(config_path)
    detector, threshold = _load_verified_detector_calibration(
        config=config,
        results_path=detector_results_path,
        summary_path=detector_summary_path,
        threshold_path=detector_threshold_path,
    )
    return _run_attack_pilot_from_files(
        config=config,
        manifest_path=manifest_path,
        results_path=results_path,
        failure_path=failure_path,
        summary_path=summary_path,
        defense_id=defense_id,
        detector=detector,
        detector_threshold=threshold,
    )


def _run_attack_pilot_from_files(
    *,
    config: ResearchConfig,
    manifest_path: Path,
    results_path: Path,
    failure_path: Path,
    summary_path: Path,
    defense_id: DefenseId,
    detector: DetectorAdapter,
    detector_threshold: float,
) -> AttackPilotSummary:
    paths = (
        manifest_path.resolve(strict=False),
        results_path.resolve(strict=False),
        failure_path.resolve(strict=False),
        summary_path.resolve(strict=False),
    )
    if len(set(paths)) != len(paths):
        raise ValueError("attack pilot paths must be distinct")

    jobs = _read_fixture_jsonl(manifest_path, ExperimentJob, "job_id")
    config_hash = _config_hash(config)
    if any(job.config_hash != config_hash for job in jobs):
        raise ValueError("attack pilot manifest config hash mismatch")

    defense_jobs = tuple(job for job in jobs if job.defense_id is defense_id)
    attacks_by_id: dict[str, AttackCase] = {}
    for job in defense_jobs:
        if job.attack is None or job.attack_id is None:
            raise ValueError("attack pilot manifest job requires an attack")
        previous = attacks_by_id.setdefault(job.attack_id, job.attack)
        if previous != job.attack:
            raise ValueError("attack pilot attack definition mismatch")
    selected_attack_ids = select_factorial_pilot_attack_ids(
        tuple(attacks_by_id.values())
    )
    selected_attack_set = set(selected_attack_ids)
    selected_jobs = tuple(
        job for job in defense_jobs if job.attack_id in selected_attack_set
    )
    if len(selected_jobs) != len(selected_attack_ids):
        raise ValueError("attack pilot requires one job per selected attack")
    selected_job_ids = tuple(
        next(
            job.job_id
            for job in selected_jobs
            if job.attack_id == attack_id
        )
        for attack_id in selected_attack_ids
    )

    generator = _ollama_generator_from_config(config)
    generator.verify_model()
    run_jobs(
        selected_jobs,
        result_path=results_path,
        failure_path=failure_path,
        defenses={defense_id: _defenses(detector_threshold)[defense_id]},
        generator=generator,
        detector=detector,
    )
    records = load_records(results_path)
    failure_records = load_failure_records(failure_path)
    selected_job_values = [
        job.model_dump(mode="json")
        for job in sorted(selected_jobs, key=lambda item: item.job_id)
    ]
    summary = summarize_attack_pilot(
        records,
        defense_id=defense_id,
        selected_attack_ids=selected_attack_ids,
        selected_job_ids=selected_job_ids,
        failure_attempt_count=len(failure_records),
        manifest_job_set_sha256=canonical_sha256(
            cast(JsonValue, selected_job_values)
        ),
        results_file_sha256=hashlib.sha256(results_path.read_bytes()).hexdigest(),
        detector_threshold=(
            detector_threshold
            if defense_id in {DefenseId.D3, DefenseId.D4}
            else None
        ),
    )
    write_attack_pilot_summary(summary_path, summary)
    return summary


def run_clean_generation_pilot_from_files(
    *,
    config_path: Path,
    chunks_path: Path,
    questions_path: Path,
    cache_path: Path,
    metadata_path: Path,
    results_path: Path,
    summary_path: Path,
    review_path: Path,
    defense_id: DefenseId,
    detector_results_path: Path | None = None,
    detector_summary_path: Path | None = None,
    detector_threshold_path: Path | None = None,
) -> CleanGenerationPilotSummary:
    detector: DetectorAdapter | None = None
    detector_threshold_paths: tuple[Path, ...] = ()
    if defense_id in {DefenseId.D3, DefenseId.D4}:
        detector_paths = (
            detector_results_path,
            detector_summary_path,
            detector_threshold_path,
        )
        if any(path is None for path in detector_paths):
            raise ValueError(
                "filtering generation pilot requires calibrated detector artifacts"
            )
        detector_threshold_paths = tuple(cast(Path, path) for path in detector_paths)
    _validate_immutable_inputs_and_outputs(
        input_paths=(
            config_path,
            chunks_path,
            questions_path,
            cache_path,
            metadata_path,
            *detector_threshold_paths,
        ),
        output_paths=(results_path, summary_path, review_path),
        label=(
            "filtering generation pilot"
            if detector_threshold_paths
            else "generation pilot"
        ),
    )
    config = load_config(config_path)
    detector_threshold = config.detector.threshold
    if detector_threshold_paths:
        detector, detector_threshold = _load_verified_detector_calibration(
            config=config,
            results_path=detector_threshold_paths[0],
            summary_path=detector_threshold_paths[1],
            threshold_path=detector_threshold_paths[2],
        )
    chunks = _read_fixture_jsonl(chunks_path, Chunk, "chunk_id")
    questions = _read_fixture_jsonl(
        questions_path,
        ResearchQuestion,
        "question_id",
    )
    corpus_sha256 = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
    questions_sha256 = hashlib.sha256(questions_path.read_bytes()).hexdigest()
    try:
        metadata = EmbeddingCacheMetadata.model_validate_json(
            metadata_path.read_text(encoding="utf-8"),
            strict=True,
        )
    except (OSError, ValidationError) as error:
        raise ValueError("invalid embedding cache metadata") from error
    embedding = config.embedding
    expected_values = {
        "corpus_sha256": corpus_sha256,
        "questions_sha256": questions_sha256,
        "model_id": embedding.model_id,
        "model_revision": embedding.model_revision,
        "dimensions": embedding.dimensions,
        "query_instruction": embedding.query_instruction,
        "chunk_ids": tuple(chunk.chunk_id for chunk in chunks),
        "chunk_text_sha256": tuple(sha256_text(chunk.text) for chunk in chunks),
        "question_ids": tuple(question.question_id for question in questions),
        "question_text_sha256": tuple(
            sha256_text(question.question) for question in questions
        ),
    }
    mismatched_fields = tuple(
        field
        for field, expected in expected_values.items()
        if getattr(metadata, field) != expected
    )
    if mismatched_fields:
        raise ValueError(
            "embedding cache metadata mismatch: " + ",".join(mismatched_fields)
        )
    cache = load_embedding_cache(
        cache_path,
        metadata_path,
        expected_metadata=metadata,
    )
    return run_clean_generation_pilot(
        chunks,
        questions,
        cache.chunk_vectors,
        cache.question_vectors,
        _ollama_generator_from_config(config),
        corpus_sha256=corpus_sha256,
        questions_sha256=questions_sha256,
        embedding_cache_sha256=hashlib.sha256(cache_path.read_bytes()).hexdigest(),
        generation_config_sha256=canonical_sha256(config.generation),
        top_k=config.retrieval.top_k,
        results_path=results_path,
        summary_path=summary_path,
        review_path=review_path,
        defense=_defenses(detector_threshold)[defense_id],
        detector=detector,
    )


def _validate_immutable_inputs_and_outputs(
    *,
    input_paths: Sequence[Path],
    output_paths: Sequence[Path],
    label: str,
) -> None:
    resolved_inputs = tuple(path.resolve(strict=False) for path in input_paths)
    resolved_outputs = tuple(path.resolve(strict=False) for path in output_paths)
    if (
        len(set(resolved_inputs)) != len(resolved_inputs)
        or len(set(resolved_outputs)) != len(resolved_outputs)
        or set(resolved_inputs) & set(resolved_outputs)
    ):
        raise ValueError(f"{label} input/output paths must be distinct")


def audit_filtering_pilots_from_files(
    *,
    d3_attack_results_path: Path,
    d4_attack_results_path: Path,
    d3_attack_summary_path: Path,
    d4_attack_summary_path: Path,
    d3_clean_results_path: Path,
    d4_clean_results_path: Path,
    d3_clean_summary_path: Path,
    d4_clean_summary_path: Path,
    questions_path: Path,
    output_path: Path,
) -> FilteringPairedAuditReport:
    source_paths = {
        "d3_attack_results": d3_attack_results_path,
        "d4_attack_results": d4_attack_results_path,
        "d3_attack_summary": d3_attack_summary_path,
        "d4_attack_summary": d4_attack_summary_path,
        "d3_clean_results": d3_clean_results_path,
        "d4_clean_results": d4_clean_results_path,
        "d3_clean_summary": d3_clean_summary_path,
        "d4_clean_summary": d4_clean_summary_path,
    }
    _validate_immutable_inputs_and_outputs(
        input_paths=(*source_paths.values(), questions_path),
        output_paths=(output_path,),
        label="filtering audit",
    )

    questions = cast(
        tuple[ResearchQuestion, ...],
        _read_fixture_jsonl(
            questions_path,
            ResearchQuestion,
            "question_id",
        ),
    )
    d3_attack_records = cast(
        tuple[ExperimentRecord, ...],
        _read_fixture_jsonl(
            d3_attack_results_path,
            ExperimentRecord,
            "job_id",
        ),
    )
    d4_attack_records = cast(
        tuple[ExperimentRecord, ...],
        _read_fixture_jsonl(
            d4_attack_results_path,
            ExperimentRecord,
            "job_id",
        ),
    )
    d3_clean_records = cast(
        tuple[CleanGenerationPilotRecord, ...],
        _read_fixture_jsonl(
            d3_clean_results_path,
            CleanGenerationPilotRecord,
            "question_id",
        ),
    )
    d4_clean_records = cast(
        tuple[CleanGenerationPilotRecord, ...],
        _read_fixture_jsonl(
            d4_clean_results_path,
            CleanGenerationPilotRecord,
            "question_id",
        ),
    )
    try:
        d3_attack_summary = AttackPilotSummary.model_validate_json(
            d3_attack_summary_path.read_text(encoding="utf-8"),
            strict=True,
        )
        d4_attack_summary = AttackPilotSummary.model_validate_json(
            d4_attack_summary_path.read_text(encoding="utf-8"),
            strict=True,
        )
        d3_clean_summary = CleanGenerationPilotSummary.model_validate_json(
            d3_clean_summary_path.read_text(encoding="utf-8"),
            strict=True,
        )
        d4_clean_summary = CleanGenerationPilotSummary.model_validate_json(
            d4_clean_summary_path.read_text(encoding="utf-8"),
            strict=True,
        )
    except (OSError, ValidationError) as error:
        raise ValueError("invalid filtering audit summary artifact") from error
    questions_sha256 = hashlib.sha256(questions_path.read_bytes()).hexdigest()
    if (
        d3_clean_summary.questions_sha256 != questions_sha256
        or d4_clean_summary.questions_sha256 != questions_sha256
    ):
        raise ValueError("clean summaries do not match the expected questions file")

    report = audit_filtering_pilots(
        d3_attack_records=d3_attack_records,
        d4_attack_records=d4_attack_records,
        d3_attack_summary=d3_attack_summary,
        d4_attack_summary=d4_attack_summary,
        d3_clean_records=d3_clean_records,
        d4_clean_records=d4_clean_records,
        d3_clean_summary=d3_clean_summary,
        d4_clean_summary=d4_clean_summary,
        expected_clean_question_ids=tuple(
            question.question_id for question in questions
        ),
        source_file_sha256={
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in source_paths.items()
        },
    )
    write_filtering_paired_audit(output_path, report)
    return report


def review_clean_generation_from_files(
    *,
    questions_path: Path,
    results_path: Path,
    annotations_path: Path,
    summary_path: Path,
    reviewer_id: str,
) -> CleanUtilityReviewSummary:
    resolved_paths = {
        path.resolve(strict=False)
        for path in (
            questions_path,
            results_path,
            annotations_path,
            summary_path,
        )
    }
    if len(resolved_paths) != 4:
        raise ValueError("review paths must be distinct")

    questions = cast(
        tuple[ResearchQuestion, ...],
        _read_fixture_jsonl(questions_path, ResearchQuestion, "question_id"),
    )
    results = cast(
        tuple[CleanGenerationPilotRecord, ...],
        _read_fixture_jsonl(
            results_path,
            CleanGenerationPilotRecord,
            "question_id",
        ),
    )
    annotations = load_clean_utility_annotations(annotations_path)
    summary = summarize_clean_utility_annotations(
        questions,
        annotations,
        results,
        reviewer_id=reviewer_id,
    )
    write_clean_utility_review_summary(summary_path, summary)
    return summary


def _summary_json(summary: FixturePipelineSummary) -> str:
    value: dict[str, JsonValue] = {
        "jobs_expected": summary.jobs_expected,
        "jobs_completed": summary.jobs_completed,
        "jobs_skipped": summary.jobs_skipped,
        "duplicate_jobs": summary.duplicate_jobs,
        "technical_failures": summary.technical_failures,
        "defense_ids": [defense_id.value for defense_id in summary.defense_ids],
    }
    return canonical_json(value)


def _run_cli(action: Callable[[], str]) -> None:
    try:
        typer.echo(action())
    except Exception as error:
        detail = str(error).splitlines()[0] or type(error).__name__
        typer.echo(f"error: {detail}", err=True)
        raise typer.Exit(code=1) from error


app = typer.Typer(no_args_is_help=True)
data_app = typer.Typer(no_args_is_help=True)
fixture_app = typer.Typer(no_args_is_help=True)
corpus_app = typer.Typer(no_args_is_help=True)
qa_app = typer.Typer(no_args_is_help=True)
retrieval_app = typer.Typer(no_args_is_help=True)
generation_app = typer.Typer(no_args_is_help=True)
detector_app = typer.Typer(no_args_is_help=True)
manifest_app = typer.Typer(no_args_is_help=True)
experiment_app = typer.Typer(no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(fixture_app, name="fixture")
app.add_typer(corpus_app, name="corpus")
app.add_typer(qa_app, name="qa")
app.add_typer(retrieval_app, name="retrieval")
app.add_typer(generation_app, name="generation")
app.add_typer(detector_app, name="detector")
app.add_typer(manifest_app, name="manifest")
app.add_typer(experiment_app, name="experiment")


@app.callback()
def main() -> None:
    pass


@app.command()
def health() -> None:
    typer.echo("thesis-rag core ready")


@data_app.command("validate")
def data_validate(
    questions: Annotated[Path, typer.Option()] = _REQUIRED_PATH,
    attacks: Annotated[Path, typer.Option()] = _REQUIRED_PATH,
) -> None:
    _run_cli(lambda: canonical_json(cast(JsonValue, validate_fixture_data(questions, attacks))))


@qa_app.command("validate")
def qa_validate(
    questions: Annotated[Path, typer.Option()] = _DEVELOPMENT_QUESTIONS_PATH,
    chunks: Annotated[Path, typer.Option()] = _CORPUS_CHUNKS_PATH,
    expected_count: Annotated[int | None, typer.Option(min=1)] = None,
) -> None:
    _run_cli(
        lambda: canonical_json(
            cast(
                JsonValue,
                validate_question_data(
                    questions,
                    chunks,
                    expected_count=expected_count,
                ),
            )
        )
    )


@qa_app.command("review-sheet")
def qa_review_sheet(
    questions: Annotated[Path, typer.Option()] = _DEVELOPMENT_QUESTIONS_PATH,
    chunks: Annotated[Path, typer.Option()] = _CORPUS_CHUNKS_PATH,
    output: Annotated[Path, typer.Option()] = (
        _REPOSITORY_ROOT / "artifacts" / "reports" / "dev-qa-review.md"
    ),
) -> None:
    _run_cli(
        lambda: canonical_json(
            cast(
                JsonValue,
                generate_question_review_sheet(questions, chunks, output),
            )
        )
    )


@retrieval_app.command("pilot")
def retrieval_pilot(
    config: Annotated[Path, typer.Option()] = _DEVELOPMENT_CONFIG_PATH,
    chunks: Annotated[Path, typer.Option()] = _CORPUS_CHUNKS_PATH,
    questions: Annotated[Path, typer.Option()] = _DEVELOPMENT_QUESTIONS_PATH,
    cache: Annotated[Path, typer.Option()] = _EMBEDDING_CACHE_PATH,
    metadata: Annotated[Path, typer.Option()] = _EMBEDDING_METADATA_PATH,
    results: Annotated[Path, typer.Option()] = _RETRIEVAL_RESULTS_PATH,
    summary: Annotated[Path, typer.Option()] = _RETRIEVAL_SUMMARY_PATH,
) -> None:
    _run_cli(
        lambda: canonical_json(
            run_retrieval_pilot_from_files(
                config_path=config,
                chunks_path=chunks,
                questions_path=questions,
                cache_path=cache,
                metadata_path=metadata,
                results_path=results,
                summary_path=summary,
            )
        )
    )


@generation_app.command("verify")
def generation_verify(
    config: Annotated[Path, typer.Option()] = _DEVELOPMENT_CONFIG_PATH,
) -> None:
    _run_cli(lambda: canonical_json(verify_generation_model(config)))


@detector_app.command("verify")
def detector_verify(
    config: Annotated[Path, typer.Option()] = _DEVELOPMENT_CONFIG_PATH,
) -> None:
    _run_cli(lambda: canonical_json(verify_detector_model(config)))


@detector_app.command("pilot")
def detector_pilot(
    config: Annotated[Path, typer.Option()] = _DEVELOPMENT_CONFIG_PATH,
    cases: Annotated[Path, typer.Option()] = _DETECTOR_CASES_PATH,
    results: Annotated[Path, typer.Option()] = _DETECTOR_PILOT_RESULTS_PATH,
    summary: Annotated[Path, typer.Option()] = _DETECTOR_PILOT_SUMMARY_PATH,
) -> None:
    _run_cli(
        lambda: canonical_json(
            run_detector_pilot_from_files(
                config_path=config,
                cases_path=cases,
                results_path=results,
                summary_path=summary,
            )
        )
    )


@detector_app.command("calibrate")
def detector_calibrate(
    results: Annotated[Path, typer.Option()] = _DETECTOR_PILOT_RESULTS_PATH,
    summary: Annotated[Path, typer.Option()] = _DETECTOR_PILOT_SUMMARY_PATH,
    output: Annotated[Path, typer.Option()] = _DETECTOR_THRESHOLD_PATH,
    maximum_false_positive_rate: Annotated[
        float,
        typer.Option(min=0.0, max=1.0),
    ] = 0.05,
) -> None:
    _run_cli(
        lambda: canonical_json(
            calibrate_detector_from_file(
                results_path=results,
                summary_path=summary,
                output_path=output,
                maximum_false_positive_rate=maximum_false_positive_rate,
            )
        )
    )


@manifest_app.command("build")
def manifest_build(
    track: Annotated[str, typer.Option()] = "forced",
    config: Annotated[Path, typer.Option()] = _DEVELOPMENT_CONFIG_PATH,
    chunks: Annotated[Path, typer.Option()] = _CORPUS_CHUNKS_PATH,
    questions: Annotated[Path, typer.Option()] = _DEVELOPMENT_QUESTIONS_PATH,
    attacks: Annotated[Path, typer.Option()] = _DEVELOPMENT_ATTACKS_PATH,
    cache: Annotated[Path, typer.Option()] = _EMBEDDING_CACHE_PATH,
    metadata: Annotated[Path, typer.Option()] = _EMBEDDING_METADATA_PATH,
    output: Annotated[Path, typer.Option()] = _FORCED_MANIFEST_PATH,
    summary: Annotated[Path, typer.Option()] = _FORCED_MANIFEST_SUMMARY_PATH,
    shuffle_seed: Annotated[int, typer.Option()] = 42,
) -> None:
    def build() -> str:
        if track != "forced":
            raise ValueError("development manifest build currently supports forced track")
        return canonical_json(
            build_forced_manifest_from_files(
                config_path=config,
                chunks_path=chunks,
                questions_path=questions,
                attacks_path=attacks,
                cache_path=cache,
                metadata_path=metadata,
                manifest_path=output,
                summary_path=summary,
                shuffle_seed=shuffle_seed,
            )
        )

    _run_cli(build)


@experiment_app.command("attack-pilot")
def experiment_attack_pilot(
    config: Annotated[Path, typer.Option()] = _DEVELOPMENT_CONFIG_PATH,
    manifest: Annotated[Path, typer.Option()] = _FORCED_MANIFEST_PATH,
    results: Annotated[Path | None, typer.Option()] = None,
    failures: Annotated[Path | None, typer.Option()] = None,
    summary: Annotated[Path | None, typer.Option()] = None,
    detector_results: Annotated[Path, typer.Option()] = (
        _DETECTOR_PILOT_RESULTS_PATH
    ),
    detector_summary: Annotated[Path, typer.Option()] = (
        _DETECTOR_PILOT_SUMMARY_PATH
    ),
    detector_threshold: Annotated[Path, typer.Option()] = (
        _DETECTOR_THRESHOLD_PATH
    ),
    defense: Annotated[_DevelopmentPilotDefenseId, typer.Option()] = (
        _DevelopmentPilotDefenseId.D0
    ),
) -> None:
    defense_id = DefenseId(defense.value)
    default_results, default_failures, default_summary = (
        _attack_pilot_default_paths(defense_id)
    )
    def run() -> str:
        if defense_id in {DefenseId.D3, DefenseId.D4}:
            pilot_summary = run_filtering_attack_pilot_from_files(
                config_path=config,
                manifest_path=manifest,
                results_path=results or default_results,
                failure_path=failures or default_failures,
                summary_path=summary or default_summary,
                defense_id=defense_id,
                detector_results_path=detector_results,
                detector_summary_path=detector_summary,
                detector_threshold_path=detector_threshold,
            )
        else:
            pilot_summary = run_non_filtering_attack_pilot_from_files(
                config_path=config,
                manifest_path=manifest,
                results_path=results or default_results,
                failure_path=failures or default_failures,
                summary_path=summary or default_summary,
                defense_id=defense_id,
            )
        return canonical_json(pilot_summary)

    _run_cli(run)


@experiment_app.command("audit-filtering")
def experiment_audit_filtering(
    d3_attack_results: Annotated[Path, typer.Option()] = _D3_ATTACK_PILOT_PATHS[0],
    d4_attack_results: Annotated[Path, typer.Option()] = _D4_ATTACK_PILOT_PATHS[0],
    d3_attack_summary: Annotated[Path, typer.Option()] = _D3_ATTACK_PILOT_PATHS[2],
    d4_attack_summary: Annotated[Path, typer.Option()] = _D4_ATTACK_PILOT_PATHS[2],
    d3_clean_results: Annotated[Path, typer.Option()] = _D3_GENERATION_PILOT_PATHS[0],
    d4_clean_results: Annotated[Path, typer.Option()] = _D4_GENERATION_PILOT_PATHS[0],
    d3_clean_summary: Annotated[Path, typer.Option()] = _D3_GENERATION_PILOT_PATHS[1],
    d4_clean_summary: Annotated[Path, typer.Option()] = _D4_GENERATION_PILOT_PATHS[1],
    questions: Annotated[Path, typer.Option()] = _DEVELOPMENT_QUESTIONS_PATH,
    output: Annotated[Path, typer.Option()] = _FILTERING_PAIRED_AUDIT_PATH,
) -> None:
    _run_cli(
        lambda: canonical_json(
            audit_filtering_pilots_from_files(
                d3_attack_results_path=d3_attack_results,
                d4_attack_results_path=d4_attack_results,
                d3_attack_summary_path=d3_attack_summary,
                d4_attack_summary_path=d4_attack_summary,
                d3_clean_results_path=d3_clean_results,
                d4_clean_results_path=d4_clean_results,
                d3_clean_summary_path=d3_clean_summary,
                d4_clean_summary_path=d4_clean_summary,
                questions_path=questions,
                output_path=output,
            )
        )
    )


@generation_app.command("pilot")
def generation_pilot(
    config: Annotated[Path, typer.Option()] = _DEVELOPMENT_CONFIG_PATH,
    chunks: Annotated[Path, typer.Option()] = _CORPUS_CHUNKS_PATH,
    questions: Annotated[Path, typer.Option()] = _DEVELOPMENT_QUESTIONS_PATH,
    cache: Annotated[Path, typer.Option()] = _EMBEDDING_CACHE_PATH,
    metadata: Annotated[Path, typer.Option()] = _EMBEDDING_METADATA_PATH,
    results: Annotated[Path | None, typer.Option()] = None,
    summary: Annotated[Path | None, typer.Option()] = None,
    review: Annotated[Path | None, typer.Option()] = None,
    detector_results: Annotated[Path, typer.Option()] = (
        _DETECTOR_PILOT_RESULTS_PATH
    ),
    detector_summary: Annotated[Path, typer.Option()] = (
        _DETECTOR_PILOT_SUMMARY_PATH
    ),
    detector_threshold: Annotated[Path, typer.Option()] = (
        _DETECTOR_THRESHOLD_PATH
    ),
    defense: Annotated[_DevelopmentPilotDefenseId, typer.Option()] = (
        _DevelopmentPilotDefenseId.D0
    ),
) -> None:
    defense_id = DefenseId(defense.value)
    default_results, default_summary, default_review = (
        _generation_pilot_default_paths(defense_id)
    )
    results_path = results or default_results
    summary_path = summary or default_summary
    review_path = review or default_review

    def run() -> str:
        resolved_paths = {
            path.resolve(strict=False)
            for path in (results_path, summary_path, review_path)
        }
        if len(resolved_paths) != 3:
            raise ValueError("generation pilot paths must be distinct")
        if defense_id in {DefenseId.D3, DefenseId.D4}:
            pilot_summary = run_clean_generation_pilot_from_files(
                config_path=config,
                chunks_path=chunks,
                questions_path=questions,
                cache_path=cache,
                metadata_path=metadata,
                results_path=results_path,
                summary_path=summary_path,
                review_path=review_path,
                defense_id=defense_id,
                detector_results_path=detector_results,
                detector_summary_path=detector_summary,
                detector_threshold_path=detector_threshold,
            )
        else:
            pilot_summary = run_clean_generation_pilot_from_files(
                config_path=config,
                chunks_path=chunks,
                questions_path=questions,
                cache_path=cache,
                metadata_path=metadata,
                results_path=results_path,
                summary_path=summary_path,
                review_path=review_path,
                defense_id=defense_id,
            )
        return canonical_json(pilot_summary)

    _run_cli(
        run
    )


@generation_app.command("review")
def generation_review(
    questions: Annotated[Path, typer.Option()] = _DEVELOPMENT_QUESTIONS_PATH,
    results: Annotated[Path, typer.Option()] = _GENERATION_PILOT_RESULTS_PATH,
    annotations: Annotated[Path, typer.Option()] = (
        _GENERATION_PILOT_CODEX_ANNOTATIONS_PATH
    ),
    summary: Annotated[Path, typer.Option()] = (
        _GENERATION_PILOT_CODEX_SUMMARY_PATH
    ),
    reviewer_id: Annotated[str, typer.Option()] = "codex",
) -> None:
    _run_cli(
        lambda: canonical_json(
            review_clean_generation_from_files(
                questions_path=questions,
                results_path=results,
                annotations_path=annotations,
                summary_path=summary,
                reviewer_id=reviewer_id,
            )
        )
    )


@fixture_app.command("run")
def fixture_run(
    config: Annotated[Path, typer.Option()] = _REQUIRED_PATH,
    output: Annotated[Path, typer.Option()] = _REQUIRED_PATH,
) -> None:
    _run_cli(lambda: _summary_json(run_fixture_pipeline(config, output)))


@fixture_app.command("verify")
def fixture_verify(results: Annotated[Path, typer.Option()] = _REQUIRED_PATH) -> None:
    _run_cli(lambda: _summary_json(verify_fixture_results(results)))


def _corpus_summary(
    report: CorpusValidationReport,
    report_path: Path,
) -> str:
    value = report.model_dump(mode="json")
    value["report_file_sha256"] = hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()
    return canonical_json(cast(JsonValue, value))


@corpus_app.command("build")
def corpus_build(
    approved: Annotated[Path, typer.Option()] = _APPROVED_CORPUS_PATH,
    parsed_dir: Annotated[Path, typer.Option()] = _PARSED_CORPUS_DIRECTORY,
    chunks: Annotated[Path, typer.Option()] = _CORPUS_CHUNKS_PATH,
    report: Annotated[Path, typer.Option()] = _CORPUS_REPORT_PATH,
) -> None:
    def action() -> str:
        summary = build_corpus(
            repository_root=_REPOSITORY_ROOT,
            approved_path=approved,
            parsed_dir=parsed_dir,
            chunks_path=chunks,
            report_path=report,
            config=ChunkingConfig(),
        )
        return _corpus_summary(summary, report)

    _run_cli(action)


@corpus_app.command("verify")
def corpus_verify(
    approved: Annotated[Path, typer.Option()] = _APPROVED_CORPUS_PATH,
    parsed_dir: Annotated[Path, typer.Option()] = _PARSED_CORPUS_DIRECTORY,
    chunks: Annotated[Path, typer.Option()] = _CORPUS_CHUNKS_PATH,
    report: Annotated[Path, typer.Option()] = _CORPUS_REPORT_PATH,
) -> None:
    def action() -> str:
        summary = verify_corpus_artifacts(
            repository_root=_REPOSITORY_ROOT,
            approved_path=approved,
            parsed_dir=parsed_dir,
            chunks_path=chunks,
            report_path=report,
            config=ChunkingConfig(),
        )
        return _corpus_summary(summary, report)

    _run_cli(action)
