import hashlib
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, TypeVar

import numpy as np
from pydantic import Field, StrictBool

from thesis_rag.canonical import canonical_json, sha256_text
from thesis_rag.config import ResearchConfig, load_config
from thesis_rag.data_approval import (
    read_test_data_approval_record,
    verify_test_data_approval,
)
from thesis_rag.detector_pilot import (
    CalibratedDetectorThreshold,
    DetectorCaseClass,
    DetectorEvaluationCase,
)
from thesis_rag.domain import (
    AttackCase,
    Chunk,
    DefenseId,
    ExperimentJob,
    ForcedResearchInputs,
    FrozenRecord,
    Sha256Hex,
)
from thesis_rag.main_embeddings import (
    MainEmbeddingCacheMetadata,
    load_main_embedding_cache,
)
from thesis_rag.main_manifest import (
    DetectorManifestEntry,
    DetectorManifestSummary,
    E2ESelectionSummary,
    GenerationManifestSummary,
    build_clean_manifest,
    build_detector_manifest,
    build_e2e_manifest,
    select_main_e2e_contexts,
)
from thesis_rag.manifests import (
    reconstruct_job_from_research_inputs,
    verify_e2e_job_from_sources,
)
from thesis_rag.qa import ResearchQuestion
from thesis_rag.research_manifest import (
    ForcedManifestSummary,
    build_forced_manifest,
)

_RecordT = TypeVar("_RecordT", bound=FrozenRecord)


class GenerationManifestVerification(FrozenRecord):
    schema_version: Literal["1.0.0"] = "1.0.0"
    track: Literal["forced", "e2e", "clean"]
    valid: StrictBool
    job_count: int = Field(strict=True, ge=1)
    condition_count: int = Field(strict=True, ge=1)
    reconstruction_count: int = Field(strict=True, ge=1)
    external_verification_count: int = Field(strict=True, ge=0)
    defense_ids: tuple[DefenseId, ...]
    manifest_sha256: Sha256Hex
    summary_sha256: Sha256Hex


class DetectorManifestVerification(FrozenRecord):
    schema_version: Literal["1.0.0"] = "1.0.0"
    track: Literal["detector"] = "detector"
    valid: StrictBool
    scoring_performed: StrictBool
    job_count: int = Field(strict=True, ge=1)
    malicious_count: int = Field(strict=True, ge=0)
    hard_negative_count: int = Field(strict=True, ge=0)
    ordinary_benign_count: int = Field(strict=True, ge=0)
    manifest_sha256: Sha256Hex
    summary_sha256: Sha256Hex


class ForcedManifestVerification(FrozenRecord):
    schema_version: Literal["1.0.0"] = "1.0.0"
    track: Literal["forced"] = "forced"
    valid: StrictBool
    job_count: int = Field(strict=True, ge=1)
    condition_count: int = Field(strict=True, ge=1)
    reconstruction_count: int = Field(strict=True, ge=1)
    malicious_rank_counts: tuple[int, ...]
    gold_context_fallback_condition_count: int = Field(strict=True, ge=0)
    gold_context_fallback_job_count: int = Field(strict=True, ge=0)
    defense_ids: tuple[DefenseId, ...]
    manifest_sha256: Sha256Hex
    summary_sha256: Sha256Hex


class MainManifestSuiteVerification(FrozenRecord):
    schema_version: Literal["1.0.0"] = "1.0.0"
    valid: StrictBool
    test_data_approval_id: str
    test_data_approval_valid: StrictBool
    generation_or_detector_scoring_performed: StrictBool
    config_sha256: Sha256Hex
    e2e_selection_sha256: Sha256Hex
    total_job_count: int = Field(strict=True, ge=1)
    forced: ForcedManifestVerification
    e2e: GenerationManifestVerification
    clean: GenerationManifestVerification
    detector: DetectorManifestVerification


def verify_main_manifest_suite(root: Path) -> MainManifestSuiteVerification:
    approval_path = root / "artifacts/approvals/test-data-approval.json"
    development_approval_path = (
        root / "artifacts/approvals/development-data-approval.json"
    )
    approval = read_test_data_approval_record(approval_path)
    approval_verification = verify_test_data_approval(
        root,
        approval,
        development_approval_path=development_approval_path,
    )
    if not approval_verification.valid:
        raise ValueError("test-data approval verification failed")

    config = load_config(root / "configs/main-draft.yaml")
    config_hash = sha256_text(canonical_json(config))
    chunks, _ = _read_canonical_jsonl(
        root / "data/chunks/core.jsonl",
        Chunk,
    )
    questions, _ = _read_canonical_jsonl(
        root / "data/questions/test.jsonl",
        ResearchQuestion,
    )
    attacks, _ = _read_canonical_jsonl(
        root / "data/attacks/test.jsonl",
        AttackCase,
    )
    detector_cases, _ = _read_canonical_jsonl(
        root / "data/detector/test-cases.jsonl",
        DetectorEvaluationCase,
    )
    metadata, _ = _read_canonical_model(
        root
        / "artifacts/embeddings/"
        "qwen3-embedding-0.6b-test-main.metadata.json",
        MainEmbeddingCacheMetadata,
    )
    _verify_embedding_metadata_sources(
        metadata,
        config=config,
        chunks=chunks,
        questions=questions,
        detector_cases=detector_cases,
        root=root,
    )
    cache = load_main_embedding_cache(
        root
        / "artifacts/embeddings/qwen3-embedding-0.6b-test-main.npz",
        root
        / "artifacts/embeddings/"
        "qwen3-embedding-0.6b-test-main.metadata.json",
        expected_metadata=metadata,
    )
    selection, _ = _read_canonical_model(
        root / "artifacts/manifests/main-e2e-selection.json",
        E2ESelectionSummary,
    )
    expected_selection = select_main_e2e_contexts(questions, attacks)
    if selection != expected_selection:
        raise ValueError("main E2E selection differs from frozen policy")

    defense_ids = tuple(DefenseId)
    shuffle_seed = config.generation.seed
    forced_jobs = build_forced_manifest(
        chunks,
        questions,
        attacks,
        chunk_vectors=cache.chunk_vectors,
        question_vectors=cache.question_vectors,
        top_k=config.retrieval.top_k,
        defense_ids=defense_ids,
        config_hash=config_hash,
        shuffle_seed=shuffle_seed,
        ensure_gold_context=True,
    )
    forced_report = verify_forced_manifest_artifact(
        root / "artifacts/manifests/main-forced.jsonl",
        root / "artifacts/reports/main-forced-manifest-summary.json",
        expected_jobs=forced_jobs,
        expected_shuffle_seed=shuffle_seed,
    )

    e2e_jobs = build_e2e_manifest(
        chunks,
        questions,
        attacks,
        selection=selection,
        chunk_vectors=cache.chunk_vectors,
        question_vectors=cache.question_vectors,
        malicious_attack_ids=metadata.malicious_attack_ids,
        malicious_vectors=cache.malicious_vectors,
        top_k=config.retrieval.top_k,
        defense_ids=defense_ids,
        config_hash=config_hash,
        shuffle_seed=shuffle_seed,
    )
    canonical_chunk_order = tuple(sorted(chunks, key=lambda item: item.chunk_id))
    chunk_vectors_by_id = dict(
        zip(metadata.chunk_ids, cache.chunk_vectors, strict=True)
    )
    canonical_chunk_matrix = np.asarray(
        [chunk_vectors_by_id[chunk.chunk_id] for chunk in canonical_chunk_order],
        dtype=np.float64,
    )
    question_vectors_by_id = dict(
        zip(metadata.question_ids, cache.question_vectors, strict=True)
    )
    malicious_vectors_by_id = dict(
        zip(
            metadata.malicious_attack_ids,
            cache.malicious_vectors,
            strict=True,
        )
    )

    def verify_e2e_external_sources(job: ExperimentJob) -> None:
        if job.attack_id is None:
            raise ValueError("E2E job requires an attack ID")
        verify_e2e_job_from_sources(
            job,
            corpus_chunks=canonical_chunk_order,
            corpus_matrix=canonical_chunk_matrix,
            query_vector=question_vectors_by_id[job.question_id],
            malicious_vector=malicious_vectors_by_id[job.attack_id],
        )

    e2e_report = verify_generation_manifest_artifact(
        root / "artifacts/manifests/main-e2e.jsonl",
        root / "artifacts/reports/main-e2e-manifest-summary.json",
        expected_jobs=e2e_jobs,
        expected_track="e2e",
        expected_shuffle_seed=shuffle_seed,
        external_verifier=verify_e2e_external_sources,
    )
    clean_jobs = build_clean_manifest(
        chunks,
        questions,
        chunk_vectors=cache.chunk_vectors,
        question_vectors=cache.question_vectors,
        top_k=config.retrieval.top_k,
        defense_ids=defense_ids,
        config_hash=config_hash,
        shuffle_seed=shuffle_seed,
    )
    clean_report = verify_generation_manifest_artifact(
        root / "artifacts/manifests/main-clean.jsonl",
        root / "artifacts/reports/main-clean-manifest-summary.json",
        expected_jobs=clean_jobs,
        expected_track="clean",
        expected_shuffle_seed=shuffle_seed,
    )
    threshold = CalibratedDetectorThreshold.model_validate_json(
        (
            root / "artifacts/reports/dev-detector-threshold.json"
        ).read_text(encoding="utf-8"),
        strict=True,
    )
    detector_entries = build_detector_manifest(
        detector_cases,
        config_hash=config_hash,
        detector_model_id=config.detector.model_id,
        detector_model_revision=config.detector.model_revision,
        scoring_policy_sha256=(
            threshold.detector_scoring_policy_sha256
        ),
        threshold=config.detector.threshold,
        shuffle_seed=shuffle_seed,
    )
    detector_report = verify_detector_manifest_artifact(
        root / "artifacts/manifests/main-detector.jsonl",
        root
        / "artifacts/reports/main-detector-manifest-summary.json",
        expected_entries=detector_entries,
        expected_shuffle_seed=shuffle_seed,
    )
    observed_counts = {
        "forced": forced_report.job_count,
        "e2e": e2e_report.job_count,
        "clean": clean_report.job_count,
        "detector": detector_report.job_count,
    }
    expected_counts = {
        "forced": 900,
        "e2e": 450,
        "clean": 500,
        "detector": 380,
    }
    if observed_counts != expected_counts:
        raise ValueError("main manifest suite counts differ from protocol")
    return MainManifestSuiteVerification(
        valid=True,
        test_data_approval_id=approval.approval_id,
        test_data_approval_valid=True,
        generation_or_detector_scoring_performed=False,
        config_sha256=config_hash,
        e2e_selection_sha256=selection.selection_sha256,
        total_job_count=sum(observed_counts.values()),
        forced=forced_report,
        e2e=e2e_report,
        clean=clean_report,
        detector=detector_report,
    )


def verify_forced_manifest_artifact(
    manifest_path: Path,
    summary_path: Path,
    *,
    expected_jobs: Sequence[ExperimentJob],
    expected_shuffle_seed: int,
) -> ForcedManifestVerification:
    jobs, manifest_contents = _read_canonical_jsonl(
        manifest_path,
        ExperimentJob,
    )
    summary, summary_contents = _read_canonical_model(
        summary_path,
        ForcedManifestSummary,
    )
    manifest_sha256 = _sha256_bytes(manifest_contents.encode("utf-8"))
    if summary.manifest_sha256 != manifest_sha256:
        raise ValueError("forced manifest hash mismatch")
    if not _canonical_records_equal(jobs, expected_jobs):
        raise ValueError("forced manifest does not match expected jobs")
    if any(job.track != "forced" for job in jobs):
        raise ValueError("forced manifest track mismatch")
    if len({job.job_id for job in jobs}) != len(jobs):
        raise ValueError("forced manifest contains duplicate job IDs")
    reconstructed = tuple(
        reconstruct_job_from_research_inputs(job) for job in jobs
    )
    if reconstructed != jobs:
        raise ValueError("forced manifest reconstruction mismatch")

    grouped: dict[str, list[ExperimentJob]] = {}
    for job in jobs:
        if job.attack_id is None:
            raise ValueError("forced manifest job requires attack_id")
        grouped.setdefault(job.attack_id, []).append(job)
    defense_sets = {
        frozenset(job.defense_id for job in group)
        for group in grouped.values()
    }
    if len(defense_sets) != 1:
        raise ValueError("forced manifest defense sets differ")
    defense_set = next(iter(defense_sets))
    if any(
        len(group) != len(defense_set)
        or len({job.defense_id for job in group}) != len(group)
        or len({job.research_input_hash for job in group}) != 1
        for group in grouped.values()
    ):
        raise ValueError("forced manifest has incomplete defense pairing")
    defense_ids = tuple(
        defense_id for defense_id in DefenseId if defense_id in defense_set
    )
    representatives = tuple(group[0] for group in grouped.values())
    if any(job.malicious_chunk_rank is None for job in representatives):
        raise ValueError("forced manifest requires malicious exposure")
    maximum_rank = max(len(job.ordered_chunks) for job in representatives)
    rank_counts = Counter(
        job.malicious_chunk_rank for job in representatives
    )
    malicious_rank_counts = tuple(
        rank_counts.get(rank, 0) for rank in range(1, maximum_rank + 1)
    )
    fallback_conditions = sum(
        isinstance(job.research_inputs, ForcedResearchInputs)
        and job.research_inputs.gold_context_fallback
        for job in representatives
    )
    fallback_jobs = sum(
        isinstance(job.research_inputs, ForcedResearchInputs)
        and job.research_inputs.gold_context_fallback
        for job in jobs
    )
    config_hashes = {job.config_hash for job in jobs}
    if len(config_hashes) != 1:
        raise ValueError("forced manifest config hashes differ")
    expected_summary = ForcedManifestSummary(
        schema_version="1.0.0",
        track="forced",
        job_count=len(jobs),
        attack_count=len(grouped),
        defense_ids=defense_ids,
        malicious_rank_counts=malicious_rank_counts,
        config_sha256=next(iter(config_hashes)),
        shuffle_seed=expected_shuffle_seed,
        manifest_sha256=manifest_sha256,
        gold_context_fallback_count=fallback_conditions,
    )
    if summary != expected_summary:
        raise ValueError("forced manifest summary mismatch")
    return ForcedManifestVerification(
        valid=True,
        job_count=len(jobs),
        condition_count=len(grouped),
        reconstruction_count=len(reconstructed),
        malicious_rank_counts=malicious_rank_counts,
        gold_context_fallback_condition_count=fallback_conditions,
        gold_context_fallback_job_count=fallback_jobs,
        defense_ids=defense_ids,
        manifest_sha256=manifest_sha256,
        summary_sha256=_sha256_bytes(summary_contents.encode("utf-8")),
    )


def verify_generation_manifest_artifact(
    manifest_path: Path,
    summary_path: Path,
    *,
    expected_jobs: Sequence[ExperimentJob],
    expected_track: Literal["forced", "e2e", "clean"],
    expected_shuffle_seed: int,
    external_verifier: Callable[[ExperimentJob], object] | None = None,
) -> GenerationManifestVerification:
    jobs, manifest_contents = _read_canonical_jsonl(
        manifest_path,
        ExperimentJob,
    )
    summary, summary_contents = _read_canonical_model(
        summary_path,
        GenerationManifestSummary,
    )
    manifest_sha256 = _sha256_bytes(manifest_contents.encode("utf-8"))
    if summary.manifest_sha256 != manifest_sha256:
        raise ValueError("generation manifest hash mismatch")
    if not _canonical_records_equal(jobs, expected_jobs):
        raise ValueError("generation manifest does not match expected jobs")
    if any(job.track != expected_track for job in jobs):
        raise ValueError("generation manifest track mismatch")
    if len({job.job_id for job in jobs}) != len(jobs):
        raise ValueError("generation manifest contains duplicate job IDs")

    reconstructed = tuple(
        reconstruct_job_from_research_inputs(job) for job in jobs
    )
    if reconstructed != jobs:
        raise ValueError("generation manifest reconstruction mismatch")
    grouped: dict[str, list[ExperimentJob]] = {}
    for job in jobs:
        grouped.setdefault(job.research_input_hash, []).append(job)
    defense_sets = {
        frozenset(job.defense_id for job in group)
        for group in grouped.values()
    }
    if len(defense_sets) != 1:
        raise ValueError("generation manifest defense sets differ")
    defense_set = next(iter(defense_sets))
    if any(
        len(group) != len(defense_set)
        or len({job.defense_id for job in group}) != len(group)
        for group in grouped.values()
    ):
        raise ValueError("generation manifest has incomplete defense pairing")
    defense_ids = tuple(
        defense_id for defense_id in DefenseId if defense_id in defense_set
    )
    config_hashes = {job.config_hash for job in jobs}
    if len(config_hashes) != 1:
        raise ValueError("generation manifest config hashes differ")
    expected_summary = GenerationManifestSummary(
        track=expected_track,
        job_count=len(jobs),
        condition_count=len(grouped),
        defense_ids=defense_ids,
        config_sha256=next(iter(config_hashes)),
        shuffle_seed=expected_shuffle_seed,
        manifest_sha256=manifest_sha256,
    )
    if summary != expected_summary:
        raise ValueError("generation manifest summary mismatch")

    external_count = 0
    if external_verifier is not None:
        for job in jobs:
            external_verifier(job)
            external_count += 1
    return GenerationManifestVerification(
        track=expected_track,
        valid=True,
        job_count=len(jobs),
        condition_count=len(grouped),
        reconstruction_count=len(reconstructed),
        external_verification_count=external_count,
        defense_ids=defense_ids,
        manifest_sha256=manifest_sha256,
        summary_sha256=_sha256_bytes(summary_contents.encode("utf-8")),
    )


def verify_detector_manifest_artifact(
    manifest_path: Path,
    summary_path: Path,
    *,
    expected_entries: Sequence[DetectorManifestEntry],
    expected_shuffle_seed: int,
) -> DetectorManifestVerification:
    entries, manifest_contents = _read_canonical_jsonl(
        manifest_path,
        DetectorManifestEntry,
    )
    summary, summary_contents = _read_canonical_model(
        summary_path,
        DetectorManifestSummary,
    )
    manifest_sha256 = _sha256_bytes(manifest_contents.encode("utf-8"))
    if summary.manifest_sha256 != manifest_sha256:
        raise ValueError("detector manifest hash mismatch")
    if not _canonical_records_equal(entries, expected_entries):
        raise ValueError("detector manifest does not match expected entries")
    if len({entry.job_id for entry in entries}) != len(entries):
        raise ValueError("detector manifest contains duplicate job IDs")
    if len({entry.case.case_id for entry in entries}) != len(entries):
        raise ValueError("detector manifest contains duplicate case IDs")
    bindings = {
        (
            entry.config_sha256,
            entry.detector_model_id,
            entry.detector_model_revision,
            entry.scoring_policy_sha256,
            entry.threshold,
        )
        for entry in entries
    }
    if len(bindings) != 1:
        raise ValueError("detector manifest bindings differ")
    (
        config_hash,
        detector_model_id,
        detector_model_revision,
        scoring_policy_sha256,
        threshold,
    ) = next(iter(bindings))
    class_counts = Counter(entry.case.case_class for entry in entries)
    expected_summary = DetectorManifestSummary(
        job_count=len(entries),
        malicious_count=class_counts[DetectorCaseClass.MALICIOUS],
        hard_negative_count=class_counts[DetectorCaseClass.HARD_NEGATIVE],
        ordinary_benign_count=class_counts[
            DetectorCaseClass.ORDINARY_BENIGN
        ],
        config_sha256=config_hash,
        detector_model_id=detector_model_id,
        detector_model_revision=detector_model_revision,
        scoring_policy_sha256=scoring_policy_sha256,
        threshold=threshold,
        shuffle_seed=expected_shuffle_seed,
        manifest_sha256=manifest_sha256,
    )
    if summary != expected_summary:
        raise ValueError("detector manifest summary mismatch")
    return DetectorManifestVerification(
        valid=True,
        scoring_performed=False,
        job_count=len(entries),
        malicious_count=class_counts[DetectorCaseClass.MALICIOUS],
        hard_negative_count=class_counts[DetectorCaseClass.HARD_NEGATIVE],
        ordinary_benign_count=class_counts[
            DetectorCaseClass.ORDINARY_BENIGN
        ],
        manifest_sha256=manifest_sha256,
        summary_sha256=_sha256_bytes(summary_contents.encode("utf-8")),
    )


def _read_canonical_jsonl(
    path: Path,
    model: type[_RecordT],
) -> tuple[tuple[_RecordT, ...], str]:
    try:
        contents = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"unable to read manifest: {path}") from error
    if not contents.endswith("\n"):
        raise ValueError("manifest must end with one newline")
    lines = contents.splitlines()
    if not lines or any(not line for line in lines):
        raise ValueError("manifest must contain nonblank JSONL records")
    try:
        records = tuple(
            model.model_validate_json(line, strict=True) for line in lines
        )
    except ValueError as error:
        raise ValueError("manifest contains an invalid record") from error
    expected_contents = "".join(
        f"{canonical_json(record)}\n" for record in records
    )
    if contents != expected_contents:
        raise ValueError("manifest serialization is not canonical")
    return records, contents


def _read_canonical_model(
    path: Path,
    model: type[_RecordT],
) -> tuple[_RecordT, str]:
    try:
        contents = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"unable to read summary: {path}") from error
    try:
        record = model.model_validate_json(contents, strict=True)
    except ValueError as error:
        raise ValueError("summary contains an invalid record") from error
    if contents != f"{canonical_json(record)}\n":
        raise ValueError("summary serialization is not canonical")
    return record, contents


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_records_equal(
    observed: Sequence[FrozenRecord],
    expected: Sequence[FrozenRecord],
) -> bool:
    return tuple(canonical_json(record) for record in observed) == tuple(
        canonical_json(record) for record in expected
    )


def _verify_embedding_metadata_sources(
    metadata: MainEmbeddingCacheMetadata,
    *,
    config: ResearchConfig,
    chunks: Sequence[Chunk],
    questions: Sequence[ResearchQuestion],
    detector_cases: Sequence[DetectorEvaluationCase],
    root: Path,
) -> None:
    embedding = config.embedding
    expected_bindings = (
        embedding.model_id,
        embedding.model_revision,
        embedding.dimensions,
        embedding.query_instruction,
    )
    observed_bindings = (
        metadata.model_id,
        metadata.model_revision,
        metadata.dimensions,
        metadata.query_instruction,
    )
    if observed_bindings != expected_bindings:
        raise ValueError("main embedding metadata model binding mismatch")
    malicious_cases = tuple(
        sorted(
            (
                case
                for case in detector_cases
                if case.case_class is DetectorCaseClass.MALICIOUS
            ),
            key=lambda case: case.source_attack_id or "",
        )
    )
    expected_source_bindings = (
        _sha256_file(root / "data/chunks/core.jsonl"),
        _sha256_file(root / "data/questions/test.jsonl"),
        _sha256_file(root / "data/detector/test-cases.jsonl"),
        tuple(chunk.chunk_id for chunk in chunks),
        tuple(sha256_text(chunk.text) for chunk in chunks),
        tuple(question.question_id for question in questions),
        tuple(sha256_text(question.question) for question in questions),
        tuple(case.source_attack_id or "" for case in malicious_cases),
        tuple(case.chunk.chunk_id for case in malicious_cases),
        tuple(sha256_text(case.chunk.text) for case in malicious_cases),
    )
    observed_source_bindings = (
        metadata.corpus_sha256,
        metadata.questions_sha256,
        metadata.detector_cases_sha256,
        metadata.chunk_ids,
        metadata.chunk_text_sha256,
        metadata.question_ids,
        metadata.question_text_sha256,
        metadata.malicious_attack_ids,
        metadata.malicious_chunk_ids,
        metadata.malicious_text_sha256,
    )
    if observed_source_bindings != expected_source_bindings:
        raise ValueError("main embedding metadata source binding mismatch")


def _sha256_file(path: Path) -> str:
    try:
        contents = path.read_bytes()
    except OSError as error:
        raise ValueError(f"unable to read source file: {path}") from error
    return _sha256_bytes(contents)
