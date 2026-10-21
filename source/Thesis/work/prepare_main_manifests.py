import hashlib
import os
import tempfile
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from thesis_rag.canonical import canonical_json, sha256_text
from thesis_rag.config import load_config
from thesis_rag.data_approval import (
    read_test_data_approval_record,
    verify_test_data_approval,
)
from thesis_rag.detector_pilot import (
    CalibratedDetectorThreshold,
    DetectorCaseClass,
    DetectorEvaluationCase,
)
from thesis_rag.domain import AttackCase, Chunk, DefenseId
from thesis_rag.embeddings import OllamaEmbeddingAdapter
from thesis_rag.main_embeddings import (
    MainEmbeddingCacheMetadata,
    build_main_embedding_cache,
    load_main_embedding_cache,
)
from thesis_rag.main_manifest import (
    build_clean_manifest,
    build_detector_manifest,
    build_e2e_manifest,
    select_main_e2e_contexts,
    write_detector_manifest,
    write_generation_manifest,
)
from thesis_rag.qa import ResearchQuestion
from thesis_rag.research_manifest import (
    build_forced_manifest,
    write_forced_manifest,
)

_ROOT = Path(__file__).resolve().parents[1]
_T = TypeVar("_T", bound=BaseModel)


def _read_jsonl(path: Path, model: type[_T]) -> tuple[_T, ...]:
    return tuple(
        model.model_validate_json(line, strict=True)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(raw_path)
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    protocol_lock_path = _ROOT / "artifacts/protocol-lock.json"
    if protocol_lock_path.exists():
        raise RuntimeError(
            "refusing to prepare manifests after protocol lock creation"
        )

    approval_path = (
        _ROOT / "artifacts/approvals/test-data-approval.json"
    )
    development_approval_path = (
        _ROOT / "artifacts/approvals/development-data-approval.json"
    )
    approval = read_test_data_approval_record(approval_path)
    verification = verify_test_data_approval(
        _ROOT,
        approval,
        development_approval_path=development_approval_path,
    )
    if not verification.valid:
        raise RuntimeError(
            "test-data approval verification failed: "
            + canonical_json(verification)
        )

    paths = {
        "config": _ROOT / "configs/main-draft.yaml",
        "chunks": _ROOT / "data/chunks/core.jsonl",
        "questions": _ROOT / "data/questions/test.jsonl",
        "attacks": _ROOT / "data/attacks/test.jsonl",
        "detector_cases": _ROOT / "data/detector/test-cases.jsonl",
        "threshold": _ROOT / "artifacts/reports/dev-detector-threshold.json",
        "cache": (
            _ROOT
            / "artifacts/embeddings/qwen3-embedding-0.6b-test-main.npz"
        ),
        "cache_metadata": (
            _ROOT
            / "artifacts/embeddings/"
            "qwen3-embedding-0.6b-test-main.metadata.json"
        ),
    }
    config = load_config(paths["config"])
    chunks = _read_jsonl(paths["chunks"], Chunk)
    questions = _read_jsonl(paths["questions"], ResearchQuestion)
    attacks = _read_jsonl(paths["attacks"], AttackCase)
    detector_cases = _read_jsonl(
        paths["detector_cases"],
        DetectorEvaluationCase,
    )
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
    if len(malicious_cases) != 180 or any(
        case.source_attack_id is None for case in malicious_cases
    ):
        raise RuntimeError(
            "expected 180 attributed malicious detector cases"
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
        timeout_seconds=600.0,
    )
    adapter.verify_model()
    runtime_version = adapter.runtime_version()
    metadata = MainEmbeddingCacheMetadata(
        corpus_sha256=_sha256_file(paths["chunks"]),
        questions_sha256=_sha256_file(paths["questions"]),
        detector_cases_sha256=_sha256_file(paths["detector_cases"]),
        model_id=adapter.model_id,
        model_revision=adapter.model_revision,
        runtime_version=runtime_version,
        dimensions=adapter.dimensions,
        query_instruction=adapter.query_instruction,
        normalization_policy="cosine_at_retrieval",
        chunk_ids=tuple(chunk.chunk_id for chunk in chunks),
        chunk_text_sha256=tuple(
            sha256_text(chunk.text) for chunk in chunks
        ),
        question_ids=tuple(
            question.question_id for question in questions
        ),
        question_text_sha256=tuple(
            sha256_text(question.question) for question in questions
        ),
        malicious_attack_ids=tuple(
            case.source_attack_id or "" for case in malicious_cases
        ),
        malicious_chunk_ids=tuple(
            case.chunk.chunk_id for case in malicious_cases
        ),
        malicious_text_sha256=tuple(
            sha256_text(case.chunk.text) for case in malicious_cases
        ),
    )
    cache_exists = paths["cache"].exists()
    metadata_exists = paths["cache_metadata"].exists()
    if cache_exists != metadata_exists:
        raise RuntimeError(
            "main embedding cache and metadata must exist together"
        )
    if cache_exists:
        cache = load_main_embedding_cache(
            paths["cache"],
            paths["cache_metadata"],
            expected_metadata=metadata,
        )
        cache_reused = True
    else:
        cache = build_main_embedding_cache(
            paths["cache"],
            paths["cache_metadata"],
            metadata=metadata,
            chunk_vectors=adapter.embed_documents(
                tuple(chunk.text for chunk in chunks)
            ),
            question_vectors=adapter.embed_queries(
                tuple(question.question for question in questions)
            ),
            malicious_vectors=adapter.embed_documents(
                tuple(case.chunk.text for case in malicious_cases)
            ),
        )
        cache_reused = False

    config_hash = sha256_text(canonical_json(config))
    selection = select_main_e2e_contexts(questions, attacks)
    selection_path = (
        _ROOT / "artifacts/manifests/main-e2e-selection.json"
    )
    _atomic_write(selection_path, f"{canonical_json(selection)}\n")

    forced_jobs = build_forced_manifest(
        chunks,
        questions,
        attacks,
        chunk_vectors=cache.chunk_vectors,
        question_vectors=cache.question_vectors,
        top_k=config.retrieval.top_k,
        defense_ids=tuple(DefenseId),
        config_hash=config_hash,
        shuffle_seed=config.generation.seed,
        ensure_gold_context=True,
    )
    forced_summary = write_forced_manifest(
        forced_jobs,
        manifest_path=(
            _ROOT / "artifacts/manifests/main-forced.jsonl"
        ),
        summary_path=(
            _ROOT / "artifacts/reports/main-forced-manifest-summary.json"
        ),
        shuffle_seed=config.generation.seed,
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
        defense_ids=tuple(DefenseId),
        config_hash=config_hash,
        shuffle_seed=config.generation.seed,
    )
    e2e_summary = write_generation_manifest(
        e2e_jobs,
        track="e2e",
        manifest_path=_ROOT / "artifacts/manifests/main-e2e.jsonl",
        summary_path=(
            _ROOT / "artifacts/reports/main-e2e-manifest-summary.json"
        ),
        shuffle_seed=config.generation.seed,
    )

    clean_jobs = build_clean_manifest(
        chunks,
        questions,
        chunk_vectors=cache.chunk_vectors,
        question_vectors=cache.question_vectors,
        top_k=config.retrieval.top_k,
        defense_ids=tuple(DefenseId),
        config_hash=config_hash,
        shuffle_seed=config.generation.seed,
    )
    clean_summary = write_generation_manifest(
        clean_jobs,
        track="clean",
        manifest_path=(
            _ROOT / "artifacts/manifests/main-clean.jsonl"
        ),
        summary_path=(
            _ROOT / "artifacts/reports/main-clean-manifest-summary.json"
        ),
        shuffle_seed=config.generation.seed,
    )

    threshold = CalibratedDetectorThreshold.model_validate_json(
        paths["threshold"].read_text(encoding="utf-8"),
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
        shuffle_seed=config.generation.seed,
    )
    detector_summary = write_detector_manifest(
        detector_entries,
        manifest_path=(
            _ROOT / "artifacts/manifests/main-detector.jsonl"
        ),
        summary_path=(
            _ROOT / "artifacts/reports/"
            "main-detector-manifest-summary.json"
        ),
        shuffle_seed=config.generation.seed,
    )

    artifact_paths = tuple(
        sorted(
            (
                paths["cache"],
                paths["cache_metadata"],
                selection_path,
                _ROOT / "artifacts/manifests/main-forced.jsonl",
                _ROOT / "artifacts/manifests/main-e2e.jsonl",
                _ROOT / "artifacts/manifests/main-clean.jsonl",
                _ROOT / "artifacts/manifests/main-detector.jsonl",
                _ROOT
                / "artifacts/reports/main-forced-manifest-summary.json",
                _ROOT
                / "artifacts/reports/main-e2e-manifest-summary.json",
                _ROOT
                / "artifacts/reports/main-clean-manifest-summary.json",
                _ROOT
                / "artifacts/reports/"
                "main-detector-manifest-summary.json",
            )
        )
    )
    preparation_report = {
        "schema_version": "1.0.0",
        "stage": "pre_protocol_freeze_manifest_preparation",
        "test_data_approval_id": approval.approval_id,
        "test_data_approval_sha256": _sha256_file(approval_path),
        "protocol_freeze_approved": False,
        "main_run_approved": False,
        "generation_or_detector_scoring_performed": False,
        "embedding_cache_reused": cache_reused,
        "config_sha256": config_hash,
        "selection_sha256": selection.selection_sha256,
        "manifest_counts": {
            "forced": forced_summary.job_count,
            "e2e": e2e_summary.job_count,
            "clean": clean_summary.job_count,
            "detector": detector_summary.job_count,
        },
        "artifacts": [
            {
                "path": path.relative_to(_ROOT).as_posix(),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in artifact_paths
        ],
    }
    report_path = (
        _ROOT / "artifacts/reports/main-manifest-preparation.json"
    )
    _atomic_write(
        report_path,
        f"{canonical_json(preparation_report)}\n",
    )
    print(
        canonical_json(
            {
                "status": "prepared_without_main_model_scoring",
                "report_path": report_path.relative_to(_ROOT).as_posix(),
                "manifest_counts": preparation_report["manifest_counts"],
            }
        )
    )


if __name__ == "__main__":
    main()
