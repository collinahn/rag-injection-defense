import hashlib
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from thesis_rag.canonical import canonical_json
from thesis_rag.config import ResearchConfig, load_config
from thesis_rag.detector import PromptGuardDetectorAdapter
from thesis_rag.detector_pilot import run_detector_pilot
from thesis_rag.domain import ExperimentRecord
from thesis_rag.generation import OllamaGeneratorAdapter
from thesis_rag.jsonl import load_failure_records, load_records
from thesis_rag.main_run import (
    build_main_defenses,
    load_frozen_detector_manifest,
    load_frozen_generation_manifest,
)
from thesis_rag.protocol import (
    ModelRevision,
    ProtocolLock,
    assert_main_run_allowed,
)
from thesis_rag.runner import run_jobs

_ROOT = Path(__file__).resolve().parents[1]
_LOCK_PATH = _ROOT / "artifacts/protocol-lock.json"
_APPROVAL_PATH = _ROOT / "artifacts/main-run-approval.json"
_OUTPUT_ROOT = _ROOT / "artifacts/runs"
_GENERATION_TRACKS = (
    ("forced", "artifacts/manifests/main-forced.jsonl", 900),
    ("e2e", "artifacts/manifests/main-e2e.jsonl", 450),
    ("clean", "artifacts/manifests/main-clean.jsonl", 500),
)


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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config_from_frozen_bytes(contents: bytes) -> ResearchConfig:
    try:
        value = yaml.safe_load(contents.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError("invalid frozen main configuration") from error
    if not isinstance(value, Mapping):
        raise ValueError("frozen main configuration must be a mapping")
    return ResearchConfig.model_validate(value, strict=True)


def _generator(config: ResearchConfig) -> OllamaGeneratorAdapter:
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


def _detector(config: ResearchConfig) -> PromptGuardDetectorAdapter:
    detector = config.detector
    return PromptGuardDetectorAdapter(
        model_id=detector.model_id,
        model_revision=detector.model_revision,
        max_length=detector.max_length,
        overlap_tokens=detector.overlap_tokens,
        malicious_label=detector.malicious_label,
    )


def _current_models(config: ResearchConfig) -> dict[str, ModelRevision]:
    return {
        "generator": ModelRevision(
            id=config.generation.model_id,
            revision=config.generation.model_revision,
        ),
        "embedding": ModelRevision(
            id=config.embedding.model_id,
            revision=config.embedding.model_revision,
        ),
        "detector": ModelRevision(
            id=config.detector.model_id,
            revision=config.detector.model_revision,
        ),
    }


def _verify_completed_generation_records(
    records: tuple[ExperimentRecord, ...],
    *,
    expected_count: int,
    expected_track: str,
) -> None:
    if len(records) != expected_count:
        raise RuntimeError(
            f"{expected_track} result count mismatch: "
            f"{len(records)} != {expected_count}"
        )
    if any(
        record.track != expected_track or record.technical_failure
        for record in records
    ):
        raise RuntimeError(
            f"{expected_track} results contain an invalid or failed record"
        )


def main() -> None:
    current_config = load_config(_ROOT / "configs/main.yaml")
    lock = ProtocolLock.model_validate_json(
        _LOCK_PATH.read_text(encoding="utf-8"),
        strict=True,
    )
    snapshot = assert_main_run_allowed(
        _ROOT,
        lock,
        _APPROVAL_PATH,
        current_models=_current_models(current_config),
        lock_path=_LOCK_PATH,
    )
    config = _config_from_frozen_bytes(snapshot.bytes_for("configs/main.yaml"))
    if config != current_config:
        raise RuntimeError("authorized and current main configurations differ")

    generator = _generator(config)
    detector = _detector(config)
    generator_identity = generator.verify_model()
    detector_identity = detector.verify_model()
    defenses = build_main_defenses(config.detector.threshold)
    _OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    authorization_report_path = (
        _OUTPUT_ROOT / "main-run-authorization.json"
    )
    authorization_report = {
        "schema_version": "1.0.0",
        "protocol_lock_sha256": snapshot.protocol_lock_sha256,
        "protocol_version": snapshot.protocol_version,
        "execution_git_commit": snapshot.execution_git_commit,
        "execution_archive_sha256": snapshot.execution_archive_sha256,
        "generator_identity": generator_identity.model_dump(mode="json"),
        "detector_identity": detector_identity.model_dump(mode="json"),
    }
    _atomic_write(
        authorization_report_path,
        f"{canonical_json(authorization_report)}\n",
    )

    detector_entries = load_frozen_detector_manifest(
        snapshot,
        "artifacts/manifests/main-detector.jsonl",
        expected_count=380,
    )
    if any(
        entry.threshold != config.detector.threshold
        or entry.detector_model_id != config.detector.model_id
        or entry.detector_model_revision != config.detector.model_revision
        for entry in detector_entries
    ):
        raise RuntimeError("frozen detector manifest binding mismatch")
    detector_results_path = _OUTPUT_ROOT / "main-detector-results.jsonl"
    detector_summary_path = _OUTPUT_ROOT / "main-detector-summary.json"
    detector_summary = run_detector_pilot(
        tuple(entry.case for entry in detector_entries),
        detector,
        threshold=config.detector.threshold,
        results_path=detector_results_path,
        summary_path=detector_summary_path,
    )
    if detector_summary.record_count != 380:
        raise RuntimeError("detector-only result count mismatch")

    track_reports: dict[str, dict[str, object]] = {}
    for track, manifest_path, expected_count in _GENERATION_TRACKS:
        jobs = load_frozen_generation_manifest(
            snapshot,
            manifest_path,
            expected_track=track,
            expected_count=expected_count,
        )
        result_path = _OUTPUT_ROOT / f"main-{track}-results.jsonl"
        failure_path = _OUTPUT_ROOT / f"main-{track}-failures.jsonl"
        run_summary = run_jobs(
            jobs,
            result_path=result_path,
            failure_path=failure_path,
            defenses=defenses,
            generator=generator,
            detector=detector,
        )
        if run_summary.technical_failures:
            raise RuntimeError(
                f"{track} produced {run_summary.technical_failures} "
                "technical failures; rerun to resume after diagnosis"
            )
        records = load_records(result_path)
        _verify_completed_generation_records(
            records,
            expected_count=expected_count,
            expected_track=track,
        )
        failure_records = load_failure_records(failure_path)
        track_reports[track] = {
            "expected_count": expected_count,
            "completed_record_count": len(records),
            "historical_failure_attempt_count": len(failure_records),
            "result_sha256": _sha256_file(result_path),
            "failure_audit_sha256": (
                _sha256_file(failure_path)
                if failure_path.exists()
                else None
            ),
        }

    completion_path = _OUTPUT_ROOT / "main-run-completion.json"
    completion = {
        "schema_version": "1.0.0",
        "status": "complete",
        "protocol_lock_sha256": snapshot.protocol_lock_sha256,
        "execution_git_commit": snapshot.execution_git_commit,
        "authorization_report_sha256": _sha256_file(
            authorization_report_path
        ),
        "detector": {
            "record_count": detector_summary.record_count,
            "results_sha256": _sha256_file(detector_results_path),
            "summary_sha256": _sha256_file(detector_summary_path),
        },
        "generation_tracks": track_reports,
    }
    _atomic_write(completion_path, f"{canonical_json(completion)}\n")
    print(
        canonical_json(
            {
                "status": "main_experiment_complete",
                "completion_path": (
                    completion_path.relative_to(_ROOT).as_posix()
                ),
                "completion_sha256": _sha256_file(completion_path),
            }
        )
    )


if __name__ == "__main__":
    main()
