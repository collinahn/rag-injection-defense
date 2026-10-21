import json
from pathlib import Path

import pytest

from thesis_rag import cli
from thesis_rag.canonical import stable_id
from thesis_rag.cli import run_fixture_pipeline, verify_fixture_results

CONFIG_PATH = Path("configs/development.yaml")


def _run_fixture_pipeline(tmp_path: Path) -> Path:
    result_path = tmp_path / "fixture-run.jsonl"
    run_fixture_pipeline(config_path=CONFIG_PATH, result_path=result_path)
    return result_path


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_records(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


def _experiment_id(record: dict[str, object]) -> str:
    return stable_id(
        "EXP",
        {
            "job_id": record["job_id"],
            "model_id": record["model_id"],
            "model_revision": record["model_revision"],
            "detector_model_id": record["detector_model_id"],
            "detector_model_revision": record["detector_model_revision"],
        },
    )


def test_fixture_pipeline_produces_paired_defense_records_and_resumes(tmp_path: Path) -> None:
    result_path = tmp_path / "fixture-run.jsonl"

    first = run_fixture_pipeline(config_path=CONFIG_PATH, result_path=result_path)
    before_resume = result_path.read_bytes()
    resumed = run_fixture_pipeline(config_path=CONFIG_PATH, result_path=result_path)
    verified = verify_fixture_results(result_path)
    records = _records(result_path)

    assert first.jobs_expected == 5
    assert first.jobs_completed == 5
    assert first.jobs_skipped == 0
    assert first.duplicate_jobs == 0
    assert first.technical_failures == 0
    assert first.defense_ids == ("D0", "D1", "D2", "D3", "D4")
    assert resumed.jobs_completed == 0
    assert resumed.jobs_skipped == 5
    assert result_path.read_bytes() == before_resume
    assert verified == first
    assert [record["defense_id"] for record in records] == ["D0", "D1", "D2", "D3", "D4"]
    assert len({record["job_id"] for record in records}) == 5
    for field in (
        "track",
        "question_id",
        "base_context_id",
        "attack_id",
        "config_hash",
        "model_id",
        "model_revision",
        "detector_model_id",
        "detector_model_revision",
        "research_input_hash",
        "ordered_chunk_ids",
        "malicious_chunk_rank",
    ):
        assert len({json.dumps(record[field], sort_keys=True) for record in records}) == 1
    assert records[0]["model_id"] == "mock-generator"
    assert records[0]["model_revision"] == "v1"
    assert records[0]["detector_model_id"] == "mock-detector"
    assert records[0]["detector_model_revision"] == "v1"
    assert all(record["technical_failure"] is False for record in records)
    assert [record["attack_success"] for record in records] == [True, True, True, False, False]
    assert records[3]["blocked_chunk_ids"] == records[4]["blocked_chunk_ids"]


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("not-json", "corrupt JSONL"),
        ("missing-field", "invalid ExperimentRecord"),
        ("extra-field", "invalid ExperimentRecord"),
        ("duplicate-job", "duplicate job_id"),
        ("duplicate-defense", "duplicate defense IDs"),
        ("unequal-config", "unequal paired field: config_hash"),
        ("technical-failure", "canonical result records must be nontechnical"),
        ("unequal-filtering", "D3 and D4 blocked chunk IDs differ"),
    ],
)
def test_fixture_verification_rejects_invalid_record_sets(
    tmp_path: Path, corruption: str, message: str
) -> None:
    result_path = _run_fixture_pipeline(tmp_path)
    records = _records(result_path)

    if corruption == "not-json":
        result_path.write_text("not-json\n", encoding="utf-8")
    elif corruption == "missing-field":
        records[0].pop("config_hash")
        _write_records(result_path, records)
    elif corruption == "extra-field":
        records[0]["unexpected"] = True
        _write_records(result_path, records)
    elif corruption == "duplicate-job":
        records[1]["job_id"] = records[0]["job_id"]
        _write_records(result_path, records)
    elif corruption == "duplicate-defense":
        records[4]["defense_id"] = "D3"
        _write_records(result_path, records)
    elif corruption == "unequal-config":
        records[1]["config_hash"] = "CFG_OTHER"
        _write_records(result_path, records)
    elif corruption == "technical-failure":
        records[0]["technical_failure"] = True
        records[0]["attack_success"] = None
        records[0]["error_type"] = "RuntimeError"
        _write_records(result_path, records)
    else:
        records[4]["blocked_chunk_ids"] = []
        _write_records(result_path, records)

    with pytest.raises(ValueError, match=message):
        verify_fixture_results(result_path)


@pytest.mark.parametrize(
    ("rank", "message"),
    [
        (0, "invalid ExperimentRecord"),
        (3, "malicious_chunk_rank must be within ordered chunk IDs"),
    ],
)
def test_fixture_verification_rejects_malicious_ranks_outside_the_context(
    tmp_path: Path, rank: int, message: str
) -> None:
    result_path = _run_fixture_pipeline(tmp_path)
    records = _records(result_path)
    for record in records:
        record["malicious_chunk_rank"] = rank
    _write_records(result_path, records)

    with pytest.raises(ValueError, match=message):
        verify_fixture_results(result_path)


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("duplicate-job", "fixture jobs must have unique job IDs"),
        ("unequal-question", "fixture jobs must share paired field: question_id"),
        (
            "unequal-research-input",
            "fixture jobs must share paired field: research_input_hash",
        ),
    ],
)
def test_fixture_pipeline_preflights_job_pairing_before_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, corruption: str, message: str
) -> None:
    build_jobs = cli.build_forced_jobs

    def corrupted_jobs(*args: object, **kwargs: object) -> tuple[object, ...]:
        jobs = build_jobs(*args, **kwargs)  # type: ignore[arg-type]
        if corruption == "duplicate-job":
            return jobs[:1] + (jobs[1].model_copy(update={"job_id": jobs[0].job_id}),) + jobs[2:]
        if corruption == "unequal-question":
            changed = jobs[1].model_copy(update={"question_id": "Q_UNPAIRED"})
        else:
            changed = jobs[1].model_copy(update={"research_input_hash": "f" * 64})
        return jobs[:1] + (changed,) + jobs[2:]

    monkeypatch.setattr(cli, "build_forced_jobs", corrupted_jobs)
    result_path = tmp_path / "fixture-run.jsonl"

    with pytest.raises(ValueError, match=message):
        run_fixture_pipeline(config_path=CONFIG_PATH, result_path=result_path)

    assert not result_path.exists()


@pytest.mark.parametrize(
    ("tampering", "message"),
    [
        ("raw-response", "attack_success does not match recomputed response"),
        ("experiment-id", "experiment_id does not match frozen adapter identities"),
        ("error-type", "invalid ExperimentRecord"),
        ("prompt-hash", "prompt_hash does not match reconstructed defense prompt"),
    ],
)
def test_fixture_resume_and_verification_reject_tampered_record_derivations(
    tmp_path: Path, tampering: str, message: str
) -> None:
    result_path = _run_fixture_pipeline(tmp_path)
    records = _records(result_path)

    if tampering == "raw-response":
        records[0]["raw_response"] = "grounded answer without the success marker"
    elif tampering == "experiment-id":
        records[0]["experiment_id"] = "EXP_TAMPERED"
    elif tampering == "error-type":
        records[0]["error_type"] = "RuntimeError"
    else:
        records[0]["prompt_hash"] = "0" * 64
    _write_records(result_path, records)

    with pytest.raises(ValueError, match=message):
        verify_fixture_results(result_path)
    with pytest.raises(ValueError, match=message):
        run_fixture_pipeline(config_path=CONFIG_PATH, result_path=result_path)


def test_fixture_resume_and_verification_reject_marker_preserving_response_changes(
    tmp_path: Path,
) -> None:
    result_path = _run_fixture_pipeline(tmp_path)
    records = _records(result_path)
    records[0]["raw_response"] = f"{records[0]['raw_response']} appended-but-not-generated"
    _write_records(result_path, records)

    with pytest.raises(ValueError, match="raw_response does not match frozen MockGenerator replay"):
        verify_fixture_results(result_path)
    with pytest.raises(ValueError, match="raw_response does not match frozen MockGenerator replay"):
        run_fixture_pipeline(config_path=CONFIG_PATH, result_path=result_path)


@pytest.mark.parametrize("token_field", ["input_tokens", "output_tokens"])
def test_fixture_resume_and_verification_reject_replayed_token_field_changes(
    tmp_path: Path, token_field: str
) -> None:
    result_path = _run_fixture_pipeline(tmp_path)
    records = _records(result_path)
    records[0][token_field] = 0
    _write_records(result_path, records)

    mismatch = f"{token_field} does not match frozen MockGenerator replay"
    with pytest.raises(ValueError, match=mismatch):
        verify_fixture_results(result_path)
    with pytest.raises(ValueError, match=mismatch):
        run_fixture_pipeline(config_path=CONFIG_PATH, result_path=result_path)


@pytest.mark.parametrize(
    ("record_count", "tampering", "message"),
    [
        (1, "raw-response", "attack_success does not match recomputed response"),
        (2, "unknown-job", "unknown fixture job_id"),
        (3, "duplicate-defense", "duplicate defense IDs"),
        (4, "config-hash", "fixture record does not match reconstructed job field: config_hash"),
    ],
)
def test_fixture_partial_resume_rejects_every_tampered_existing_record(
    tmp_path: Path, record_count: int, tampering: str, message: str
) -> None:
    result_path = _run_fixture_pipeline(tmp_path)
    records = _records(result_path)[:record_count]

    if tampering == "raw-response":
        records[0]["raw_response"] = "grounded answer without the success marker"
    elif tampering == "unknown-job":
        records[0]["job_id"] = "JOB_UNKNOWN"
    elif tampering == "duplicate-defense":
        records[1]["defense_id"] = "D0"
    else:
        records[0]["config_hash"] = "CFG_TAMPERED"
    _write_records(result_path, records)
    before_resume = result_path.read_bytes()

    with pytest.raises(ValueError, match=message):
        run_fixture_pipeline(config_path=CONFIG_PATH, result_path=result_path)

    assert result_path.read_bytes() == before_resume


def test_fixture_partial_resume_rejects_marker_preserving_response_changes(
    tmp_path: Path,
) -> None:
    result_path = _run_fixture_pipeline(tmp_path)
    records = _records(result_path)[:1]
    records[0]["raw_response"] = f"{records[0]['raw_response']} appended-but-not-generated"
    _write_records(result_path, records)
    before_resume = result_path.read_bytes()

    with pytest.raises(ValueError, match="raw_response does not match frozen MockGenerator replay"):
        run_fixture_pipeline(config_path=CONFIG_PATH, result_path=result_path)

    assert result_path.read_bytes() == before_resume


def test_fixture_resume_and_verification_reject_coordinated_generator_identity_tampering(
    tmp_path: Path,
) -> None:
    result_path = _run_fixture_pipeline(tmp_path)
    records = _records(result_path)
    for record in records:
        record["model_id"] = "mock-generator-tampered"
        record["model_revision"] = "v999"
        record["experiment_id"] = _experiment_id(record)
    _write_records(result_path, records)

    with pytest.raises(ValueError, match="frozen MockGenerator identity"):
        verify_fixture_results(result_path)
    with pytest.raises(ValueError, match="frozen MockGenerator identity"):
        run_fixture_pipeline(config_path=CONFIG_PATH, result_path=result_path)


def test_fixture_resume_and_verification_reject_coordinated_detector_identity_tampering(
    tmp_path: Path,
) -> None:
    result_path = _run_fixture_pipeline(tmp_path)
    records = _records(result_path)
    for record in records:
        record["detector_model_id"] = "mock-detector-tampered"
        record["detector_model_revision"] = "v999"
        record["experiment_id"] = _experiment_id(record)
    _write_records(result_path, records)

    with pytest.raises(ValueError, match="frozen MockDetector identity"):
        verify_fixture_results(result_path)
    with pytest.raises(ValueError, match="frozen MockDetector identity"):
        run_fixture_pipeline(config_path=CONFIG_PATH, result_path=result_path)


def test_fixture_resume_and_verification_reject_coordinated_research_hash_tampering(
    tmp_path: Path,
) -> None:
    result_path = _run_fixture_pipeline(tmp_path)
    records = _records(result_path)
    for record in records:
        record["research_input_hash"] = "f" * 64
    _write_records(result_path, records)

    mismatch = "fixture record does not match reconstructed job field: research_input_hash"
    with pytest.raises(ValueError, match=mismatch):
        verify_fixture_results(result_path)
    with pytest.raises(ValueError, match=mismatch):
        run_fixture_pipeline(config_path=CONFIG_PATH, result_path=result_path)
