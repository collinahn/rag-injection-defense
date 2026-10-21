import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from thesis_rag.domain import ExperimentRecord, FailureAttemptRecord
from thesis_rag.jsonl import (
    CorruptJsonlError,
    append_failure_record,
    append_record,
    load_failure_records,
    load_records,
    read_job_ids,
)


def _record(*, job_id: str = "JOB_001", raw_response: str = "근거 있는 답변") -> ExperimentRecord:
    return ExperimentRecord(
        experiment_id=f"EXP_{job_id}",
        job_id=job_id,
        track="clean",
        question_id="Q_001",
        base_context_id="CTX_001",
        attack_id=None,
        defense_id="D0",
        config_hash="CFG_001",
        clean_source_commitment=None,
        attack_commitment=None,
        model_id="mock-generator",
        model_revision="v1",
        detector_model_id="mock-detector",
        detector_model_revision="v1",
        research_input_hash="b" * 64,
        prompt_hash="a" * 64,
        ordered_chunk_ids=("CH_001",),
        malicious_chunk_rank=None,
        detector_scores={"CH_001": 0.0},
        blocked_chunk_ids=(),
        raw_response=raw_response,
        attack_success=None,
        technical_failure=False,
        error_type=None,
        latency_ms=3.0,
        detector_latency_ms=1.0,
        generation_latency_ms=1.0,
        input_tokens=4,
        output_tokens=2,
    )


def _failure_attempt(
    *,
    attempt_number: int,
    job_id: str = "JOB_001",
    error_type: str = "RuntimeError",
) -> FailureAttemptRecord:
    failure = _record(job_id=job_id).model_copy(
        update={
            "raw_response": "",
            "technical_failure": True,
            "error_type": error_type,
        }
    )
    return FailureAttemptRecord.from_record(
        failure,
        attempt_number=attempt_number,
    )


@pytest.mark.parametrize("create_empty", [False, True])
def test_read_job_ids_treats_absent_and_empty_files_as_no_completed_jobs(
    tmp_path: Path, create_empty: bool
) -> None:
    path = tmp_path / "run.jsonl"
    if create_empty:
        path.touch()

    assert read_job_ids(path) == ()


@pytest.mark.parametrize(
    ("contents", "line_number"),
    [
        ('{"job_id":"JOB_001"}\nnot-json\n{"job_id":"JOB_003"}\n', 2),
        ('{"job_id":"JOB_001"}\nnot-json\n', 2),
    ],
)
def test_read_job_ids_rejects_malformed_complete_lines_with_their_line_number(
    tmp_path: Path, contents: str, line_number: int
) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(CorruptJsonlError, match=rf"line {line_number}\b"):
        read_job_ids(path)


def test_read_job_ids_rejects_valid_json_without_a_final_newline(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text('{"job_id":"JOB_001"}', encoding="utf-8")

    with pytest.raises(CorruptJsonlError, match=r"line 1\b"):
        read_job_ids(path)


def test_read_job_ids_rejects_nonstandard_json_numeric_constants(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text('{"job_id":"JOB_001","score":NaN}\n', encoding="utf-8")

    with pytest.raises(CorruptJsonlError, match=r"line 1\b"):
        read_job_ids(path)


@pytest.mark.parametrize(
    "line",
    [
        "[]",
        '{"job_id":null}',
        '{"job_id":""}',
        '{"job_id":"   "}',
    ],
)
def test_read_job_ids_requires_each_line_to_have_a_nonempty_string_job_id(
    tmp_path: Path, line: str
) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text(f"{line}\n", encoding="utf-8")

    with pytest.raises(CorruptJsonlError, match=r"line 1\b"):
        read_job_ids(path)


def test_read_job_ids_rejects_duplicate_existing_job_ids_at_the_second_line(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text('{"job_id":"JOB_001"}\n{"job_id":"JOB_001"}\n', encoding="utf-8")

    with pytest.raises(CorruptJsonlError, match=r"line 2\b.*duplicate job_id"):
        read_job_ids(path)


def test_load_records_round_trips_complete_experiment_records(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    records = (_record(job_id="JOB_001"), _record(job_id="JOB_002"))
    path.write_text(
        "".join(record.model_dump_json() + "\n" for record in records),
        encoding="utf-8",
    )

    assert load_records(path) == records


def test_load_records_reports_the_line_of_an_invalid_experiment_record(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    valid = _record(job_id="JOB_001").model_dump(mode="json")
    invalid = valid | {"job_id": "JOB_002", "technical_failure": False, "attack_success": None}
    invalid["track"] = "forced"
    path.write_text(
        json.dumps(valid, ensure_ascii=False)
        + "\n"
        + json.dumps(invalid, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CorruptJsonlError, match=r"line 2\b"):
        load_records(path)


def test_load_records_requires_config_hash_in_every_experiment_record(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run.jsonl"
    payload = _record().model_dump(mode="json")
    payload.pop("config_hash", None)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(CorruptJsonlError, match=r"line 1\b"):
        load_records(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("detector_scores", {"CH_001": -0.01}),
        ("detector_scores", {"CH_001": 1.01}),
        ("detector_scores", {"CH_001": float("nan")}),
        ("detector_scores", {"CH_001": float("inf")}),
        ("detector_scores", {"CH_001": "0.5"}),
        ("latency_ms", -0.01),
        ("latency_ms", float("nan")),
        ("detector_latency_ms", float("inf")),
        ("generation_latency_ms", "1.0"),
        ("malicious_chunk_rank", 0),
        ("input_tokens", -1),
        ("output_tokens", "2"),
    ],
)
def test_load_records_rejects_invalid_persisted_numeric_states(
    tmp_path: Path, field: str, value: object
) -> None:
    path = tmp_path / "run.jsonl"
    payload = _record().model_dump(mode="json") | {field: value}
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(CorruptJsonlError, match=r"line 1\b"):
        load_records(path)


def test_load_records_rejects_json_numeric_overflow(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    serialized = _record().model_dump_json()
    serialized = serialized.replace('"latency_ms":3.0', '"latency_ms":1e999')
    path.write_text(serialized + "\n", encoding="utf-8")

    with pytest.raises(CorruptJsonlError, match=r"line 1\b"):
        load_records(path)


@pytest.mark.parametrize(
    ("technical_failure", "error_type"),
    [
        (True, None),
        (True, ""),
        (True, " \t"),
        (False, "unexpected_error"),
    ],
)
def test_load_records_requires_error_type_exactly_for_technical_failures(
    tmp_path: Path,
    technical_failure: bool,
    error_type: str | None,
) -> None:
    path = tmp_path / "run.jsonl"
    payload = _record().model_dump(mode="json") | {
        "technical_failure": technical_failure,
        "error_type": error_type,
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(CorruptJsonlError, match=r"line 1\b"):
        load_records(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("technical_failure", "false"),
        ("technical_failure", 0),
        ("attack_success", "false"),
        ("attack_success", 0),
    ],
)
def test_load_records_rejects_coercive_persisted_booleans_with_line_number(
    tmp_path: Path, field: str, value: object
) -> None:
    path = tmp_path / "run.jsonl"
    invalid = _record(job_id="JOB_002").model_dump(mode="json")
    if field == "attack_success":
        invalid |= {
            "track": "forced",
            "attack_id": "ATK_001",
            "clean_source_commitment": "a" * 64,
            "attack_commitment": "b" * 64,
            "malicious_chunk_rank": 1,
        }
    invalid[field] = value
    path.write_text(
        _record(job_id="JOB_001").model_dump_json()
        + "\n"
        + json.dumps(invalid)
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CorruptJsonlError, match=r"line 2\b"):
        load_records(path)


@pytest.mark.parametrize(
    "mutation",
    [
        {"clean_source_commitment": "a" * 63},
        {"attack_commitment": "not-hex" * 9 + "x"},
        {"clean_source_commitment": "a" * 64},
    ],
)
def test_load_records_rejects_invalid_or_inconsistent_commitments(
    tmp_path: Path, mutation: dict[str, object]
) -> None:
    path = tmp_path / "run.jsonl"
    payload = _record().model_dump(mode="json") | mutation
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(CorruptJsonlError, match=r"line 1\b"):
        load_records(path)


@pytest.mark.parametrize(
    ("track", "technical_failure", "attack_success"),
    [
        ("clean", False, True),
        ("detector", False, False),
        ("forced", True, True),
    ],
)
def test_load_records_rejects_attack_outcomes_outside_the_truth_table(
    tmp_path: Path,
    track: str,
    technical_failure: bool,
    attack_success: bool,
) -> None:
    path = tmp_path / "run.jsonl"
    payload = _record().model_dump(mode="json") | {
        "track": track,
        "technical_failure": technical_failure,
        "attack_success": attack_success,
        "error_type": "RuntimeError" if technical_failure else None,
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(CorruptJsonlError, match=r"line 1\b"):
        load_records(path)


def test_load_records_rejects_an_empty_nontechnical_response_with_line_number(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run.jsonl"
    payload = _record().model_dump(mode="json") | {"raw_response": " \t\n"}
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(CorruptJsonlError, match=r"line 1\b"):
        load_records(path)


def test_append_record_rejects_a_duplicate_job_id(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    append_record(path, _record())

    with pytest.raises(ValueError, match="duplicate job_id"):
        append_record(path, _record(raw_response="different response"))


def test_append_record_does_not_extend_a_schema_invalid_result_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run.jsonl"
    invalid = _record(job_id="JOB_EXISTING").model_dump(mode="json")
    invalid.pop("config_hash")
    original = (json.dumps(invalid) + "\n").encode()
    path.write_bytes(original)

    with pytest.raises(CorruptJsonlError, match=r"line 1\b"):
        append_record(path, _record(job_id="JOB_NEW"))

    assert path.read_bytes() == original


def test_append_record_strict_existing_file_corruption_wins_over_invalid_input(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run.jsonl"
    invalid_existing = _record(job_id="JOB_EXISTING").model_dump(mode="json")
    invalid_existing.pop("config_hash")
    original = (json.dumps(invalid_existing) + "\n").encode()
    path.write_bytes(original)
    valid = _record(job_id="JOB_NEW")
    incoming_payload = {
        field: getattr(valid, field)
        for field in ExperimentRecord.model_fields
    }
    del incoming_payload["config_hash"]
    invalid_incoming = ExperimentRecord.model_construct(**incoming_payload)

    with pytest.raises(CorruptJsonlError, match=r"line 1\b"):
        append_record(path, invalid_incoming)

    assert path.read_bytes() == original


@pytest.mark.parametrize("job_id", ["", " \t\n"])
def test_experiment_record_domain_rejects_blank_job_ids(job_id: str) -> None:
    payload = _record().model_dump(mode="json") | {"job_id": job_id}

    with pytest.raises(ValidationError, match="job_id must be a non-empty string"):
        ExperimentRecord.model_validate(payload)


@pytest.mark.parametrize("job_id", ["", " \t\n"])
def test_append_record_rejects_a_forged_blank_job_id_without_writing(
    tmp_path: Path, job_id: str
) -> None:
    path = tmp_path / "uncreated" / "run.jsonl"
    record = _record().model_copy(update={"job_id": job_id})

    with pytest.raises(ValueError, match="job_id must be a non-empty string"):
        append_record(path, record)

    assert not path.parent.exists()


@pytest.mark.parametrize(
    "corruption",
    [
        "missing-config",
        "nonnull-clean-outcome",
        "nonfinite-latency",
        "nonfinite-detector-score",
    ],
)
def test_append_record_revalidates_the_exact_serialized_record_before_writing(
    tmp_path: Path, corruption: str
) -> None:
    valid = _record()
    payload = {
        field: getattr(valid, field)
        for field in ExperimentRecord.model_fields
    }
    if corruption == "missing-config":
        del payload["config_hash"]
    elif corruption == "nonnull-clean-outcome":
        payload["attack_success"] = True
    elif corruption == "nonfinite-latency":
        payload["latency_ms"] = float("nan")
    else:
        payload["detector_scores"] = {"CH_001": float("inf")}
    record = ExperimentRecord.model_construct(**payload)
    path = tmp_path / "uncreated" / "run.jsonl"

    with pytest.raises(ValidationError):
        append_record(path, record)

    assert not path.parent.exists()


def test_append_record_writes_the_validated_canonical_serialization(
    tmp_path: Path,
) -> None:
    class PrettySerializedRecord(ExperimentRecord):
        def model_dump_json(self, **kwargs: object) -> str:
            return json.dumps(self.model_dump(mode="json"), indent=2)

    valid = _record()
    record = PrettySerializedRecord.model_validate(valid.model_dump(mode="python"))
    assert "\n" in record.model_dump_json()
    path = tmp_path / "run.jsonl"

    append_record(path, record)

    assert path.read_text(encoding="utf-8") == valid.model_dump_json() + "\n"


def test_append_record_creates_parents_and_flushes_before_one_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "nested" / "run.jsonl"
    record = _record()
    fsync_calls: list[int] = []

    def observe_fsync(file_descriptor: int) -> None:
        fsync_calls.append(file_descriptor)
        persisted = path.read_text(encoding="utf-8")
        assert persisted == record.model_dump_json() + "\n"

    monkeypatch.setattr(os, "fsync", observe_fsync)

    append_record(path, record)

    assert len(fsync_calls) == 1
    assert path.read_bytes().endswith(b"\n")


def test_canonical_append_rejects_a_technical_failure_record_without_writing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run.jsonl"
    failure = _record().model_copy(
        update={
            "raw_response": "",
            "technical_failure": True,
            "error_type": "RuntimeError",
        }
    )

    with pytest.raises(ValueError, match="canonical result.*nontechnical"):
        append_record(path, failure)

    assert not path.exists()


def test_failure_audit_allows_numbered_attempts_for_the_same_job(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run.failures.jsonl"
    first = _failure_attempt(attempt_number=1)
    second = _failure_attempt(attempt_number=2, error_type="TimeoutError")

    append_failure_record(path, first)
    append_failure_record(path, second)

    assert load_failure_records(path) == (first, second)
    assert [record.job_id for record in load_failure_records(path)] == [
        "JOB_001",
        "JOB_001",
    ]


def test_failure_audit_rejects_duplicate_or_nonsequential_attempts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run.failures.jsonl"
    append_failure_record(path, _failure_attempt(attempt_number=1))

    with pytest.raises(ValueError, match="attempt_number"):
        append_failure_record(path, _failure_attempt(attempt_number=3))

    assert len(load_failure_records(path)) == 1


def test_failure_audit_strictly_preflights_existing_lines_before_append(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run.failures.jsonl"
    original = b'{"attempt_id":"ATTEMPT_CORRUPT","job_id":"JOB_001"}\n'
    path.write_bytes(original)

    with pytest.raises(CorruptJsonlError, match=r"line 1\b"):
        append_failure_record(path, _failure_attempt(attempt_number=1))

    assert path.read_bytes() == original
