import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from thesis_rag.domain import ExperimentRecord, FailureAttemptRecord


class CorruptJsonlError(ValueError):
    def __init__(self, line_number: int, detail: str) -> None:
        self.line_number = line_number
        super().__init__(f"corrupt JSONL at line {line_number}: {detail}")


class _DuplicateJsonKeyError(ValueError):
    pass


def _reject_non_json_constant(value: str) -> float:
    raise ValueError(f"invalid JSON numeric constant: {value}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateJsonKeyError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _objects(
    path: Path,
    *,
    identity_field: str = "job_id",
) -> Iterator[tuple[int, dict[str, Any]]]:
    if not path.exists():
        return

    contents = path.read_bytes()
    if not contents:
        return
    if not contents.endswith(b"\n"):
        raise CorruptJsonlError(contents.count(b"\n") + 1, "missing final newline")

    seen_identities: set[str] = set()
    for line_number, raw_line in enumerate(contents.splitlines(), start=1):
        try:
            line = raw_line.decode("utf-8")
            value = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_non_json_constant,
            )
        except _DuplicateJsonKeyError as error:
            raise CorruptJsonlError(line_number, str(error)) from error
        except (UnicodeDecodeError, ValueError) as error:
            raise CorruptJsonlError(line_number, "invalid JSON") from error
        if not isinstance(value, dict):
            raise CorruptJsonlError(line_number, "line must be a JSON object")

        identity = value.get(identity_field)
        if not isinstance(identity, str) or not identity.strip():
            raise CorruptJsonlError(
                line_number,
                f"{identity_field} must be a non-empty string",
            )
        if identity in seen_identities:
            raise CorruptJsonlError(
                line_number,
                f"duplicate {identity_field}: {identity}",
            )
        seen_identities.add(identity)
        yield line_number, value


def read_job_ids(path: Path) -> tuple[str, ...]:
    return tuple(value["job_id"] for _, value in _objects(path))


def load_records(path: Path) -> tuple[ExperimentRecord, ...]:
    records: list[ExperimentRecord] = []
    for line_number, value in _objects(path):
        try:
            record = ExperimentRecord.model_validate(value)
        except ValidationError as error:
            raise CorruptJsonlError(line_number, "invalid ExperimentRecord") from error
        if record.technical_failure:
            raise CorruptJsonlError(
                line_number,
                "canonical result records must be nontechnical",
            )
        records.append(record)
    return tuple(records)


def append_record(path: Path, record: ExperimentRecord) -> None:
    existing_records = load_records(path)
    serialized = record.model_dump_json()
    validated = ExperimentRecord.model_validate_json(serialized, strict=True)
    if validated.technical_failure:
        raise ValueError("canonical result record must be nontechnical")
    if validated.job_id in {existing.job_id for existing in existing_records}:
        raise ValueError(f"duplicate job_id: {validated.job_id}")
    validated_serialized = validated.model_dump_json()

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(validated_serialized + "\n")
        output.flush()
        os.fsync(output.fileno())


def load_failure_records(path: Path) -> tuple[FailureAttemptRecord, ...]:
    records: list[FailureAttemptRecord] = []
    next_attempt_by_job: dict[str, int] = {}
    for line_number, value in _objects(path, identity_field="attempt_id"):
        try:
            record = FailureAttemptRecord.model_validate(value)
        except ValidationError as error:
            raise CorruptJsonlError(
                line_number,
                "invalid FailureAttemptRecord",
            ) from error
        expected_attempt = next_attempt_by_job.get(record.job_id, 1)
        if record.attempt_number != expected_attempt:
            raise CorruptJsonlError(
                line_number,
                f"attempt_number for {record.job_id} must be {expected_attempt}",
            )
        next_attempt_by_job[record.job_id] = expected_attempt + 1
        records.append(record)
    return tuple(records)


def append_failure_record(path: Path, record: FailureAttemptRecord) -> None:
    existing_records = load_failure_records(path)
    serialized = record.model_dump_json()
    validated = FailureAttemptRecord.model_validate_json(serialized, strict=True)
    expected_attempt = (
        sum(existing.job_id == validated.job_id for existing in existing_records) + 1
    )
    if validated.attempt_number != expected_attempt:
        raise ValueError(
            f"attempt_number for {validated.job_id} must be {expected_attempt}"
        )
    if validated.attempt_id in {
        existing.attempt_id for existing in existing_records
    }:
        raise ValueError(f"duplicate attempt_id: {validated.attempt_id}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(validated.model_dump_json() + "\n")
        output.flush()
        os.fsync(output.fileno())


load_failure_attempts = load_failure_records
append_failure_attempt = append_failure_record
