from collections.abc import Mapping
from typing import Literal, Protocol, cast

from thesis_rag.canonical import canonical_json
from thesis_rag.defenses.base import Defense
from thesis_rag.defenses.baseline import D0Defense, D1Defense
from thesis_rag.defenses.datamarking import D2Defense
from thesis_rag.defenses.filtering import D3Defense, D4Defense
from thesis_rag.domain import DefenseId, ExperimentJob
from thesis_rag.main_manifest import DetectorManifestEntry
from thesis_rag.manifests import reconstruct_job_from_research_inputs


class FrozenInputReader(Protocol):
    def bytes_for(self, relative_path: str) -> bytes: ...


def load_frozen_generation_manifest(
    snapshot: FrozenInputReader,
    relative_path: str,
    *,
    expected_track: Literal["forced", "e2e", "clean"],
    expected_count: int,
) -> tuple[ExperimentJob, ...]:
    lines, contents = _canonical_jsonl_lines(snapshot, relative_path)
    try:
        jobs = tuple(
            ExperimentJob.model_validate_json(line, strict=True)
            for line in lines
        )
    except ValueError as error:
        raise ValueError("invalid frozen generation manifest record") from error
    if len(jobs) != expected_count:
        raise ValueError("frozen generation manifest count mismatch")
    if len({job.job_id for job in jobs}) != len(jobs):
        raise ValueError("frozen generation manifest contains duplicate job IDs")
    if any(job.track != expected_track for job in jobs):
        raise ValueError("frozen generation manifest track mismatch")
    canonical_contents = "".join(
        f"{canonical_json(job)}\n" for job in jobs
    ).encode("utf-8")
    if canonical_contents != contents:
        raise ValueError("frozen generation manifest is not canonical")
    reconstructed = tuple(
        reconstruct_job_from_research_inputs(job) for job in jobs
    )
    if reconstructed != jobs:
        raise ValueError("frozen generation manifest reconstruction mismatch")
    return jobs


def load_frozen_detector_manifest(
    snapshot: FrozenInputReader,
    relative_path: str,
    *,
    expected_count: int,
) -> tuple[DetectorManifestEntry, ...]:
    lines, contents = _canonical_jsonl_lines(snapshot, relative_path)
    try:
        entries = tuple(
            DetectorManifestEntry.model_validate_json(line, strict=True)
            for line in lines
        )
    except ValueError as error:
        raise ValueError("invalid frozen detector manifest record") from error
    if len(entries) != expected_count:
        raise ValueError("frozen detector manifest count mismatch")
    if len({entry.job_id for entry in entries}) != len(entries):
        raise ValueError("frozen detector manifest contains duplicate job IDs")
    if len({entry.case.case_id for entry in entries}) != len(entries):
        raise ValueError("frozen detector manifest contains duplicate case IDs")
    canonical_contents = "".join(
        f"{canonical_json(entry)}\n" for entry in entries
    ).encode("utf-8")
    if canonical_contents != contents:
        raise ValueError("frozen detector manifest is not canonical")
    return entries


def build_main_defenses(threshold: float) -> Mapping[DefenseId, Defense]:
    return {
        DefenseId.D0: D0Defense(),
        DefenseId.D1: D1Defense(),
        DefenseId.D2: D2Defense(),
        DefenseId.D3: cast(Defense, D3Defense(threshold=threshold)),
        DefenseId.D4: cast(Defense, D4Defense(threshold=threshold)),
    }


def _canonical_jsonl_lines(
    snapshot: FrozenInputReader,
    relative_path: str,
) -> tuple[tuple[str, ...], bytes]:
    try:
        contents = snapshot.bytes_for(relative_path)
        decoded = contents.decode("utf-8")
    except (KeyError, UnicodeDecodeError) as error:
        raise ValueError(
            f"unable to read authorized snapshot input: {relative_path}"
        ) from error
    if not contents.endswith(b"\n"):
        raise ValueError("frozen manifest must end with one newline")
    lines = tuple(decoded.splitlines())
    if not lines or any(not line for line in lines):
        raise ValueError("frozen manifest must contain nonblank JSONL records")
    return lines, contents
