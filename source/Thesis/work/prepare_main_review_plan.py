import hashlib
import os
import tempfile
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from thesis_rag.canonical import canonical_json
from thesis_rag.domain import ExperimentJob
from thesis_rag.main_human_review import build_main_clean_review_plan
from thesis_rag.manifest_verification import verify_main_manifest_suite
from thesis_rag.qa import ResearchQuestion

_ROOT = Path(__file__).resolve().parents[1]
_RecordT = TypeVar("_RecordT", bound=BaseModel)


def _read_jsonl(
    path: Path,
    model: type[_RecordT],
) -> tuple[_RecordT, ...]:
    return tuple(
        model.model_validate_json(line, strict=True)
        for line in path.read_text(encoding="utf-8").splitlines()
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


def main() -> None:
    if (_ROOT / "artifacts/protocol-lock.json").exists():
        raise RuntimeError(
            "refusing to change the review plan after protocol lock creation"
        )
    suite = verify_main_manifest_suite(_ROOT)
    jobs = _read_jsonl(
        _ROOT / "artifacts/manifests/main-clean.jsonl",
        ExperimentJob,
    )
    questions = _read_jsonl(
        _ROOT / "data/questions/test.jsonl",
        ResearchQuestion,
    )
    assignments, summary = build_main_clean_review_plan(
        jobs,
        questions,
        seed=42,
    )
    assignment_path = (
        _ROOT / "artifacts/review/main-clean-review-assignments.jsonl"
    )
    summary_path = (
        _ROOT / "artifacts/reports/main-clean-review-plan-summary.json"
    )
    assignment_contents = "".join(
        f"{canonical_json(assignment)}\n"
        for assignment in assignments
    )
    _atomic_write(assignment_path, assignment_contents)
    _atomic_write(summary_path, f"{canonical_json(summary)}\n")
    print(
        canonical_json(
            {
                "status": "review_plan_prepared_without_main_model_scoring",
                "main_manifest_suite_valid": suite.valid,
                "primary_item_count": summary.primary_item_count,
                "secondary_item_count": summary.secondary_item_count,
                "assignment_path": (
                    assignment_path.relative_to(_ROOT).as_posix()
                ),
                "assignment_sha256": _sha256_file(assignment_path),
                "summary_sha256": _sha256_file(summary_path),
            }
        )
    )


if __name__ == "__main__":
    main()
