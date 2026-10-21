import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from thesis_rag.canonical import canonical_json

_ROOT = Path(__file__).resolve().parents[1]
_REPORT_PATH = _ROOT / "artifacts/reports/pre-freeze-verification.json"


def _run(command: tuple[str, ...]) -> str:
    result = subprocess.run(
        command,
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _match_count(pattern: str, output: str, *, label: str) -> int:
    match = re.search(pattern, output)
    if match is None:
        raise RuntimeError(f"unable to parse {label} verification output")
    return int(match.group(1))


def _json_output(output: str, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(output)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"unable to parse {label} JSON output") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} output must be a JSON object")
    return value


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
    pytest_output = _run((".venv/bin/pytest", "-q"))
    ruff_output = _run((".venv/bin/ruff", "check", "."))
    mypy_output = _run((".venv/bin/mypy", "src"))
    manifest_output = _json_output(
        _run(
            (
                ".venv/bin/python",
                "work/verify_main_manifests.py",
            )
        ),
        label="manifest verification",
    )
    review_output = _json_output(
        _run(
            (
                ".venv/bin/python",
                "work/prepare_main_review_plan.py",
            )
        ),
        label="review-plan verification",
    )
    _run(("git", "diff", "--check"))

    if manifest_output.get("status") != "verified_without_main_model_scoring":
        raise RuntimeError("main manifest verification status is invalid")
    if (
        review_output.get("status")
        != "review_plan_prepared_without_main_model_scoring"
    ):
        raise RuntimeError("main review-plan verification status is invalid")

    report = {
        "schema_version": "1.0.0",
        "status": "passed",
        "checks": {
            "pytest": {
                "command": ".venv/bin/pytest -q",
                "passed_test_count": _match_count(
                    r"(\d+) passed",
                    pytest_output,
                    label="pytest",
                ),
            },
            "ruff": {
                "command": ".venv/bin/ruff check .",
                "passed": ruff_output == "All checks passed!",
            },
            "mypy": {
                "command": ".venv/bin/mypy src",
                "checked_source_file_count": _match_count(
                    r"(\d+) source files",
                    mypy_output,
                    label="mypy",
                ),
            },
            "git_diff_check": {
                "command": "git diff --check",
                "passed": True,
            },
            "main_manifest_verification": manifest_output,
            "main_review_plan_verification": review_output,
        },
        "generation_or_detector_scoring_performed": False,
        "protocol_freeze_approved": False,
        "main_run_approved": False,
    }
    if not report["checks"]["ruff"]["passed"]:
        raise RuntimeError("ruff verification output is unexpected")
    _atomic_write(_REPORT_PATH, f"{canonical_json(report)}\n")
    print(
        canonical_json(
            {
                "status": "pre_freeze_verification_passed",
                "passed_test_count": report["checks"]["pytest"][
                    "passed_test_count"
                ],
                "report_path": _REPORT_PATH.relative_to(_ROOT).as_posix(),
            }
        )
    )


if __name__ == "__main__":
    main()
