import os
import tempfile
from pathlib import Path

from thesis_rag.canonical import canonical_json
from thesis_rag.manifest_verification import verify_main_manifest_suite

_ROOT = Path(__file__).resolve().parents[1]


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
    report = verify_main_manifest_suite(_ROOT)
    report_path = (
        _ROOT / "artifacts/reports/main-manifest-verification.json"
    )
    _atomic_write(report_path, f"{canonical_json(report)}\n")
    print(
        canonical_json(
            {
                "status": "verified_without_main_model_scoring",
                "report_path": report_path.relative_to(_ROOT).as_posix(),
                "total_job_count": report.total_job_count,
                "e2e_external_verification_count": (
                    report.e2e.external_verification_count
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
