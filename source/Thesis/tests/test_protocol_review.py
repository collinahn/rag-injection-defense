from pathlib import Path

import pytest

from thesis_rag.protocol import ModelRevision
from thesis_rag.protocol_review import (
    build_protocol_freeze_candidate,
    verify_protocol_freeze_candidate,
)


def _models() -> dict[str, ModelRevision]:
    return {
        "generator": ModelRevision(id="generator", revision="gen-rev"),
        "embedding": ModelRevision(id="embedding", revision="emb-rev"),
        "detector": ModelRevision(id="detector", revision="det-rev"),
    }


def test_builds_a_stable_sorted_freeze_candidate_with_merged_categories(
    tmp_path: Path,
) -> None:
    first = tmp_path / "data" / "first.json"
    second = tmp_path / "src" / "second.py"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b'{"value":1}\n')
    second.write_bytes(b"VALUE = 2\n")

    candidate = build_protocol_freeze_candidate(
        tmp_path,
        {
            "execution": (Path("src/second.py"),),
            "data": (Path("data/first.json"), Path("src/second.py")),
        },
        models=_models(),
        counts={"main_jobs": 2},
        test_data_approval_id="APPROVAL_TEST",
        pre_freeze_git_commit="a" * 40,
        title="Protocol title",
    )
    repeated = build_protocol_freeze_candidate(
        tmp_path,
        {
            "data": (Path("src/second.py"), Path("data/first.json")),
            "execution": (Path("src/second.py"),),
        },
        models=_models(),
        counts={"main_jobs": 2},
        test_data_approval_id="APPROVAL_TEST",
        pre_freeze_git_commit="a" * 40,
        title="Protocol title",
    )

    assert candidate == repeated
    assert tuple(item.path for item in candidate.files) == (
        "data/first.json",
        "src/second.py",
    )
    assert candidate.files[1].categories == ("data", "execution")
    assert candidate.candidate_id.startswith("PFC_")
    assert candidate.protocol_freeze_approved is False
    assert candidate.main_run_approved is False
    assert candidate.test_data_scoring_performed is False
    assert candidate.local_commit_required_after_approval is True


def test_rejects_missing_outside_and_symlinked_candidate_files(
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / "outside-freeze-candidate.txt"
    outside.write_text("outside", encoding="utf-8")
    regular = tmp_path / "regular.txt"
    regular.write_text("regular", encoding="utf-8")
    symlink = tmp_path / "alias.txt"
    symlink.symlink_to(regular)

    common = {
        "models": _models(),
        "counts": {"main_jobs": 1},
        "test_data_approval_id": "APPROVAL_TEST",
        "pre_freeze_git_commit": "a" * 40,
        "title": "Protocol title",
    }
    with pytest.raises(FileNotFoundError, match="missing"):
        build_protocol_freeze_candidate(
            tmp_path,
            {"data": (Path("missing.txt"),)},
            **common,
        )
    with pytest.raises(ValueError, match="outside"):
        build_protocol_freeze_candidate(
            tmp_path,
            {"data": (outside,)},
            **common,
        )
    with pytest.raises(ValueError, match="symlink"):
        build_protocol_freeze_candidate(
            tmp_path,
            {"data": (Path("alias.txt"),)},
            **common,
        )


def test_verification_reports_changed_and_missing_files(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    candidate = build_protocol_freeze_candidate(
        tmp_path,
        {"data": (Path("first.txt"), Path("second.txt"))},
        models=_models(),
        counts={"main_jobs": 2},
        test_data_approval_id="APPROVAL_TEST",
        pre_freeze_git_commit="a" * 40,
        title="Protocol title",
    )

    first.write_text("changed", encoding="utf-8")
    second.unlink()
    verification = verify_protocol_freeze_candidate(tmp_path, candidate)

    assert verification.valid is False
    assert verification.changed_files == ("first.txt",)
    assert verification.missing_files == ("second.txt",)
