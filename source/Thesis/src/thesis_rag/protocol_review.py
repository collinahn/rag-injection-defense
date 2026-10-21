import hashlib
import os
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import Field, StrictBool, field_validator, model_validator

from thesis_rag.canonical import canonical_sha256
from thesis_rag.domain import FrozenRecord, Sha256Hex
from thesis_rag.protocol import ModelRevision

_REQUIRED_MODEL_ROLES = frozenset({"generator", "embedding", "detector"})


class ProtocolFreezeFile(FrozenRecord):
    path: str
    categories: tuple[str, ...] = Field(min_length=1)
    sha256: Sha256Hex
    size_bytes: int = Field(strict=True, ge=0)

    @field_validator("path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        normalized = PurePosixPath(value)
        if (
            not value
            or normalized.is_absolute()
            or normalized.as_posix() != value
            or any(part in {"", ".", ".."} for part in normalized.parts)
        ):
            raise ValueError("freeze candidate paths must be normalized and relative")
        return value

    @field_validator("categories")
    @classmethod
    def validate_categories(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if tuple(sorted(set(value))) != value or any(
            not category.strip() for category in value
        ):
            raise ValueError("freeze candidate categories must be unique and sorted")
        return value


class ProtocolFreezeCandidate(FrozenRecord):
    schema_version: Literal["1.0.0"]
    candidate_id: str
    status: Literal["awaiting_protocol_freeze_approval"]
    title: str
    test_data_approval_id: str
    pre_freeze_git_commit: str
    models: dict[str, ModelRevision]
    counts: dict[str, int]
    files: tuple[ProtocolFreezeFile, ...] = Field(min_length=1)
    file_set_sha256: Sha256Hex
    test_data_scoring_performed: StrictBool
    protocol_freeze_approved: StrictBool
    main_run_approved: StrictBool
    local_commit_required_after_approval: StrictBool

    @field_validator(
        "candidate_id",
        "title",
        "test_data_approval_id",
        "pre_freeze_git_commit",
    )
    @classmethod
    def require_nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("freeze candidate text fields must be nonblank")
        return value

    @model_validator(mode="after")
    def validate_commitments(self) -> Self:
        paths = tuple(item.path for item in self.files)
        if tuple(sorted(paths)) != paths or len(set(paths)) != len(paths):
            raise ValueError("freeze candidate files must be unique and sorted")
        if set(self.models) != _REQUIRED_MODEL_ROLES:
            raise ValueError("freeze candidate must bind all research model roles")
        if tuple(self.counts) != tuple(sorted(self.counts)):
            raise ValueError("freeze candidate counts must be sorted")
        if any(value < 0 for value in self.counts.values()):
            raise ValueError("freeze candidate counts must be nonnegative")
        expected_file_set = canonical_sha256(
            [item.model_dump(mode="json") for item in self.files]
        )
        if self.file_set_sha256 != expected_file_set:
            raise ValueError("freeze candidate file-set hash mismatch")
        if self.candidate_id != _candidate_id(
            title=self.title,
            test_data_approval_id=self.test_data_approval_id,
            pre_freeze_git_commit=self.pre_freeze_git_commit,
            models=self.models,
            counts=self.counts,
            file_set_sha256=self.file_set_sha256,
        ):
            raise ValueError("freeze candidate ID mismatch")
        if (
            self.test_data_scoring_performed
            or self.protocol_freeze_approved
            or self.main_run_approved
            or not self.local_commit_required_after_approval
        ):
            raise ValueError("freeze candidate gate flags are invalid")
        return self


class ProtocolFreezeCandidateVerification(FrozenRecord):
    missing_files: tuple[str, ...] = ()
    changed_files: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.missing_files and not self.changed_files


def build_protocol_freeze_candidate(
    root: Path,
    file_groups: Mapping[str, Sequence[Path]],
    *,
    models: Mapping[str, ModelRevision],
    counts: Mapping[str, int],
    test_data_approval_id: str,
    pre_freeze_git_commit: str,
    title: str,
) -> ProtocolFreezeCandidate:
    categories_by_path: dict[str, set[str]] = {}
    for category, paths in file_groups.items():
        normalized_category = category.strip()
        if not normalized_category:
            raise ValueError("freeze candidate category must be nonblank")
        for path in paths:
            relative_path, _ = _read_candidate_file(root, path)
            categories_by_path.setdefault(relative_path, set()).add(
                normalized_category
            )
    if not categories_by_path:
        raise ValueError("freeze candidate requires at least one file")

    files: list[ProtocolFreezeFile] = []
    for relative_path in sorted(categories_by_path):
        _, contents = _read_candidate_file(root, Path(relative_path))
        files.append(
            ProtocolFreezeFile(
                path=relative_path,
                categories=tuple(sorted(categories_by_path[relative_path])),
                sha256=hashlib.sha256(contents).hexdigest(),
                size_bytes=len(contents),
            )
        )
    normalized_models = {
        role: ModelRevision.model_validate(
            identity.model_dump(mode="python", round_trip=True),
            strict=True,
        )
        for role, identity in sorted(models.items())
    }
    normalized_counts = {
        name: int(value) for name, value in sorted(counts.items())
    }
    frozen_files = tuple(files)
    file_set_sha256 = canonical_sha256(
        [item.model_dump(mode="json") for item in frozen_files]
    )
    return ProtocolFreezeCandidate(
        schema_version="1.0.0",
        candidate_id=_candidate_id(
            title=title,
            test_data_approval_id=test_data_approval_id,
            pre_freeze_git_commit=pre_freeze_git_commit,
            models=normalized_models,
            counts=normalized_counts,
            file_set_sha256=file_set_sha256,
        ),
        status="awaiting_protocol_freeze_approval",
        title=title,
        test_data_approval_id=test_data_approval_id,
        pre_freeze_git_commit=pre_freeze_git_commit,
        models=normalized_models,
        counts=normalized_counts,
        files=frozen_files,
        file_set_sha256=file_set_sha256,
        test_data_scoring_performed=False,
        protocol_freeze_approved=False,
        main_run_approved=False,
        local_commit_required_after_approval=True,
    )


def verify_protocol_freeze_candidate(
    root: Path,
    candidate: ProtocolFreezeCandidate,
) -> ProtocolFreezeCandidateVerification:
    missing: list[str] = []
    changed: list[str] = []
    for item in candidate.files:
        try:
            _, contents = _read_candidate_file(root, Path(item.path))
        except FileNotFoundError:
            missing.append(item.path)
            continue
        except ValueError:
            changed.append(item.path)
            continue
        if (
            hashlib.sha256(contents).hexdigest() != item.sha256
            or len(contents) != item.size_bytes
        ):
            changed.append(item.path)
    return ProtocolFreezeCandidateVerification(
        missing_files=tuple(sorted(missing)),
        changed_files=tuple(sorted(changed)),
    )


def _candidate_id(
    *,
    title: str,
    test_data_approval_id: str,
    pre_freeze_git_commit: str,
    models: Mapping[str, ModelRevision],
    counts: Mapping[str, int],
    file_set_sha256: str,
) -> str:
    digest = canonical_sha256(
        {
            "title": title,
            "test_data_approval_id": test_data_approval_id,
            "pre_freeze_git_commit": pre_freeze_git_commit,
            "models": {
                role: identity.model_dump(mode="json")
                for role, identity in sorted(models.items())
            },
            "counts": dict(sorted(counts.items())),
            "file_set_sha256": file_set_sha256,
        }
    )
    return f"PFC_{digest[:24]}"


def _read_candidate_file(root: Path, path: Path) -> tuple[str, bytes]:
    resolved_root = root.resolve()
    if not resolved_root.is_dir():
        raise ValueError("freeze candidate root must be an existing directory")
    requested = path if path.is_absolute() else resolved_root / path
    try:
        relative = requested.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"freeze candidate file is outside root: {path}") from error
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"freeze candidate path is unsafe: {path}")

    current = resolved_root
    for component in relative.parts:
        current = current / component
        if os.path.islink(current):
            raise ValueError(f"freeze candidate path contains a symlink: {path}")
    if not requested.exists():
        raise FileNotFoundError(f"freeze candidate file is missing: {path}")
    if not requested.is_file():
        raise ValueError(f"freeze candidate path is not a file: {path}")
    return relative.as_posix(), requested.read_bytes()
