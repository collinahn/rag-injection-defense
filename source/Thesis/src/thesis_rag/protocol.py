import hashlib
import json
import os
import stat
import subprocess
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import Field, field_validator

from thesis_rag.canonical import canonical_json, sha256_text
from thesis_rag.domain import FrozenRecord

_PROTOCOL_VERSION: Final[Literal["0.1.0"]] = "0.1.0"
_MAX_SYMLINK_DEPTH = 40
_REQUIRED_MODEL_ROLES: Final[frozenset[str]] = frozenset(
    {"generator", "embedding", "detector"}
)


class ModelRevision(FrozenRecord):
    id: str
    revision: str

    @field_validator("id", "revision")
    @classmethod
    def _require_nonblank_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model identity fields must be nonblank")
        return value


class ProtocolLock(FrozenRecord):
    protocol_version: Literal["0.1.0"]
    files: dict[str, str]
    models: dict[str, ModelRevision]
    created_from_git_commit: str

    @field_validator("models")
    @classmethod
    def _require_nonblank_model_names(
        cls, models: dict[str, ModelRevision]
    ) -> dict[str, ModelRevision]:
        if any(not name.strip() for name in models):
            raise ValueError("model names must be nonblank")
        missing_roles = _REQUIRED_MODEL_ROLES.difference(models)
        if missing_roles:
            raise ValueError(
                f"missing required model roles: {', '.join(sorted(missing_roles))}"
            )
        return models


@dataclass(frozen=True, slots=True)
class FrozenFileSnapshot:
    """Approved bytes captured from one securely opened frozen file."""

    relative_path: str
    sha256: str
    content: bytes

    def __post_init__(self) -> None:
        if hashlib.sha256(self.content).hexdigest() != self.sha256:
            raise ValueError("frozen snapshot hash does not match its content")


class ProtocolVerification(FrozenRecord):
    snapshot: tuple[FrozenFileSnapshot, ...] | None = None
    missing_files: tuple[str, ...] = Field(default=())
    changed_files: tuple[str, ...] = Field(default=())

    @property
    def valid(self) -> bool:
        return (
            self.snapshot is not None
            and not self.missing_files
            and not self.changed_files
        )


@dataclass(frozen=True, slots=True)
class AuthorizedProtocolSnapshot:
    """Immutable inputs for a main run executed only from the captured Git archive."""

    protocol_lock_bytes: bytes
    protocol_lock_sha256: str
    protocol_version: Literal["0.1.0"]
    created_from_git_commit: str
    execution_git_commit: str
    execution_archive_sha256: str
    execution_archive_bytes: bytes
    frozen_files: tuple[FrozenFileSnapshot, ...]

    def __post_init__(self) -> None:
        if self.execution_git_commit != self.created_from_git_commit:
            raise ValueError("execution archive commit does not match protocol lock")
        if (
            hashlib.sha256(self.execution_archive_bytes).hexdigest()
            != self.execution_archive_sha256
        ):
            raise ValueError("execution archive hash does not match its content")

    def bytes_for(self, relative_path: str) -> bytes:
        for frozen_file in self.frozen_files:
            if frozen_file.relative_path == relative_path:
                return frozen_file.content
        raise KeyError(f"file is not present in the authorized snapshot: {relative_path}")


class MainRunBlocked(RuntimeError):
    pass


class _UnsafeFrozenPathError(OSError):
    pass


def _lexical_root(root: Path) -> Path:
    return Path(os.path.abspath(root))


def _repository_root(root: Path) -> Path:
    resolved_root = root.resolve()
    if not resolved_root.is_dir():
        raise ValueError("repository root must be an existing directory")
    return resolved_root


def _traversal_components(parts: Sequence[str], *, require_file: bool) -> tuple[str, ...]:
    components = tuple(part for part in parts if part not in {"", "."})
    if any("\x00" in part for part in components):
        raise ValueError("path contains a NUL character")
    if require_file and not components:
        raise ValueError("path must name a file")
    return components


def _absolute_file_components(
    lexical_root: Path,
    resolved_root: Path,
    path: Path,
    *,
    require_file: bool,
) -> tuple[str, ...]:
    absolute_path = path
    for root in (lexical_root, resolved_root):
        try:
            return _traversal_components(
                absolute_path.relative_to(root).parts,
                require_file=require_file,
            )
        except ValueError:
            continue
    raise ValueError(f"frozen file is outside repository root: {path}")


def _frozen_file_components(
    lexical_root: Path, resolved_root: Path, path: Path
) -> tuple[str, ...]:
    if path.is_absolute():
        return _absolute_file_components(lexical_root, resolved_root, path, require_file=True)
    return _traversal_components(path.parts, require_file=True)


def _locked_file_components(relative_path: str) -> tuple[str, ...]:
    if (
        not relative_path
        or relative_path.startswith("/")
        or "\x00" in relative_path
    ):
        raise ValueError(f"locked file path is unsafe: {relative_path}")
    components = relative_path.split("/")
    if any(component in {"", ".", ".."} for component in components):
        raise ValueError(f"locked file path is unsafe: {relative_path}")
    return tuple(components)


def _symlink_target(
    directory_descriptor: int,
    component: str,
) -> str | None:
    try:
        metadata = os.stat(component, dir_fd=directory_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        raise
    except OSError as error:
        raise _UnsafeFrozenPathError("frozen file path is unsafe") from error
    if not stat.S_ISLNK(metadata.st_mode):
        return None
    try:
        return os.readlink(component, dir_fd=directory_descriptor)
    except FileNotFoundError:
        raise
    except OSError as error:
        raise _UnsafeFrozenPathError("frozen file path is unsafe") from error


def _symlink_components(
    lexical_root: Path,
    resolved_root: Path,
    target: str,
) -> tuple[tuple[str, ...], bool]:
    target_path = Path(target)
    if target_path.is_absolute():
        return (
            _absolute_file_components(
                lexical_root,
                resolved_root,
                target_path,
                require_file=False,
            ),
            True,
        )
    return _traversal_components(target_path.parts, require_file=False), False


def _close_descriptor(descriptor: int) -> None:
    with suppress(OSError):
        os.close(descriptor)


def _close_descriptors(descriptors: Sequence[int]) -> None:
    for descriptor in reversed(descriptors):
        _close_descriptor(descriptor)


def _open_root_anchored_file(
    root: Path,
    lexical_root: Path,
    components: tuple[str, ...],
) -> tuple[int, tuple[str, ...]]:
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    file_flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        root_descriptor = os.open(root, directory_flags)
    except OSError as error:
        raise ValueError("repository root cannot be securely opened") from error

    directory_descriptors: list[int] = []
    try:
        directory_descriptors.append(os.dup(root_descriptor))
        pending = list(components)
        resolved_components: list[str] = []
        symlink_depth = 0
        while pending:
            component = pending.pop(0)
            if component == "..":
                if len(directory_descriptors) == 1:
                    raise _UnsafeFrozenPathError("frozen file path is outside repository root")
                _close_descriptor(directory_descriptors.pop())
                resolved_components.pop()
                continue
            is_final_component = not pending
            flags = file_flags if is_final_component else directory_flags
            directory_descriptor = directory_descriptors[-1]
            try:
                opened_descriptor = os.open(component, flags, dir_fd=directory_descriptor)
            except FileNotFoundError:
                raise
            except OSError as error:
                target = _symlink_target(directory_descriptor, component)
                if target is None:
                    raise _UnsafeFrozenPathError("frozen file path is unsafe") from error
                symlink_depth += 1
                if symlink_depth > _MAX_SYMLINK_DEPTH:
                    raise _UnsafeFrozenPathError("frozen symlink depth is unsafe") from error
                try:
                    target_components, target_is_absolute = _symlink_components(
                        lexical_root,
                        root,
                        target,
                    )
                except ValueError as target_error:
                    raise _UnsafeFrozenPathError(
                        "frozen symlink target is outside repository root"
                    ) from target_error
                pending = [*target_components, *pending]
                if target_is_absolute:
                    replacement_descriptor = os.dup(root_descriptor)
                    _close_descriptors(directory_descriptors)
                    directory_descriptors = [replacement_descriptor]
                    resolved_components.clear()
                continue

            if is_final_component:
                try:
                    if not stat.S_ISREG(os.fstat(opened_descriptor).st_mode):
                        raise _UnsafeFrozenPathError("frozen path must be a regular file")
                except Exception:
                    _close_descriptor(opened_descriptor)
                    raise
                return opened_descriptor, tuple((*resolved_components, component))

            directory_descriptors.append(opened_descriptor)
            resolved_components.append(component)
    finally:
        _close_descriptors(directory_descriptors)
        _close_descriptor(root_descriptor)

    raise _UnsafeFrozenPathError("frozen file path is unsafe")


def _sha256_file_descriptor(file_descriptor: int) -> str:
    digest = hashlib.sha256()
    while chunk := os.read(file_descriptor, 1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def _read_file_descriptor(file_descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while chunk := os.read(file_descriptor, 1024 * 1024):
        chunks.append(chunk)
    return b"".join(chunks)


def _capture_root_anchored_file(
    root: Path,
    lexical_root: Path,
    components: tuple[str, ...],
    *,
    relative_path: str,
) -> FrozenFileSnapshot:
    file_descriptor, _ = _open_root_anchored_file(
        root,
        lexical_root,
        components,
    )
    try:
        content = _read_file_descriptor(file_descriptor)
    finally:
        os.close(file_descriptor)
    return FrozenFileSnapshot(
        relative_path=relative_path,
        sha256=hashlib.sha256(content).hexdigest(),
        content=content,
    )


def _normalized_models(
    models: Mapping[str, ModelRevision | Mapping[str, str]],
) -> dict[str, ModelRevision]:
    normalized: dict[str, ModelRevision] = {}
    for name, identity in models.items():
        value: ModelRevision | Mapping[str, str] = identity
        if isinstance(identity, ModelRevision):
            value = identity.model_dump(mode="python", round_trip=True, warnings=False)
        normalized[name] = ModelRevision.model_validate(value, strict=True)
    if any(not name.strip() for name in normalized):
        raise ValueError("model names must be nonblank")
    missing_roles = _REQUIRED_MODEL_ROLES.difference(normalized)
    if missing_roles:
        raise ValueError(
            f"missing required model roles: {', '.join(sorted(missing_roles))}"
        )
    return {name: normalized[name] for name in sorted(normalized)}


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_archive(root: Path, commit: str) -> bytes:
    result = subprocess.run(
        ["git", "archive", "--format=tar", commit],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return result.stdout


def create_protocol_lock(
    root: Path,
    frozen_files: Sequence[Path],
    *,
    models: Mapping[str, ModelRevision | Mapping[str, str]],
) -> ProtocolLock:
    lexical_root = _lexical_root(root)
    repository_root = _repository_root(root)
    normalized_models = _normalized_models(models)
    files: dict[str, str] = {}
    for file_path in frozen_files:
        components = _frozen_file_components(lexical_root, repository_root, file_path)
        try:
            file_descriptor, canonical_components = _open_root_anchored_file(
                repository_root,
                lexical_root,
                components,
            )
        except FileNotFoundError as error:
            raise ValueError(f"frozen file is missing: {file_path}") from error
        except _UnsafeFrozenPathError as error:
            detail = str(error)
            if detail == "frozen file path is unsafe":
                detail = "frozen file path is unsafe or outside repository root"
            raise ValueError(f"{detail}: {file_path}") from error
        canonical_path = "/".join(canonical_components)
        try:
            if canonical_path in files:
                raise ValueError(f"duplicate frozen file: {canonical_path}")
            files[canonical_path] = _sha256_file_descriptor(file_descriptor)
        finally:
            os.close(file_descriptor)

    return ProtocolLock(
        protocol_version=_PROTOCOL_VERSION,
        files={path: files[path] for path in sorted(files)},
        models=normalized_models,
        created_from_git_commit=_git_head(repository_root),
    )


def verify_protocol_lock(root: Path, lock: ProtocolLock) -> ProtocolVerification:
    lexical_root = _lexical_root(root)
    repository_root = _repository_root(root)
    missing_files: list[str] = []
    changed_files: list[str] = []
    captured_files: list[FrozenFileSnapshot] = []

    for relative_path in sorted(lock.files):
        try:
            components = _locked_file_components(relative_path)
        except ValueError:
            changed_files.append(relative_path)
            continue
        try:
            captured_file = _capture_root_anchored_file(
                repository_root,
                lexical_root,
                components,
                relative_path=relative_path,
            )
        except FileNotFoundError:
            missing_files.append(relative_path)
            continue
        except _UnsafeFrozenPathError:
            changed_files.append(relative_path)
            continue
        if captured_file.sha256 != lock.files[relative_path]:
            changed_files.append(relative_path)
            continue
        captured_files.append(captured_file)

    missing = tuple(sorted(missing_files))
    changed = tuple(sorted(changed_files))
    return ProtocolVerification(
        snapshot=tuple(captured_files) if not missing and not changed else None,
        missing_files=missing,
        changed_files=changed,
    )


def protocol_lock_sha256(lock: ProtocolLock) -> str:
    return sha256_text(canonical_json(lock))


def _model_mismatches(
    locked_models: Mapping[str, ModelRevision], current_models: Mapping[str, ModelRevision]
) -> tuple[str, ...]:
    mismatches: list[str] = []
    for name in sorted(set(locked_models) | set(current_models)):
        locked = locked_models.get(name)
        current = current_models.get(name)
        if locked == current:
            continue
        if locked is None:
            assert current is not None
            mismatches.append(f"{name}: unexpected current model {current.id}@{current.revision}")
        elif current is None:
            mismatches.append(f"{name}: missing current model")
        else:
            mismatches.append(
                f"{name}: locked {locked.id}@{locked.revision}, "
                f"current {current.id}@{current.revision}"
            )
    return tuple(mismatches)


def _reject_non_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON numeric constant: {value}")


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_protocol_lock(
    root: Path,
    lock_path: Path | None,
) -> tuple[bytes, ProtocolLock]:
    lexical_root = _lexical_root(root)
    repository_root = _repository_root(root)
    requested_path = lock_path or Path("artifacts/protocol-lock.json")
    components = _frozen_file_components(lexical_root, repository_root, requested_path)
    file_descriptor, _ = _open_root_anchored_file(
        repository_root,
        lexical_root,
        components,
    )
    try:
        serialized_bytes = _read_file_descriptor(file_descriptor)
    finally:
        os.close(file_descriptor)
    serialized = serialized_bytes.decode("utf-8")
    value = json.loads(
        serialized,
        object_pairs_hook=_strict_json_object,
        parse_constant=_reject_non_json_constant,
    )
    return serialized_bytes, ProtocolLock.model_validate(value, strict=True)


def _revalidate_protocol_lock(lock: ProtocolLock) -> ProtocolLock:
    serialized = lock.model_dump(mode="python", round_trip=True, warnings=False)
    return ProtocolLock.model_validate(serialized, strict=True)


def _approval_is_valid(approval_path: Path, lock: ProtocolLock) -> bool:
    try:
        value = json.loads(
            approval_path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_non_json_constant,
        )
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        return False
    return (
        isinstance(value, dict)
        and set(value) == {"lock_sha256", "approved_for_main_run"}
        and value["lock_sha256"] == protocol_lock_sha256(lock)
        and value["approved_for_main_run"] is True
    )


def _git_is_clean(root: Path) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return not result.stdout


def assert_main_run_allowed(
    root: Path,
    lock: ProtocolLock,
    approval_path: Path,
    *,
    current_models: Mapping[str, ModelRevision | Mapping[str, str]],
    lock_path: Path | None = None,
) -> AuthorizedProtocolSnapshot:
    try:
        lock = _revalidate_protocol_lock(lock)
    except Exception as error:
        raise MainRunBlocked("supplied protocol lock validation failed") from error

    try:
        protocol_lock_bytes, on_disk_lock = _load_protocol_lock(root, lock_path)
    except Exception as error:
        raise MainRunBlocked(
            "on-disk protocol lock is missing, malformed, or unsafe"
        ) from error
    if on_disk_lock != lock:
        raise MainRunBlocked("on-disk protocol lock does not match supplied lock")
    lock = on_disk_lock

    try:
        mismatches = _model_mismatches(lock.models, _normalized_models(current_models))
    except Exception as error:
        raise MainRunBlocked("current model validation failed") from error
    if mismatches:
        raise MainRunBlocked(f"model identity mismatch: {', '.join(mismatches)}")

    try:
        approval_is_valid = _approval_is_valid(approval_path, lock)
    except Exception as error:
        raise MainRunBlocked("approval verification failed") from error
    if not approval_is_valid:
        raise MainRunBlocked("main-run approval is missing, malformed, or invalid")

    try:
        current_git_head = _git_head(_repository_root(root))
    except Exception as error:
        raise MainRunBlocked("Git HEAD verification failed") from error
    if current_git_head != lock.created_from_git_commit:
        raise MainRunBlocked("Git HEAD does not match protocol lock")

    try:
        git_is_clean = _git_is_clean(_repository_root(root))
    except Exception as error:
        raise MainRunBlocked("Git clean-state verification failed") from error
    if not git_is_clean:
        raise MainRunBlocked("Git tree is not clean")

    try:
        execution_archive_bytes = _git_archive(
            _repository_root(root),
            lock.created_from_git_commit,
        )
    except Exception as error:
        raise MainRunBlocked("Git execution archive capture failed") from error

    try:
        verification = verify_protocol_lock(root, lock)
    except Exception as error:
        raise MainRunBlocked("protocol lock verification failed") from error
    if not verification.valid:
        raise MainRunBlocked(
            "protocol lock verification failed: "
            f"missing={', '.join(verification.missing_files)}; "
            f"changed={', '.join(verification.changed_files)}"
        )
    assert verification.snapshot is not None

    return AuthorizedProtocolSnapshot(
        protocol_lock_bytes=protocol_lock_bytes,
        protocol_lock_sha256=protocol_lock_sha256(lock),
        protocol_version=lock.protocol_version,
        created_from_git_commit=lock.created_from_git_commit,
        execution_git_commit=lock.created_from_git_commit,
        execution_archive_sha256=hashlib.sha256(execution_archive_bytes).hexdigest(),
        execution_archive_bytes=execution_archive_bytes,
        frozen_files=verification.snapshot,
    )
