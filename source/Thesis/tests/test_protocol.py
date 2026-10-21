import hashlib
import inspect
import io
import json
import multiprocessing
import os
import shutil
import subprocess
import tarfile
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

import thesis_rag.protocol as protocol
from thesis_rag.protocol import (
    MainRunBlocked,
    ModelRevision,
    ProtocolLock,
    ProtocolVerification,
    assert_main_run_allowed,
    create_protocol_lock,
    protocol_lock_sha256,
    verify_protocol_lock,
)


def _run_git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _git_repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    root.mkdir()
    _run_git(root, "init")
    _run_git(root, "config", "user.email", "tests@example.invalid")
    _run_git(root, "config", "user.name", "Protocol Tests")
    (root / ".gitignore").write_text(
        "artifacts/*.json\n",
        encoding="utf-8",
    )
    (root / "frozen.txt").write_bytes(b"frozen bytes\x00\n")
    (root / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    _run_git(root, "add", ".")
    _run_git(root, "commit", "-m", "initial fixture")
    return root


def _lock(root: Path) -> ProtocolLock:
    return create_protocol_lock(
        root,
        [root / "frozen.txt"],
        models=_required_models(),
    )


def _write_approval(root: Path, lock: ProtocolLock) -> Path:
    path = root / "artifacts" / "main-run-approval.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "lock_sha256": protocol_lock_sha256(lock),
                "approved_for_main_run": True,
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_protocol_lock(root: Path, lock: ProtocolLock) -> Path:
    path = root / "artifacts" / "protocol-lock.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(lock.model_dump(mode="json"), sort_keys=True),
        encoding="utf-8",
    )
    return path


def _current_models() -> dict[str, dict[str, str]]:
    return _required_models()


def _required_models() -> dict[str, dict[str, str]]:
    return {
        "generator": {"id": "mock-generator", "revision": "v1"},
        "embedding": {"id": "mock-embedding", "revision": "v1"},
        "detector": {"id": "mock-detector", "revision": "v1"},
    }


def _required_model_revisions() -> dict[str, ModelRevision]:
    return {
        name: ModelRevision.model_validate(identity, strict=True)
        for name, identity in _required_models().items()
    }


def _create_fifo_lock_in_subprocess(root: str, fifo: str, result_queue: object) -> None:
    try:
        create_protocol_lock(Path(root), [Path(fifo)], models=_required_models())
    except ValueError:
        result_queue.put("value-error")  # type: ignore[attr-defined]
    else:
        result_queue.put("accepted")  # type: ignore[attr-defined]


def _verify_fifo_lock_in_subprocess(root: str, fifo: str, result_queue: object) -> None:
    lock = ProtocolLock(
        protocol_version="0.1.0",
        files={Path(fifo).name: "a" * 64},
        models=_required_model_revisions(),
        created_from_git_commit="b" * 40,
    )
    result_queue.put(  # type: ignore[attr-defined]
        verify_protocol_lock(Path(root), lock).changed_files
    )


def _run_fifo_subprocess(
    target: Callable[[str, str, object], None], root: Path, fifo: Path
) -> object:
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(target=target, args=(str(root), str(fifo), result_queue))
    process.start()
    process.join(timeout=3)
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("FIFO handling blocked")
    assert process.exitcode == 0
    return result_queue.get(timeout=1)


def _ready_main_run(root: Path) -> tuple[ProtocolLock, Path]:
    lock = _lock(root)
    _write_protocol_lock(root, lock)
    approval = _write_approval(root, lock)
    return lock, approval


def _swap_before_final_open(
    monkeypatch: pytest.MonkeyPatch, locked: Path, outside: Path
) -> Callable[[], bool]:
    original_open = os.open
    outside_stat = outside.stat()
    original_read = os.read
    swapped = False

    def swap_before_open(
        path: str | bytes | Path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if path == locked.name and dir_fd is not None and not swapped:
            locked.unlink()
            locked.symlink_to(outside)
            swapped = True
        return original_open(path, flags, mode, dir_fd=dir_fd)

    def reject_outside_read(file_descriptor: int, size: int) -> bytes:
        opened_stat = os.fstat(file_descriptor)
        if (opened_stat.st_dev, opened_stat.st_ino) == (outside_stat.st_dev, outside_stat.st_ino):
            raise AssertionError("attempted to read an outside target")
        return original_read(file_descriptor, size)

    monkeypatch.setattr(protocol.os, "open", swap_before_open)
    monkeypatch.setattr(protocol.os, "read", reject_outside_read)
    return lambda: swapped


def test_protocol_records_are_frozen_and_forbid_extra_fields() -> None:
    model = ModelRevision(id="mock-generator", revision="v1")
    lock = ProtocolLock(
        protocol_version="0.1.0",
        files={"frozen.txt": "a" * 64},
        models=_required_model_revisions() | {"generator": model},
        created_from_git_commit="b" * 40,
    )
    report = ProtocolVerification(snapshot=(), missing_files=(), changed_files=())

    with pytest.raises(ValidationError, match="frozen"):
        model.id = "different"  # type: ignore[misc]
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ProtocolLock.model_validate(lock.model_dump() | {"extra": True})
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ProtocolVerification.model_validate(report.model_dump() | {"extra": True})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", ""),
        ("id", " \t"),
        ("revision", ""),
        ("revision", "\n"),
    ],
)
def test_model_revision_rejects_blank_identity_fields(field: str, value: str) -> None:
    identity = {"id": "model", "revision": "v1"}
    identity[field] = value

    with pytest.raises(ValidationError, match=field):
        ModelRevision.model_validate(identity)


def test_protocol_lock_rejects_unsupported_versions_and_blank_model_names() -> None:
    valid_lock = {
        "protocol_version": "0.1.0",
        "files": {},
        "models": _required_models(),
        "created_from_git_commit": "b" * 40,
    }

    with pytest.raises(ValidationError, match="protocol_version"):
        ProtocolLock.model_validate(valid_lock | {"protocol_version": "9.9.9"})
    with pytest.raises(ValidationError, match="models"):
        ProtocolLock.model_validate(
            valid_lock
            | {
                "models": _required_models()
                | {" \t": {"id": "model", "revision": "v1"}}
            }
        )


@pytest.mark.parametrize(
    "models",
    [
        {},
        {"generator": {"id": "generator", "revision": "v1"}},
        {
            "generator": {"id": "generator", "revision": "v1"},
            "embedding": {"id": "embedding", "revision": "v1"},
        },
    ],
    ids=["empty", "generator-only", "missing-detector"],
)
def test_create_protocol_lock_requires_all_research_model_roles(
    tmp_path: Path,
    models: dict[str, dict[str, str]],
) -> None:
    root = _git_repository(tmp_path)

    with pytest.raises(ValueError, match="required model roles"):
        create_protocol_lock(root, [root / "frozen.txt"], models=models)


def test_create_protocol_lock_hashes_raw_bytes_and_orders_posix_paths(tmp_path: Path) -> None:
    root = _git_repository(tmp_path)
    nested = root / "zeta" / "input.bin"
    nested.parent.mkdir()
    nested.write_bytes(b"\x00\xff\n")
    alpha = root / "alpha.txt"
    alpha.write_bytes(b"alpha\r\n")

    lock = create_protocol_lock(
        root,
        [nested, alpha, root / "frozen.txt"],
        models={
            "detector": {"revision": "v2", "id": "detector"},
            "embedding": {"revision": "v3", "id": "embedding"},
            "generator": ModelRevision(id="generator", revision="v1"),
        },
    )

    assert tuple(lock.files) == ("alpha.txt", "frozen.txt", "zeta/input.bin")
    assert lock.files["zeta/input.bin"] == hashlib.sha256(b"\x00\xff\n").hexdigest()
    assert tuple(lock.models) == ("detector", "embedding", "generator")
    assert lock.models["generator"] == ModelRevision(id="generator", revision="v1")
    assert lock.protocol_version == "0.1.0"
    assert len(lock.created_from_git_commit) == 40


def test_create_protocol_lock_interprets_relative_frozen_files_from_the_repository_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _git_repository(tmp_path)
    monkeypatch.chdir(tmp_path)

    lock = create_protocol_lock(root, [Path("frozen.txt")], models=_required_models())

    assert tuple(lock.files) == ("frozen.txt",)


@pytest.mark.parametrize(
    "paths",
    [
        lambda root: [root / "frozen.txt", root / "directory" / ".." / "frozen.txt"],
        lambda root: [root / "frozen.txt", root / "frozen.txt"],
    ],
)
def test_create_protocol_lock_rejects_duplicate_resolved_files(
    tmp_path: Path, paths: Callable[[Path], list[Path]]
) -> None:
    root = _git_repository(tmp_path)
    (root / "directory").mkdir()

    with pytest.raises(ValueError, match="duplicate"):
        create_protocol_lock(root, paths(root), models=_required_models())


def test_create_protocol_lock_rejects_missing_and_non_file_paths(tmp_path: Path) -> None:
    root = _git_repository(tmp_path)
    directory = root / "directory"
    directory.mkdir()

    with pytest.raises(ValueError, match="missing"):
        create_protocol_lock(root, [root / "missing.txt"], models=_required_models())
    with pytest.raises(ValueError, match="regular file"):
        create_protocol_lock(root, [directory], models=_required_models())


def test_create_and_verify_protocol_lock_reject_fifos_without_blocking(tmp_path: Path) -> None:
    root = _git_repository(tmp_path)
    fifo = root / "frozen.pipe"
    os.mkfifo(fifo)

    assert _run_fifo_subprocess(_create_fifo_lock_in_subprocess, root, fifo) == "value-error"
    assert _run_fifo_subprocess(_verify_fifo_lock_in_subprocess, root, fifo) == ("frozen.pipe",)


def test_create_protocol_lock_rejects_outside_paths_and_escaping_symlinks(tmp_path: Path) -> None:
    root = _git_repository(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    escape = root / "escape.txt"
    escape.symlink_to(outside)

    with pytest.raises(ValueError, match="outside"):
        create_protocol_lock(root, [outside], models=_required_models())
    with pytest.raises(ValueError, match="outside"):
        create_protocol_lock(root, [escape], models=_required_models())


def test_create_protocol_lock_resolves_final_and_intermediate_symlinks_within_root(
    tmp_path: Path,
) -> None:
    root = _git_repository(tmp_path)
    real = root / "real" / "frozen.txt"
    real.parent.mkdir()
    real.write_bytes(b"real bytes")
    final_alias = root / "final-alias.txt"
    final_alias.symlink_to("real/frozen.txt")
    directory_alias = root / "directory-alias"
    directory_alias.symlink_to("real", target_is_directory=True)

    final_lock = create_protocol_lock(root, [final_alias], models=_required_models())
    directory_lock = create_protocol_lock(
        root,
        [directory_alias / "frozen.txt"],
        models=_required_models(),
    )

    assert tuple(final_lock.files) == ("real/frozen.txt",)
    assert final_lock.files == directory_lock.files


def test_create_protocol_lock_rejects_a_real_path_and_in_root_symlink_alias_as_duplicates(
    tmp_path: Path,
) -> None:
    root = _git_repository(tmp_path)
    real = root / "real" / "frozen.txt"
    real.parent.mkdir()
    real.write_bytes(b"real bytes")
    alias = root / "alias.txt"
    alias.symlink_to("real/frozen.txt")

    with pytest.raises(ValueError, match="duplicate"):
        create_protocol_lock(root, [real, alias], models=_required_models())


@pytest.mark.parametrize("absolute_target", [False, True])
@pytest.mark.parametrize("frozen_path_is_absolute", [False, True])
def test_create_protocol_lock_applies_parent_components_after_symlink_resolution(
    tmp_path: Path, absolute_target: bool, frozen_path_is_absolute: bool
) -> None:
    root = _git_repository(tmp_path)
    target = root / "real" / "target.txt"
    target.parent.mkdir()
    target.write_bytes(b"canonical target")
    alias = root / "alias"
    alias_target = root / "real" / "subdirectory" if absolute_target else "real/subdirectory"
    (root / "real" / "subdirectory").mkdir()
    alias.symlink_to(alias_target, target_is_directory=True)
    requested = Path("alias/../target.txt")
    if frozen_path_is_absolute:
        requested = root / requested

    lock = create_protocol_lock(root, [requested], models=_required_models())

    assert lock.files == {"real/target.txt": hashlib.sha256(b"canonical target").hexdigest()}
    with pytest.raises(ValueError, match="duplicate"):
        create_protocol_lock(root, [target, requested], models=_required_models())


def test_create_protocol_lock_rejects_parent_after_an_escaping_symlink(tmp_path: Path) -> None:
    root = _git_repository(tmp_path)
    outside_directory = tmp_path / "outside" / "subdirectory"
    outside_directory.mkdir(parents=True)
    (outside_directory.parent / "secret.txt").write_bytes(b"outside")
    alias = root / "escape"
    alias.symlink_to(outside_directory, target_is_directory=True)

    with pytest.raises(ValueError, match="unsafe|outside"):
        create_protocol_lock(
            root,
            [root / "escape" / ".." / "secret.txt"],
            models=_required_models(),
        )


def test_create_protocol_lock_rejects_escaping_cyclic_and_overdeep_symlinks(tmp_path: Path) -> None:
    root = _git_repository(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    (root / "escape.txt").symlink_to(outside)
    (root / "cycle-a").symlink_to("cycle-b")
    (root / "cycle-b").symlink_to("cycle-a")
    for index in range(41):
        target = f"depth-{index + 1}" if index < 40 else "frozen.txt"
        (root / f"depth-{index}").symlink_to(target)

    for path in (root / "escape.txt", root / "cycle-a", root / "depth-0"):
        with pytest.raises(ValueError, match="unsafe|outside"):
            create_protocol_lock(root, [path], models=_required_models())


def test_create_protocol_lock_accepts_an_absolute_child_through_a_lexical_root_alias(
    tmp_path: Path,
) -> None:
    root = _git_repository(tmp_path)
    alias_parent = tmp_path / "lexical-alias"
    alias_parent.symlink_to(tmp_path, target_is_directory=True)
    lexical_root = alias_parent / root.name

    lock = create_protocol_lock(
        lexical_root,
        [lexical_root / "frozen.txt"],
        models=_required_models(),
    )

    assert tuple(lock.files) == ("frozen.txt",)


def test_create_protocol_lock_never_hashes_an_outside_target_swapped_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _git_repository(tmp_path)
    locked = root / "locked.txt"
    locked.write_bytes(b"inside")
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside")
    original_read_bytes = Path.read_bytes

    def swap_before_read(path: Path) -> bytes:
        if path == locked:
            locked.unlink()
            locked.symlink_to(outside)
            data = original_read_bytes(path)
            if data == b"outside":
                raise AssertionError("attempted to read an outside target")
            return data
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", swap_before_read)

    lock = create_protocol_lock(root, [locked], models=_required_models())

    assert lock.files == {"locked.txt": hashlib.sha256(b"inside").hexdigest()}


def test_create_protocol_lock_rejects_an_escape_swapped_immediately_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _git_repository(tmp_path)
    locked = root / "locked.txt"
    locked.write_bytes(b"inside")
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside")
    swapped = _swap_before_final_open(monkeypatch, locked, outside)

    with pytest.raises(ValueError, match="unsafe|outside"):
        create_protocol_lock(root, [locked], models=_required_models())

    assert swapped()


def test_create_protocol_lock_closes_the_root_descriptor_when_initial_duplication_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _git_repository(tmp_path)
    original_open = os.open
    original_fstat = os.fstat
    opened_descriptors: list[int] = []

    def track_open(
        path: str | bytes | Path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        opened_descriptors.append(descriptor)
        return descriptor

    def fail_first_duplication(_: int) -> int:
        raise OSError("injected initial duplication failure")

    monkeypatch.setattr(protocol.os, "open", track_open)
    monkeypatch.setattr(protocol.os, "dup", fail_first_duplication)

    with pytest.raises(OSError, match="injected initial duplication failure"):
        create_protocol_lock(
            root,
            [root / "frozen.txt"],
            models=_required_models(),
        )

    for descriptor in opened_descriptors:
        with pytest.raises(OSError):
            original_fstat(descriptor)


def test_create_protocol_lock_preserves_a_restart_duplication_failure_and_closes_descriptors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _git_repository(tmp_path)
    alias = root / "absolute-alias.txt"
    alias.symlink_to(root / "frozen.txt")
    original_open = os.open
    original_dup = os.dup
    original_fstat = os.fstat
    opened_descriptors: list[int] = []
    duplication_count = 0

    def track_open(
        path: str | bytes | Path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        opened_descriptors.append(descriptor)
        return descriptor

    def track_duplication(descriptor: int) -> int:
        nonlocal duplication_count
        duplication_count += 1
        if duplication_count == 2:
            raise OSError("injected restart duplication failure")
        duplicate = original_dup(descriptor)
        opened_descriptors.append(duplicate)
        return duplicate

    monkeypatch.setattr(protocol.os, "open", track_open)
    monkeypatch.setattr(protocol.os, "dup", track_duplication)

    with pytest.raises(OSError, match="injected restart duplication failure"):
        create_protocol_lock(root, [alias], models=_required_models())

    assert duplication_count == 2
    for descriptor in opened_descriptors:
        with pytest.raises(OSError):
            original_fstat(descriptor)


def test_protocol_verification_reports_lexically_sorted_missing_and_changed_files(
    tmp_path: Path,
) -> None:
    root = _git_repository(tmp_path)
    alpha = root / "alpha.txt"
    zeta = root / "zeta.txt"
    alpha.write_text("alpha\n", encoding="utf-8")
    zeta.write_text("zeta\n", encoding="utf-8")
    lock = create_protocol_lock(
        root,
        [zeta, alpha, root / "frozen.txt"],
        models=_required_models(),
    )
    alpha.write_text("changed\n", encoding="utf-8")
    zeta.unlink()

    report = verify_protocol_lock(root, lock)

    assert report == ProtocolVerification(
        missing_files=("zeta.txt",),
        changed_files=("alpha.txt",),
    )


@pytest.mark.parametrize("escaped_path", ["../outside.txt", "/outside.txt"])
def test_protocol_verification_reports_invalid_locked_paths_as_changed(
    tmp_path: Path, escaped_path: str
) -> None:
    root = _git_repository(tmp_path)
    lock = ProtocolLock.model_construct(
        protocol_version="0.1.0",
        files={escaped_path: "a" * 64},
        models=_required_model_revisions(),
        created_from_git_commit="b" * 40,
    )

    report = verify_protocol_lock(root, lock)

    assert report == ProtocolVerification(
        missing_files=(),
        changed_files=(escaped_path,),
    )


@pytest.mark.parametrize(
    "locked_path",
    ["", "./frozen.txt", "alias/../frozen.txt", "frozen//txt", "frozen.txt\x00suffix"],
)
def test_protocol_verification_reports_noncanonical_locked_paths_as_changed(
    tmp_path: Path, locked_path: str
) -> None:
    root = _git_repository(tmp_path)
    lock = ProtocolLock.model_construct(
        protocol_version="0.1.0",
        files={locked_path: hashlib.sha256(b"frozen bytes\x00\n").hexdigest()},
        models=_required_model_revisions(),
        created_from_git_commit="b" * 40,
    )

    report = verify_protocol_lock(root, lock)

    assert report == ProtocolVerification(
        missing_files=(),
        changed_files=(locked_path,),
    )


def test_protocol_verification_resolves_a_legacy_in_root_symlink_path_and_reports_later_escape(
    tmp_path: Path,
) -> None:
    root = _git_repository(tmp_path)
    real = root / "real" / "frozen.txt"
    real.parent.mkdir()
    real.write_bytes(b"real bytes")
    alias = root / "alias.txt"
    alias.symlink_to("real/frozen.txt")
    lock = ProtocolLock(
        protocol_version="0.1.0",
        files={"alias.txt": hashlib.sha256(b"real bytes").hexdigest()},
        models=_required_model_revisions(),
        created_from_git_commit="b" * 40,
    )

    assert verify_protocol_lock(root, lock).valid is True

    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside")
    alias.unlink()
    alias.symlink_to(outside)

    assert verify_protocol_lock(root, lock).changed_files == ("alias.txt",)


def test_protocol_verification_treats_a_later_path_escape_as_changed_without_reading_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _git_repository(tmp_path)
    locked = root / "locked.txt"
    locked.write_text("inside\n", encoding="utf-8")
    lock = create_protocol_lock(root, [locked], models=_required_models())
    outside = tmp_path / "outside.txt"
    outside.write_text("do not read\n", encoding="utf-8")
    locked.unlink()
    locked.symlink_to(outside)
    original_read_bytes = Path.read_bytes

    def prohibit_outside_read(path: Path) -> bytes:
        if path == outside:
            raise AssertionError("verification read an outside target")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", prohibit_outside_read)

    report = verify_protocol_lock(root, lock)

    assert report == ProtocolVerification(
        missing_files=(),
        changed_files=("locked.txt",),
    )


def test_protocol_verification_never_reads_an_outside_target_swapped_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _git_repository(tmp_path)
    locked = root / "locked.txt"
    locked.write_bytes(b"inside")
    lock = create_protocol_lock(root, [locked], models=_required_models())
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside")
    original_read_bytes = Path.read_bytes

    def swap_before_read(path: Path) -> bytes:
        if path == locked:
            locked.unlink()
            locked.symlink_to(outside)
            data = original_read_bytes(path)
            if data == b"outside":
                raise AssertionError("attempted to read an outside target")
            return data
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", swap_before_read)

    assert verify_protocol_lock(root, lock).valid is True


def test_protocol_verification_reports_an_escape_swapped_immediately_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _git_repository(tmp_path)
    locked = root / "locked.txt"
    locked.write_bytes(b"inside")
    lock = create_protocol_lock(root, [locked], models=_required_models())
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside")
    swapped = _swap_before_final_open(monkeypatch, locked, outside)

    report = verify_protocol_lock(root, lock)

    assert swapped()
    assert report == ProtocolVerification(
        missing_files=(),
        changed_files=("locked.txt",),
    )


@pytest.mark.parametrize("replacement_bytes", [b"unapproved", b"approved"])
def test_protocol_verification_captures_approved_bytes_without_reopening_swapped_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_bytes: bytes,
) -> None:
    root = _git_repository(tmp_path)
    locked = root / "locked.txt"
    locked.write_bytes(b"approved")
    lock = create_protocol_lock(root, [locked], models=_required_models())
    original_open = protocol._open_root_anchored_file
    original_fstat = os.fstat
    opened_descriptors: list[int] = []
    swapped = False

    def open_then_swap(
        root_path: Path,
        lexical_root: Path,
        components: tuple[str, ...],
    ) -> tuple[int, tuple[str, ...]]:
        nonlocal swapped
        descriptor, canonical_components = original_open(
            root_path,
            lexical_root,
            components,
        )
        opened_descriptors.append(descriptor)
        if not swapped:
            locked.rename(root / "opened-approved.txt")
            locked.write_bytes(replacement_bytes)
            swapped = True
        return descriptor, canonical_components

    monkeypatch.setattr(protocol, "_open_root_anchored_file", open_then_swap)

    report = verify_protocol_lock(root, lock)

    assert swapped is True
    assert report.valid is True
    assert report.missing_files == ()
    assert report.changed_files == ()
    assert report.snapshot is not None
    assert len(report.snapshot) == 1
    assert report.snapshot[0].relative_path == "locked.txt"
    assert report.snapshot[0].content == b"approved"
    assert report.snapshot[0].sha256 == hashlib.sha256(b"approved").hexdigest()
    assert locked.read_bytes() == replacement_bytes
    assert len(opened_descriptors) == 1
    for descriptor in opened_descriptors:
        with pytest.raises(OSError):
            original_fstat(descriptor)


def test_protocol_lock_sha256_is_stable_for_equivalent_records() -> None:
    first = ProtocolLock(
        protocol_version="0.1.0",
        files={"b.txt": "b" * 64, "a.txt": "a" * 64},
        models={
            "detector": ModelRevision(id="d-model", revision="r3"),
            "generator": ModelRevision(id="g-model", revision="r1"),
            "embedding": ModelRevision(id="e-model", revision="r2"),
        },
        created_from_git_commit="c" * 40,
    )
    second = ProtocolLock(
        protocol_version="0.1.0",
        files={"a.txt": "a" * 64, "b.txt": "b" * 64},
        models={
            "embedding": ModelRevision(id="e-model", revision="r2"),
            "generator": ModelRevision(id="g-model", revision="r1"),
            "detector": ModelRevision(id="d-model", revision="r3"),
        },
        created_from_git_commit="c" * 40,
    )

    assert protocol_lock_sha256(first) == protocol_lock_sha256(second)
    assert protocol_lock_sha256(first) == hashlib.sha256(
        b'{"created_from_git_commit":"cccccccccccccccccccccccccccccccccccccccc",'
        b'"files":{"a.txt":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
        b'"b.txt":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},'
        b'"models":{"detector":{"id":"d-model","revision":"r3"},'
        b'"embedding":{"id":"e-model","revision":"r2"},'
        b'"generator":{"id":"g-model","revision":"r1"}},"protocol_version":"0.1.0"}'
    ).hexdigest()


def test_main_run_requires_an_exact_current_model_identity(tmp_path: Path) -> None:
    root = _git_repository(tmp_path)
    lock = _lock(root)
    _write_protocol_lock(root, lock)
    approval = _write_approval(root, lock)
    wrong_generator = _required_models()
    wrong_generator["generator"] = {"id": "mock-generator", "revision": "v2"}

    with pytest.raises(MainRunBlocked, match="model identity"):
        assert_main_run_allowed(
            root,
            lock,
            approval,
            current_models=wrong_generator,
        )
    with pytest.raises(MainRunBlocked) as error:
        assert_main_run_allowed(
            root,
            lock,
            approval,
            current_models=_required_models()
            | {
                "detector": {"id": "different-detector", "revision": "v1"},
                "z-detector": {"id": "detector", "revision": "v2"},
                "a-generator": {"id": "generator", "revision": "v1"},
            },
        )
    message = str(error.value)
    assert message.index("a-generator:") < message.index("detector:")
    assert message.index("detector:") < message.index("z-detector:")


@pytest.mark.parametrize(
    "current_models",
    [
        {},
        {
            "generator": {"id": "mock-generator", "revision": "v1"},
            "embedding": {"id": "mock-embedding", "revision": "v1"},
        },
    ],
    ids=["empty", "missing-detector"],
)
def test_main_run_rejects_current_models_missing_required_research_roles(
    tmp_path: Path,
    current_models: dict[str, dict[str, str]],
) -> None:
    root = _git_repository(tmp_path)
    lock = create_protocol_lock(
        root,
        [root / "frozen.txt"],
        models=_required_models(),
    )
    _write_protocol_lock(root, lock)
    approval = _write_approval(root, lock)

    with pytest.raises(MainRunBlocked, match="current model validation") as error:
        assert_main_run_allowed(
            root,
            lock,
            approval,
            current_models=current_models,
        )

    assert isinstance(error.value.__cause__, ValueError)


@pytest.mark.parametrize(
    "models",
    [
        {},
        {
            "generator": ModelRevision(id="mock-generator", revision="v1"),
            "embedding": ModelRevision(id="mock-embedding", revision="v1"),
        },
    ],
    ids=["empty", "missing-detector"],
)
def test_main_run_revalidates_required_roles_in_tampered_protocol_locks(
    tmp_path: Path,
    models: dict[str, ModelRevision],
) -> None:
    root = _git_repository(tmp_path)
    valid_lock = create_protocol_lock(
        root,
        [root / "frozen.txt"],
        models=_required_models(),
    )
    tampered_lock = valid_lock.model_copy(update={"models": models})
    _write_protocol_lock(root, tampered_lock)
    approval = _write_approval(root, tampered_lock)

    with pytest.raises(MainRunBlocked, match="supplied protocol lock validation") as error:
        assert_main_run_allowed(
            root,
            tampered_lock,
            approval,
            current_models=models,
        )

    assert isinstance(error.value.__cause__, ValidationError)


@pytest.mark.parametrize(
    "approval_contents",
    [
        None,
        "not-json",
        "[]",
        '{"lock_sha256":"wrong","approved_for_main_run":true}',
        '{"lock_sha256":"{digest}","approved_for_main_run":false}',
        '{"lock_sha256":"{digest}","approved_for_main_run":true,"extra":true}',
    ],
    ids=["missing", "malformed", "array", "mismatched", "false", "extra-field"],
)
def test_main_run_rejects_missing_or_nonexact_approval(
    tmp_path: Path, approval_contents: str | None
) -> None:
    root = _git_repository(tmp_path)
    lock = _lock(root)
    _write_protocol_lock(root, lock)
    approval = root / "artifacts" / "main-run-approval.json"
    approval.parent.mkdir(parents=True, exist_ok=True)
    if approval_contents is not None:
        approval.write_text(
            approval_contents.replace("{digest}", protocol_lock_sha256(lock)), encoding="utf-8"
        )

    with pytest.raises(MainRunBlocked, match="approval"):
        assert_main_run_allowed(root, lock, approval, current_models=_current_models())


def test_main_run_requires_the_conventional_on_disk_protocol_lock(tmp_path: Path) -> None:
    root = _git_repository(tmp_path)
    lock = _lock(root)
    approval = _write_approval(root, lock)

    with pytest.raises(MainRunBlocked, match="on-disk protocol lock"):
        assert_main_run_allowed(root, lock, approval, current_models=_current_models())


def test_main_run_requires_the_on_disk_lock_to_match_the_supplied_lock(tmp_path: Path) -> None:
    root = _git_repository(tmp_path)
    lock = _lock(root)
    mismatched_lock = lock.model_copy(
        update={
            "models": lock.models
            | {
                "generator": ModelRevision(id="different-generator", revision="v1")
            }
        }
    )
    _write_protocol_lock(root, mismatched_lock)
    approval = _write_approval(root, lock)

    with pytest.raises(MainRunBlocked, match="does not match supplied lock"):
        assert_main_run_allowed(root, lock, approval, current_models=_current_models())


def test_main_run_returns_approved_frozen_bytes_when_path_swaps_after_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _git_repository(tmp_path)
    lock, approval = _ready_main_run(root)
    frozen_path = root / "frozen.txt"
    original_open = protocol._open_root_anchored_file
    frozen_open_count = 0

    def open_then_swap(
        root_path: Path,
        lexical_root: Path,
        components: tuple[str, ...],
    ) -> tuple[int, tuple[str, ...]]:
        nonlocal frozen_open_count
        descriptor, canonical_components = original_open(
            root_path,
            lexical_root,
            components,
        )
        if components == ("frozen.txt",):
            frozen_open_count += 1
            if frozen_open_count == 1:
                frozen_path.rename(root / "opened-frozen.txt")
                frozen_path.write_bytes(b"unapproved replacement")
        return descriptor, canonical_components

    monkeypatch.setattr(protocol, "_open_root_anchored_file", open_then_swap)

    snapshot = assert_main_run_allowed(
        root,
        lock,
        approval,
        current_models=_current_models(),
    )

    assert snapshot.bytes_for("frozen.txt") == b"frozen bytes\x00\n"
    assert frozen_path.read_bytes() == b"unapproved replacement"
    assert frozen_open_count == 1


def test_main_run_carries_exact_lock_bytes_when_lock_path_swaps_after_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _git_repository(tmp_path)
    lock, approval = _ready_main_run(root)
    lock_path = root / "artifacts" / "protocol-lock.json"
    approved_lock_bytes = lock_path.read_bytes()
    original_open = protocol._open_root_anchored_file
    original_fstat = os.fstat
    lock_descriptors: list[int] = []
    lock_open_count = 0

    def open_then_swap(
        root_path: Path,
        lexical_root: Path,
        components: tuple[str, ...],
    ) -> tuple[int, tuple[str, ...]]:
        nonlocal lock_open_count
        descriptor, canonical_components = original_open(
            root_path,
            lexical_root,
            components,
        )
        if components == ("artifacts", "protocol-lock.json"):
            lock_descriptors.append(descriptor)
            lock_open_count += 1
            if lock_open_count == 1:
                lock_path.rename(lock_path.with_name("opened-protocol-lock.json"))
                lock_path.write_text(
                    '{"protocol_version":"unsupported"}',
                    encoding="utf-8",
                )
        return descriptor, canonical_components

    monkeypatch.setattr(protocol, "_open_root_anchored_file", open_then_swap)

    snapshot = assert_main_run_allowed(
        root,
        lock,
        approval,
        current_models=_current_models(),
    )

    assert snapshot.protocol_lock_bytes == approved_lock_bytes
    assert snapshot.bytes_for("frozen.txt") == b"frozen bytes\x00\n"
    assert lock_path.read_text(encoding="utf-8") == '{"protocol_version":"unsupported"}'
    assert lock_open_count == 1
    for descriptor in lock_descriptors:
        with pytest.raises(OSError):
            original_fstat(descriptor)


def test_main_run_strictly_loads_an_explicit_protocol_lock_path(tmp_path: Path) -> None:
    root = _git_repository(tmp_path)
    lock = _lock(root)
    default_lock_path = _write_protocol_lock(root, lock)
    explicit_lock_path = default_lock_path.with_name("frozen-main-lock.json")
    default_lock_path.rename(explicit_lock_path)
    approval = _write_approval(root, lock)

    assert_main_run_allowed(
        root,
        lock,
        approval,
        current_models=_current_models(),
        lock_path=explicit_lock_path,
    )

    serialized_lock = json.dumps(lock.model_dump(mode="json")).removesuffix("}")
    explicit_lock_path.write_text(
        f'{serialized_lock},"protocol_version":"0.1.0"}}',
        encoding="utf-8",
    )

    with pytest.raises(MainRunBlocked, match="on-disk protocol lock") as error:
        assert_main_run_allowed(
            root,
            lock,
            approval,
            current_models=_current_models(),
            lock_path=explicit_lock_path,
        )

    assert isinstance(error.value.__cause__, ValueError)


def test_main_run_rejects_a_clean_commit_created_after_protocol_approval(
    tmp_path: Path,
) -> None:
    root = _git_repository(tmp_path)
    lock, approval = _ready_main_run(root)
    (root / "code.py").write_text("SAFE = False\n", encoding="utf-8")
    _run_git(root, "add", "code.py")
    _run_git(root, "commit", "-m", "change execution code after approval")

    with pytest.raises(MainRunBlocked, match="Git HEAD"):
        assert_main_run_allowed(root, lock, approval, current_models=_current_models())


def test_main_run_pins_approved_code_when_head_changes_after_clean_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _git_repository(tmp_path)
    execution_file = root / "code.py"
    execution_file.write_text("SAFE = True\n", encoding="utf-8")
    _run_git(root, "add", "code.py")
    _run_git(root, "commit", "-m", "approve execution code")
    lock = create_protocol_lock(
        root,
        [root / "frozen.txt"],
        models=_required_models(),
    )
    _write_protocol_lock(root, lock)
    approval = _write_approval(root, lock)
    approved_archive = subprocess.run(
        ["git", "archive", "--format=tar", lock.created_from_git_commit],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    original_verify_protocol_lock = protocol.verify_protocol_lock
    committed_replacement = False

    def commit_then_verify(
        root_path: Path,
        protocol_lock: ProtocolLock,
    ) -> ProtocolVerification:
        nonlocal committed_replacement
        if not committed_replacement:
            execution_file.write_text("SAFE = False\n", encoding="utf-8")
            _run_git(root, "add", "code.py")
            _run_git(root, "commit", "-m", "replace execution code after clean check")
            committed_replacement = True
        return original_verify_protocol_lock(root_path, protocol_lock)

    monkeypatch.setattr(protocol, "verify_protocol_lock", commit_then_verify)

    snapshot = assert_main_run_allowed(
        root,
        lock,
        approval,
        current_models=_required_models(),
    )

    assert committed_replacement is True
    assert snapshot.execution_git_commit == lock.created_from_git_commit
    assert snapshot.execution_git_commit != subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert snapshot.execution_archive_bytes == approved_archive
    assert snapshot.execution_archive_sha256 == hashlib.sha256(approved_archive).hexdigest()
    assert execution_file.read_text(encoding="utf-8") == "SAFE = False\n"
    with tarfile.open(fileobj=io.BytesIO(snapshot.execution_archive_bytes), mode="r:") as archive:
        archived_file = archive.extractfile("code.py")
        assert archived_file is not None
        assert archived_file.read() == b"SAFE = True\n"


def test_main_run_allows_a_clean_real_git_repository(tmp_path: Path) -> None:
    root = _git_repository(tmp_path)
    lock, approval = _ready_main_run(root)

    assert_main_run_allowed(root, lock, approval, current_models=_current_models())


@pytest.mark.parametrize("change", ["tracked", "staged", "untracked"])
def test_main_run_blocks_real_git_tracked_staged_and_untracked_changes(
    tmp_path: Path, change: str
) -> None:
    root = _git_repository(tmp_path)
    lock, approval = _ready_main_run(root)
    if change == "tracked":
        (root / "tracked.txt").write_text("changed\n", encoding="utf-8")
    elif change == "staged":
        (root / "tracked.txt").write_text("staged\n", encoding="utf-8")
        _run_git(root, "add", "tracked.txt")
    else:
        (root / "untracked.txt").write_text("untracked\n", encoding="utf-8")

    with pytest.raises(MainRunBlocked, match="Git tree"):
        assert_main_run_allowed(root, lock, approval, current_models=_current_models())


def test_main_run_forces_untracked_reporting_but_leaves_ignored_files_ignored(
    tmp_path: Path,
) -> None:
    root = _git_repository(tmp_path)
    (root / ".gitignore").write_text(
        "artifacts/*.json\nignored.txt\n",
        encoding="utf-8",
    )
    _run_git(root, "add", ".gitignore")
    _run_git(root, "commit", "-m", "ignore fixture output")
    lock, approval = _ready_main_run(root)
    _run_git(root, "config", "status.showUntrackedFiles", "no")
    (root / "ordinary.txt").write_text("ordinary\n", encoding="utf-8")
    (root / "ignored.txt").write_text("ignored\n", encoding="utf-8")

    with pytest.raises(MainRunBlocked, match="Git tree"):
        assert_main_run_allowed(root, lock, approval, current_models=_current_models())

    (root / "ordinary.txt").unlink()

    assert_main_run_allowed(root, lock, approval, current_models=_current_models())


def test_main_run_wraps_malformed_current_models_with_the_model_stage(tmp_path: Path) -> None:
    root = _git_repository(tmp_path)
    lock, approval = _ready_main_run(root)
    malformed_models: dict[str, object] = _required_models()
    malformed_models["generator"] = {
        "id": "mock-generator",
        "revision": "v1",
        "extra": True,
    }

    with pytest.raises(MainRunBlocked, match="current model validation") as error:
        assert_main_run_allowed(
            root,
            lock,
            approval,
            current_models=malformed_models,  # type: ignore[arg-type]
        )

    assert isinstance(error.value.__cause__, ValidationError)


@pytest.mark.parametrize(
    "current_models",
    [
        _required_model_revisions()
        | {"generator": ModelRevision.model_construct(id="", revision="v1")},
        _required_model_revisions()
        | {"generator": ModelRevision.model_construct(id="model", revision=" \t")},
        _required_model_revisions() | {" \t": ModelRevision(id="model", revision="v1")},
    ],
    ids=["constructed-blank-id", "constructed-blank-revision", "blank-model-name"],
)
def test_main_run_revalidates_nonblank_current_model_identities(
    tmp_path: Path,
    current_models: dict[str, ModelRevision],
) -> None:
    root = _git_repository(tmp_path)
    lock, approval = _ready_main_run(root)

    with pytest.raises(MainRunBlocked, match="current model validation") as error:
        assert_main_run_allowed(
            root,
            lock,
            approval,
            current_models=current_models,
        )

    assert isinstance(error.value.__cause__, (ValidationError, ValueError))


@pytest.mark.parametrize(
    "tampered_lock",
    [
        lambda lock: lock.model_copy(update={"protocol_version": "9.9.9"}),
        lambda lock: lock.model_copy(
            update={
                "models": lock.models
                | {
                    "generator": ModelRevision.model_construct(id="", revision="v1")
                }
            }
        ),
        lambda lock: lock.model_copy(
            update={
                "models": lock.models
                | {
                    "generator": ModelRevision.model_construct(id="model", revision=" \t")
                }
            }
        ),
        lambda lock: lock.model_copy(
            update={
                "models": lock.models
                | {" \t": ModelRevision(id="model", revision="v1")}
            }
        ),
    ],
    ids=["unsupported-version", "blank-id", "blank-revision", "blank-model-name"],
)
def test_main_run_revalidates_a_constructed_or_tampered_supplied_lock(
    tmp_path: Path,
    tampered_lock: Callable[[ProtocolLock], ProtocolLock],
) -> None:
    root = _git_repository(tmp_path)
    lock, approval = _ready_main_run(root)

    with pytest.raises(MainRunBlocked, match="supplied protocol lock validation") as error:
        assert_main_run_allowed(
            root,
            tampered_lock(lock),
            approval,
            current_models=_current_models(),
        )

    assert isinstance(error.value.__cause__, ValidationError)


@pytest.mark.parametrize("invalid_root", ["missing-root", "not-a-directory"])
def test_main_run_wraps_invalid_roots_with_the_on_disk_lock_stage(
    tmp_path: Path, invalid_root: str
) -> None:
    root = _git_repository(tmp_path)
    lock, approval = _ready_main_run(root)
    root_to_check = tmp_path / invalid_root
    if invalid_root == "not-a-directory":
        root_to_check.write_text("not a root\n", encoding="utf-8")

    with pytest.raises(MainRunBlocked, match="on-disk protocol lock") as error:
        assert_main_run_allowed(
            root_to_check,
            lock,
            approval,
            current_models=_current_models(),
        )

    assert isinstance(error.value.__cause__, (OSError, ValueError))


def test_main_run_blocks_an_invalid_locked_path_with_the_protocol_lock_stage(
    tmp_path: Path,
) -> None:
    root = _git_repository(tmp_path)
    lock, approval = _ready_main_run(root)
    invalid_lock = ProtocolLock.model_construct(
        protocol_version=lock.protocol_version,
        files={"../outside.txt": "a" * 64},
        models=lock.models,
        created_from_git_commit=lock.created_from_git_commit,
    )
    approval = _write_approval(root, invalid_lock)
    _write_protocol_lock(root, invalid_lock)

    with pytest.raises(MainRunBlocked, match="protocol lock verification") as error:
        assert_main_run_allowed(root, invalid_lock, approval, current_models=_current_models())

    assert error.value.__cause__ is None


def test_main_run_revalidates_a_malformed_lock_record_at_the_guard_boundary(
    tmp_path: Path,
) -> None:
    root = _git_repository(tmp_path)
    lock, approval = _ready_main_run(root)
    malformed_lock = ProtocolLock.model_construct(
        protocol_version=lock.protocol_version,
        files=None,
        models=lock.models,
        created_from_git_commit=lock.created_from_git_commit,
    )

    with pytest.raises(MainRunBlocked, match="supplied protocol lock validation") as error:
        assert_main_run_allowed(root, malformed_lock, approval, current_models=_current_models())

    assert isinstance(error.value.__cause__, ValidationError)


def test_main_run_revalidates_a_non_string_commit_at_the_guard_boundary(
    tmp_path: Path,
) -> None:
    root = _git_repository(tmp_path)
    lock, _ = _ready_main_run(root)
    invalid_lock = ProtocolLock.model_construct(
        protocol_version=lock.protocol_version,
        files={},
        models=_required_model_revisions(),
        created_from_git_commit=object(),
    )
    approval = root / "approval.json"
    approval.write_text(
        '{"lock_sha256":"not-used","approved_for_main_run":true}', encoding="utf-8"
    )

    with pytest.raises(MainRunBlocked, match="supplied protocol lock validation") as error:
        assert_main_run_allowed(
            root,
            invalid_lock,
            approval,
            current_models=_current_models(),
        )

    assert isinstance(error.value.__cause__, ValidationError)


def test_main_run_wraps_missing_repository_git_state_with_the_git_stage(tmp_path: Path) -> None:
    root = _git_repository(tmp_path)
    lock, approval = _ready_main_run(root)
    shutil.rmtree(root / ".git")

    with pytest.raises(MainRunBlocked, match="Git HEAD verification") as error:
        assert_main_run_allowed(root, lock, approval, current_models=_current_models())

    assert isinstance(error.value.__cause__, subprocess.CalledProcessError)


def test_main_run_wraps_a_missing_git_executable_with_the_git_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _git_repository(tmp_path)
    lock, approval = _ready_main_run(root)

    def missing_git(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("git")

    monkeypatch.setattr(protocol.subprocess, "run", missing_git)

    with pytest.raises(MainRunBlocked, match="Git HEAD verification") as error:
        assert_main_run_allowed(root, lock, approval, current_models=_current_models())

    assert isinstance(error.value.__cause__, FileNotFoundError)


def test_main_run_guard_has_no_override_or_bypass_parameter() -> None:
    signature = inspect.signature(assert_main_run_allowed)

    assert tuple(signature.parameters) == (
        "root",
        "lock",
        "approval_path",
        "current_models",
        "lock_path",
    )
    assert signature.parameters["current_models"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["lock_path"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["lock_path"].default is None
