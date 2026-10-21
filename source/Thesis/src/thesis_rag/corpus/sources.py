import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlparse

import yaml  # type: ignore[import-untyped]
from pydantic import ValidationError

from thesis_rag.canonical import JsonValue, canonical_sha256
from thesis_rag.corpus.models import ApprovedCorpus, ApprovedSource, VerifiedSnapshot

EXPECTED_CORE_MAPPING = (
    ("DOC_001", "CAND_001"),
    ("DOC_002", "CAND_002"),
    ("DOC_003", "CAND_003"),
    ("DOC_004", "CAND_004"),
    ("DOC_005", "CAND_005"),
    ("DOC_006", "CAND_006"),
    ("DOC_007", "CAND_007"),
    ("DOC_008", "CAND_009"),
    ("DOC_009", "CAND_010"),
    ("DOC_010", "CAND_011"),
)
_HASH_LINE_PATTERN = re.compile(r"^([0-9a-f]{64})  ([^\r\n]+)$")
_CAPSTONE_REVIEW_DOCUMENT_IDS = frozenset({"DOC_001", "DOC_002"})
_CAPSTONE_REVIEW_CODE = "CAPSTONE_THESIS_STAGE_TIMING_REVIEW"
_CAPSTONE_REVIEW_TERMS = frozenset({"종합설계", "졸업논문"})


def load_approved_corpus(path: Path) -> ApprovedCorpus:
    try:
        raw_value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"unable to read approved corpus manifest: {path}") from error
    if not isinstance(raw_value, dict):
        raise ValueError("approved corpus manifest must contain a mapping")
    try:
        corpus = ApprovedCorpus.model_validate(raw_value)
    except ValidationError:
        raise

    mapping = tuple(
        (source.document_id, source.candidate_id) for source in corpus.sources
    )
    if mapping != EXPECTED_CORE_MAPPING:
        raise ValueError("sources must match the exact approved core mapping")

    for source in corpus.sources:
        if source.document_id not in _CAPSTONE_REVIEW_DOCUMENT_IDS:
            continue
        if not any(
            rule.code == _CAPSTONE_REVIEW_CODE
            and frozenset(rule.all_terms) == _CAPSTONE_REVIEW_TERMS
            for rule in source.review_rules
        ):
            raise ValueError(
                f"required review rule is missing: {source.document_id}"
            )

    allowed_domains = set(corpus.allowed_domains)
    for source in corpus.sources:
        parsed = urlparse(source.source_url)
        if parsed.scheme != "https":
            raise ValueError(f"source URL must use HTTPS: {source.document_id}")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError(f"source URL must not contain credentials: {source.document_id}")
        hostname = parsed.hostname.casefold() if parsed.hostname is not None else ""
        if hostname not in allowed_domains:
            raise ValueError(
                f"source URL is outside allowed domains: {source.document_id}"
            )
    return corpus


def load_snapshot_hashes(path: Path) -> dict[str, str]:
    try:
        contents = path.read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"unable to read snapshot hash manifest: {path}") from error
    if not contents:
        raise ValueError("snapshot hash manifest is empty")
    if not contents.endswith("\n"):
        raise ValueError("snapshot hash manifest must end with a newline")

    hashes: dict[str, str] = {}
    for line_number, line in enumerate(contents.splitlines(), start=1):
        match = _HASH_LINE_PATTERN.fullmatch(line)
        if match is None:
            raise ValueError(f"invalid snapshot hash line: {line_number}")
        digest, relative_path = match.groups()
        relative = _safe_relative_path(relative_path)
        if relative in hashes:
            raise ValueError(f"duplicate snapshot hash path: {relative}")
        hashes[relative] = digest
    return hashes


def verify_source_snapshot(
    repository_root: Path,
    source: ApprovedSource,
    hashes: Mapping[str, str],
) -> VerifiedSnapshot:
    root = repository_root.resolve()
    source_root = _resolve_below(root, source.snapshot_path)
    if not source_root.is_dir():
        raise ValueError(f"snapshot directory is missing: {source.snapshot_path}")

    prefix = f"{source.snapshot_path.rstrip('/')}/"
    expected_paths = {
        path for path in hashes if path == source.snapshot_path or path.startswith(prefix)
    }
    actual_paths: set[str] = set()
    for path in source_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"snapshot files must not be symlinks: {path}")
        if path.is_file():
            actual_paths.add(path.relative_to(root).as_posix())

    unmanifested = sorted(actual_paths - expected_paths)
    if unmanifested:
        raise ValueError(f"unmanifested snapshot files: {','.join(unmanifested)}")
    missing = sorted(expected_paths - actual_paths)
    if missing:
        raise ValueError(f"missing snapshot files: {','.join(missing)}")
    if not actual_paths:
        raise ValueError(f"snapshot directory is empty: {source.snapshot_path}")

    verified_hashes: dict[str, str] = {}
    for relative in sorted(actual_paths):
        actual_digest = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        expected_digest = hashes[relative]
        if actual_digest != expected_digest:
            raise ValueError(f"snapshot hash mismatch: {relative}")
        verified_hashes[relative] = actual_digest

    commitment_payload: JsonValue = [
        {"path": relative, "sha256": verified_hashes[relative]}
        for relative in sorted(verified_hashes)
    ]
    return VerifiedSnapshot(
        document_id=source.document_id,
        root_path=source.snapshot_path,
        files=tuple(sorted(verified_hashes)),
        file_hashes=verified_hashes,
        commitment=canonical_sha256(commitment_payload),
    )


def verify_approved_snapshots(
    repository_root: Path,
    corpus: ApprovedCorpus,
) -> dict[str, VerifiedSnapshot]:
    hash_path = _resolve_below(repository_root.resolve(), corpus.hash_manifest)
    hashes = load_snapshot_hashes(hash_path)
    verified: dict[str, VerifiedSnapshot] = {}
    for source in corpus.sources:
        verified[source.document_id] = verify_source_snapshot(
            repository_root,
            source,
            hashes,
        )
    return verified


def _safe_relative_path(value: str) -> str:
    if not value or "\\" in value:
        raise ValueError("snapshot hash path must be a relative repository path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() == ".":
        raise ValueError("snapshot hash path must be a relative repository path")
    return path.as_posix()


def _resolve_below(root: Path, relative_path: str) -> Path:
    safe = _safe_relative_path(relative_path)
    resolved = (root / safe).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("path resolves outside repository root") from error
    return resolved
