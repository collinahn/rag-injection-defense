import hashlib
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from thesis_rag.canonical import canonical_sha256
from thesis_rag.corpus.models import (
    ApprovedSource,
    CorpusSourceType,
)
from thesis_rag.corpus.sources import (
    EXPECTED_CORE_MAPPING,
    load_approved_corpus,
    load_snapshot_hashes,
    verify_approved_snapshots,
    verify_source_snapshot,
)
from thesis_rag.domain import Language


def _approved_source(
    *,
    document_id: str = "DOC_001",
    candidate_id: str = "CAND_001",
    snapshot_path: str = "raw/CAND_001",
) -> ApprovedSource:
    return ApprovedSource(
        document_id=document_id,
        candidate_id=candidate_id,
        title=f"Title {document_id}",
        source_url=f"https://www.hufs.ac.kr/{document_id.lower()}",
        authority="HUFS",
        source_type=CorpusSourceType.K2WEB_CONTENT,
        snapshot_path=snapshot_path,
        language=Language.KO,
    )


def _source_values(document_id: str, candidate_id: str) -> dict[str, object]:
    candidate_number = int(candidate_id.removeprefix("CAND_"))
    source_type = (
        "ebook_html_collection"
        if candidate_number in {9, 10, 11}
        else "k2web_board"
        if candidate_number in {2, 3}
        else "k2web_content"
    )
    values: dict[str, object] = {
        "document_id": document_id,
        "candidate_id": candidate_id,
        "title": f"Title {document_id}",
        "source_url": f"https://www.hufs.ac.kr/{document_id.lower()}",
        "authority": "HUFS",
        "source_type": source_type,
        "snapshot_path": f"data/candidates/raw/{candidate_id}",
        "language": "ko",
    }
    if source_type == "ebook_html_collection":
        values["page_count"] = 2
    if document_id in {"DOC_001", "DOC_002"}:
        values["review_rules"] = [
            {
                "code": "CAPSTONE_THESIS_STAGE_TIMING_REVIEW",
                "all_terms": ["종합설계", "졸업논문"],
                "note": "사람 검토 필요",
            }
        ]
    return values


def _write_approved_manifest(
    path: Path,
    *,
    mapping: tuple[tuple[str, str], ...] = EXPECTED_CORE_MAPPING,
    allowed_domains: list[str] | None = None,
) -> None:
    payload = {
        "schema_version": 1,
        "collection_status": "approved",
        "approved_at": "2026-07-28",
        "corpus_id": "hufs-core-v1",
        "allowed_domains": allowed_domains or ["www.hufs.ac.kr"],
        "hash_manifest": "data/candidates/snapshot-files.sha256",
        "sources": [
            _source_values(document_id, candidate_id)
            for document_id, candidate_id in mapping
        ],
    }
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _write_hash_manifest(path: Path, entries: dict[str, bytes]) -> dict[str, str]:
    hashes = {
        relative: hashlib.sha256(contents).hexdigest()
        for relative, contents in entries.items()
    }
    path.write_text(
        "".join(f"{hashes[relative]}  {relative}\n" for relative in entries),
        encoding="ascii",
    )
    return hashes


def test_load_approved_corpus_accepts_exact_core_mapping(tmp_path: Path) -> None:
    manifest = tmp_path / "approved.yaml"
    _write_approved_manifest(manifest)

    corpus = load_approved_corpus(manifest)

    assert tuple(
        (source.document_id, source.candidate_id) for source in corpus.sources
    ) == EXPECTED_CORE_MAPPING


def test_load_approved_corpus_rejects_changed_core_mapping(tmp_path: Path) -> None:
    manifest = tmp_path / "approved.yaml"
    changed = (("DOC_001", "CAND_008"),) + EXPECTED_CORE_MAPPING[1:]
    _write_approved_manifest(manifest, mapping=changed)

    with pytest.raises(ValueError, match="exact approved core mapping"):
        load_approved_corpus(manifest)


def test_load_approved_corpus_rejects_duplicate_ids(tmp_path: Path) -> None:
    manifest = tmp_path / "approved.yaml"
    duplicated = EXPECTED_CORE_MAPPING[:-1] + (("DOC_001", "CAND_011"),)
    _write_approved_manifest(manifest, mapping=duplicated)

    with pytest.raises(ValueError, match="duplicate document_id"):
        load_approved_corpus(manifest)


def test_load_approved_corpus_rejects_unapproved_host(tmp_path: Path) -> None:
    manifest = tmp_path / "approved.yaml"
    _write_approved_manifest(manifest)
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    payload["sources"][0]["source_url"] = "https://example.com/document"
    manifest.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="outside allowed domains"):
        load_approved_corpus(manifest)


def test_load_approved_corpus_requires_https(tmp_path: Path) -> None:
    manifest = tmp_path / "approved.yaml"
    _write_approved_manifest(manifest)
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    payload["sources"][0]["source_url"] = "http://www.hufs.ac.kr/document"
    manifest.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="HTTPS"):
        load_approved_corpus(manifest)


def test_load_snapshot_hashes_rejects_duplicate_path(tmp_path: Path) -> None:
    manifest = tmp_path / "hashes.sha256"
    manifest.write_text(
        f"{'0' * 64}  raw/source.html\n{'1' * 64}  raw/source.html\n",
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="duplicate snapshot hash path"):
        load_snapshot_hashes(manifest)


def test_load_snapshot_hashes_rejects_malformed_line(tmp_path: Path) -> None:
    manifest = tmp_path / "hashes.sha256"
    manifest.write_text("not-a-hash source.html\n", encoding="ascii")

    with pytest.raises(ValueError, match="invalid snapshot hash line"):
        load_snapshot_hashes(manifest)


def test_snapshot_verification_rejects_changed_bytes(tmp_path: Path) -> None:
    source_dir = tmp_path / "raw" / "CAND_001"
    source_dir.mkdir(parents=True)
    source_file = source_dir / "source.html"
    source_file.write_bytes(b"changed")
    manifest = tmp_path / "hashes.sha256"
    manifest.write_text(
        f"{'0' * 64}  raw/CAND_001/source.html\n",
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="snapshot hash mismatch"):
        verify_source_snapshot(
            tmp_path,
            _approved_source(),
            load_snapshot_hashes(manifest),
        )


def test_snapshot_verification_rejects_unmanifested_file(tmp_path: Path) -> None:
    source_dir = tmp_path / "raw" / "CAND_001"
    source_dir.mkdir(parents=True)
    source_file = source_dir / "source.html"
    source_file.write_bytes(b"source")
    (source_dir / "extra.txt").write_bytes(b"extra")
    manifest = tmp_path / "hashes.sha256"
    _write_hash_manifest(
        manifest,
        {"raw/CAND_001/source.html": source_file.read_bytes()},
    )

    with pytest.raises(ValueError, match="unmanifested snapshot files"):
        verify_source_snapshot(
            tmp_path,
            _approved_source(),
            load_snapshot_hashes(manifest),
        )


def test_snapshot_verification_rejects_manifested_missing_file(tmp_path: Path) -> None:
    source_dir = tmp_path / "raw" / "CAND_001"
    source_dir.mkdir(parents=True)
    manifest = tmp_path / "hashes.sha256"
    _write_hash_manifest(
        manifest,
        {"raw/CAND_001/source.html": b"missing"},
    )

    with pytest.raises(ValueError, match="missing snapshot files"):
        verify_source_snapshot(
            tmp_path,
            _approved_source(),
            load_snapshot_hashes(manifest),
        )


def test_snapshot_commitment_is_sorted_by_relative_path(tmp_path: Path) -> None:
    source_dir = tmp_path / "raw" / "CAND_001"
    source_dir.mkdir(parents=True)
    entries = {
        "raw/CAND_001/z.txt": b"z",
        "raw/CAND_001/a.txt": b"a",
    }
    for relative, contents in entries.items():
        (tmp_path / relative).write_bytes(contents)
    manifest = tmp_path / "hashes.sha256"
    hashes = _write_hash_manifest(manifest, entries)

    verified = verify_source_snapshot(
        tmp_path,
        _approved_source(),
        load_snapshot_hashes(manifest),
    )

    assert verified.files == (
        "raw/CAND_001/a.txt",
        "raw/CAND_001/z.txt",
    )
    assert verified.commitment == canonical_sha256(
        [
            {"path": relative, "sha256": hashes[relative]}
            for relative in sorted(entries)
        ]
    )


def test_verify_all_snapshots_covers_every_approved_document(tmp_path: Path) -> None:
    manifest_path = tmp_path / "approved.yaml"
    _write_approved_manifest(manifest_path)
    corpus = load_approved_corpus(manifest_path)
    entries: dict[str, bytes] = {}
    for source in corpus.sources:
        source_dir = tmp_path / source.snapshot_path
        source_dir.mkdir(parents=True)
        filename = "page_001.html" if source.page_count else "source.html"
        source_file = source_dir / filename
        source_file.write_bytes(source.document_id.encode("ascii"))
        entries[source_file.relative_to(tmp_path).as_posix()] = source_file.read_bytes()
        if source.page_count:
            second_page = source_dir / "page_002.html"
            second_page.write_bytes(f"{source.document_id}-2".encode("ascii"))
            entries[second_page.relative_to(tmp_path).as_posix()] = second_page.read_bytes()
    hash_manifest = tmp_path / corpus.hash_manifest
    hash_manifest.parent.mkdir(parents=True, exist_ok=True)
    _write_hash_manifest(hash_manifest, entries)

    verified = verify_approved_snapshots(tmp_path, corpus)

    assert tuple(verified) == tuple(
        source.document_id for source in corpus.sources
    )


def test_manifest_model_rejects_unknown_fields(tmp_path: Path) -> None:
    manifest = tmp_path / "approved.yaml"
    _write_approved_manifest(manifest)
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    manifest.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_approved_corpus(manifest)
