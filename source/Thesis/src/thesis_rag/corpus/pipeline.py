import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Never, cast

from thesis_rag.canonical import canonical_json
from thesis_rag.corpus import chunking as chunking_module
from thesis_rag.corpus.chunking import chunk_document
from thesis_rag.corpus.html_tree import normalize_source_text
from thesis_rag.corpus.models import (
    ApprovedCorpus,
    ChunkingConfig,
    CorpusValidationReport,
    HeadingBlock,
    ParsedDocument,
    TableBlock,
    TextBlock,
)
from thesis_rag.corpus.parsers import parse_source
from thesis_rag.corpus.sources import (
    load_approved_corpus,
    verify_approved_snapshots,
)
from thesis_rag.corpus.tables import render_logical_row
from thesis_rag.domain import Chunk

_REQUIRED_CHUNK_METADATA = frozenset(
    {
        "candidate_id",
        "source_url",
        "source_title",
        "authority",
        "source_type",
        "published_or_revised",
        "snapshot_commitment",
        "source_locator",
        "page_number",
        "heading_path",
        "block_start",
        "block_end",
        "is_table",
        "review_required",
        "review_codes",
        "oversize",
    }
)


@dataclass(frozen=True, slots=True)
class _DerivedCorpus:
    documents: tuple[ParsedDocument, ...]
    chunks: tuple[Chunk, ...]
    parsed_bytes: dict[str, bytes]
    chunks_bytes: bytes
    report: CorpusValidationReport
    report_bytes: bytes


@dataclass(frozen=True, slots=True)
class _StagedArtifact:
    target: Path
    staged: Path
    backup: Path | None


def build_corpus(
    *,
    repository_root: Path,
    approved_path: Path,
    parsed_dir: Path,
    chunks_path: Path,
    report_path: Path,
    config: ChunkingConfig,
) -> CorpusValidationReport:
    corpus = load_approved_corpus(approved_path)
    _validate_artifact_paths(
        repository_root,
        corpus,
        approved_path,
        parsed_dir,
        chunks_path,
        report_path,
    )
    expected_names = {f"{source.document_id}.json" for source in corpus.sources}
    _reject_unexpected_parsed_artifacts(parsed_dir, expected_names)
    derived = _derive_corpus(repository_root, corpus, config)

    artifacts = [
        (parsed_dir / name, derived.parsed_bytes[name])
        for name in sorted(derived.parsed_bytes)
    ]
    artifacts.extend(
        (
            (chunks_path, derived.chunks_bytes),
            (report_path, derived.report_bytes),
        )
    )
    _publish_artifacts(tuple(artifacts))
    return derived.report


def verify_corpus_artifacts(
    *,
    repository_root: Path,
    approved_path: Path,
    parsed_dir: Path,
    chunks_path: Path,
    report_path: Path,
    config: ChunkingConfig,
) -> CorpusValidationReport:
    corpus = load_approved_corpus(approved_path)
    _validate_artifact_paths(
        repository_root,
        corpus,
        approved_path,
        parsed_dir,
        chunks_path,
        report_path,
    )
    derived = _derive_corpus(repository_root, corpus, config)
    expected_names = set(derived.parsed_bytes)
    actual_names = (
        {path.name for path in parsed_dir.iterdir() if path.is_file()}
        if parsed_dir.is_dir()
        else set()
    )
    unexpected = sorted(actual_names - expected_names)
    if unexpected:
        raise ValueError(f"unexpected parsed artifacts: {','.join(unexpected)}")
    missing = sorted(expected_names - actual_names)
    if missing:
        raise ValueError(f"missing parsed artifacts: {','.join(missing)}")

    for name, expected in derived.parsed_bytes.items():
        actual = (parsed_dir / name).read_bytes()
        if actual != expected:
            raise ValueError(f"parsed artifact mismatch: {name}")
    if not chunks_path.is_file() or chunks_path.read_bytes() != derived.chunks_bytes:
        raise ValueError(f"chunk artifact mismatch: {chunks_path}")
    if not report_path.is_file() or report_path.read_bytes() != derived.report_bytes:
        raise ValueError(f"validation report mismatch: {report_path}")
    return derived.report


def _derive_corpus(
    repository_root: Path,
    corpus: ApprovedCorpus,
    config: ChunkingConfig,
) -> _DerivedCorpus:
    snapshots = verify_approved_snapshots(repository_root, corpus)
    documents = tuple(
        parse_source(
            repository_root,
            source,
            snapshots[source.document_id],
        )
        for source in corpus.sources
    )
    if len(documents) != 10:
        raise ValueError("approved core corpus must contain exactly ten documents")
    chunks_by_document = tuple(
        chunk_document(document, config) for document in documents
    )
    chunks = tuple(
        chunk
        for document_chunks in chunks_by_document
        for chunk in document_chunks
    )
    _validate_corpus_records(documents, chunks, config)

    parsed_bytes = {
        f"{document.document_id}.json": (
            canonical_json(document) + "\n"
        ).encode("utf-8")
        for document in documents
    }
    chunks_bytes = "".join(
        f"{canonical_json(chunk)}\n" for chunk in chunks
    ).encode("utf-8")
    _validate_serialized_records(parsed_bytes, chunks_bytes)

    output_hashes = {
        f"parsed/{name}": hashlib.sha256(contents).hexdigest()
        for name, contents in parsed_bytes.items()
    }
    output_hashes["chunks/core.jsonl"] = hashlib.sha256(
        chunks_bytes
    ).hexdigest()
    report = CorpusValidationReport(
        corpus_id=corpus.corpus_id,
        document_count=len(documents),
        block_count=sum(len(document.blocks) for document in documents),
        table_count=sum(
            isinstance(block, TableBlock)
            for document in documents
            for block in document.blocks
        ),
        chunk_count=len(chunks),
        character_count=sum(
            _document_character_count(document) for document in documents
        ),
        review_document_count=sum(
            document.review_required for document in documents
        ),
        review_chunk_count=sum(
            chunk.metadata["review_required"] is True for chunk in chunks
        ),
        oversize_chunk_count=sum(
            chunk.metadata["oversize"] is True for chunk in chunks
        ),
        document_commitments={
            document.document_id: document.snapshot_commitment
            for document in documents
        },
        output_hashes=output_hashes,
    )
    report_bytes = (canonical_json(report) + "\n").encode("utf-8")
    CorpusValidationReport.model_validate_json(report_bytes, strict=True)
    return _DerivedCorpus(
        documents=documents,
        chunks=chunks,
        parsed_bytes=parsed_bytes,
        chunks_bytes=chunks_bytes,
        report=report,
        report_bytes=report_bytes,
    )


def _validate_corpus_records(
    documents: tuple[ParsedDocument, ...],
    chunks: tuple[Chunk, ...],
    config: ChunkingConfig,
) -> None:
    document_ids = tuple(document.document_id for document in documents)
    if len(set(document_ids)) != len(document_ids):
        raise ValueError("parsed corpus contains duplicate document IDs")
    chunk_ids = tuple(chunk.chunk_id for chunk in chunks)
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("parsed corpus contains duplicate chunk IDs")

    expected_chunks = tuple(
        expected_chunk
        for document in documents
        for expected_chunk in chunking_module.chunk_document(document, config)
    )
    if len(chunks) != len(expected_chunks):
        raise ValueError("parsed corpus chunk plan does not match source documents")

    by_document = {document.document_id: document for document in documents}
    for chunk, expected_chunk in zip(chunks, expected_chunks, strict=True):
        document = by_document.get(chunk.document_id)
        if document is None:
            raise ValueError(f"chunk references unknown document: {chunk.chunk_id}")
        if chunk.document_id != expected_chunk.document_id:
            _raise_provenance_mismatch(chunk, "chunk order")
        if set(chunk.metadata) != _REQUIRED_CHUNK_METADATA:
            raise ValueError(f"chunk provenance is incomplete: {chunk.chunk_id}")
        _validate_chunk_provenance(
            document,
            chunk,
            config,
            expected_chunk,
        )


def _validate_chunk_provenance(
    document: ParsedDocument,
    chunk: Chunk,
    config: ChunkingConfig,
    expected_chunk: Chunk,
) -> None:
    if chunk.chunk_id != expected_chunk.chunk_id:
        _raise_provenance_mismatch(chunk, "chunk_id")
    if chunk.text != expected_chunk.text:
        _raise_provenance_mismatch(chunk, "text")
    for field in sorted(_REQUIRED_CHUNK_METADATA):
        actual_value = chunk.metadata[field]
        expected_value = expected_chunk.metadata[field]
        if (
            type(actual_value) is not type(expected_value)
            or actual_value != expected_value
        ):
            _raise_provenance_mismatch(chunk, field)

    expected_document_metadata: dict[str, object] = {
        "candidate_id": document.candidate_id,
        "source_url": document.source_url,
        "source_title": document.title,
        "authority": document.authority,
        "source_type": document.source_type.value,
        "published_or_revised": document.published_or_revised,
        "snapshot_commitment": document.snapshot_commitment,
    }
    if chunk.language != document.language:
        _raise_provenance_mismatch(chunk, "language")
    for field, expected in expected_document_metadata.items():
        if chunk.metadata[field] != expected:
            _raise_provenance_mismatch(chunk, field)

    block_start = chunk.metadata["block_start"]
    block_end = chunk.metadata["block_end"]
    if (
        type(block_start) is not int
        or type(block_end) is not int
        or block_start < 1
        or block_end < block_start
        or block_end > len(document.blocks)
    ):
        _raise_provenance_mismatch(chunk, "block range")
    start = block_start
    end = block_end
    selected_blocks = document.blocks[start - 1 : end]
    if any(isinstance(block, HeadingBlock) for block in selected_blocks):
        _raise_provenance_mismatch(chunk, "block range")
    content_blocks = cast(tuple[TextBlock | TableBlock, ...], selected_blocks)

    page_numbers = {block.page_number for block in content_blocks}
    heading_paths = {block.heading_path for block in content_blocks}
    if len(page_numbers) != 1 or chunk.metadata["page_number"] != next(
        iter(page_numbers)
    ):
        _raise_provenance_mismatch(chunk, "page_number")
    if len(heading_paths) != 1:
        _raise_provenance_mismatch(chunk, "heading_path")
    heading_path = next(iter(heading_paths))
    if chunk.metadata["heading_path"] != " > ".join(heading_path):
        _raise_provenance_mismatch(chunk, "heading_path")

    is_table = len(content_blocks) == 1 and isinstance(
        content_blocks[0],
        TableBlock,
    )
    if chunk.metadata["is_table"] is not is_table:
        _raise_provenance_mismatch(chunk, "is_table")
    source_locator = chunk.metadata["source_locator"]
    expected_source_locator = expected_chunk.metadata["source_locator"]
    if (
        not isinstance(expected_source_locator, str)
        or source_locator != expected_source_locator
    ):
        _raise_provenance_mismatch(chunk, "source_locator")
    expected_text_locators = (
        frozenset({expected_source_locator})
        if not is_table
        else None
    )
    if not isinstance(source_locator, str) or not _locator_matches_blocks(
        source_locator,
        content_blocks,
        is_table,
        chunk.text,
        expected_text_locators,
    ):
        _raise_provenance_mismatch(chunk, "source_locator")

    normalized_text = normalize_source_text(chunk.text)
    review_codes = tuple(
        sorted(
            rule.code
            for rule in document.review_rules
            if all(
                normalize_source_text(term) in normalized_text
                for term in rule.all_terms
            )
        )
    )
    if chunk.metadata["review_required"] is not bool(review_codes):
        _raise_provenance_mismatch(chunk, "review_required")
    if chunk.metadata["review_codes"] != ",".join(review_codes):
        _raise_provenance_mismatch(chunk, "review_codes")
    if chunk.metadata["oversize"] is not (len(chunk.text) > config.max_chars):
        _raise_provenance_mismatch(chunk, "oversize")


def _locator_matches_blocks(
    source_locator: str,
    blocks: tuple[TextBlock | TableBlock, ...],
    is_table: bool,
    chunk_text: str,
    expected_text_locators: frozenset[str] | None = None,
) -> bool:
    if is_table:
        block = blocks[0]
        if not isinstance(block, TableBlock):
            return False
        prefix = f"{block.source_locator}#rows:"
        if not source_locator.startswith(prefix):
            return False
        bounds = source_locator.removeprefix(prefix).split("-", maxsplit=1)
        if len(bounds) != 2 or any(not bound.isdigit() for bound in bounds):
            return False
        first_row, last_row = (int(bound) for bound in bounds)
        if not 1 <= first_row <= last_row <= len(block.table.logical_rows):
            return False
        expected_rows = tuple(
            f"행: {render_logical_row(block.table, row_index)}"
            for row_index in range(first_row - 1, last_row)
        )
        actual_rows = tuple(
            line for line in chunk_text.splitlines() if line.startswith("행: ")
        )
        return actual_rows == expected_rows

    if any(not isinstance(block, TextBlock) for block in blocks):
        return False
    if source_locator.count("..") > 1:
        return False
    expected = (
        frozenset(
            {_locator_span(blocks[0].source_locator, blocks[-1].source_locator)}
        )
        if expected_text_locators is None
        else expected_text_locators
    )
    return source_locator in expected


def _locator_span(first: str, last: str) -> str:
    start = first.split("..", maxsplit=1)[0]
    end = last.rsplit("..", maxsplit=1)[-1]
    return start if start == end else f"{start}..{end}"


def _raise_provenance_mismatch(chunk: Chunk, field: str) -> Never:
    raise ValueError(
        f"chunk provenance mismatch ({field}): {chunk.chunk_id}"
    )


def _validate_serialized_records(
    parsed_bytes: dict[str, bytes],
    chunks_bytes: bytes,
) -> None:
    for contents in parsed_bytes.values():
        ParsedDocument.model_validate_json(contents, strict=True)
    if not chunks_bytes.endswith(b"\n"):
        raise ValueError("chunk JSONL must end with a newline")
    for line in chunks_bytes.splitlines():
        Chunk.model_validate_json(line, strict=True)


def _document_character_count(document: ParsedDocument) -> int:
    total = 0
    for block in document.blocks:
        if isinstance(block, HeadingBlock | TextBlock):
            total += len(block.text)
        else:
            total += sum(
                len(cell.text)
                for row in block.table.source_rows
                for cell in row.cells
            )
    return total


def _reject_unexpected_parsed_artifacts(
    parsed_dir: Path,
    expected_names: set[str],
) -> None:
    if not parsed_dir.exists():
        return
    if not parsed_dir.is_dir():
        raise ValueError(f"parsed artifact path is not a directory: {parsed_dir}")
    unexpected = sorted(
        path.name
        for path in parsed_dir.iterdir()
        if path.is_file() and path.name not in expected_names
    )
    if unexpected:
        raise ValueError(f"unexpected parsed artifacts: {','.join(unexpected)}")


def _validate_artifact_paths(
    repository_root: Path,
    corpus: ApprovedCorpus,
    approved_path: Path,
    parsed_dir: Path,
    chunks_path: Path,
    report_path: Path,
) -> None:
    root = repository_root.resolve()
    parsed = parsed_dir.resolve()
    chunks = chunks_path.resolve()
    report = report_path.resolve()

    if chunks == report or _is_below(chunks, parsed) or _is_below(report, parsed):
        raise ValueError("output paths must not overlap")
    if parsed in {chunks, report}:
        raise ValueError("output paths must not overlap")
    for output_file in (chunks, report):
        if output_file.exists() and output_file.is_dir():
            raise ValueError(f"output file path is a directory: {output_file}")

    immutable_files = {
        approved_path.resolve(),
        (root / corpus.hash_manifest).resolve(),
    }
    immutable_directories = {(root / "data/candidates/raw").resolve()}
    immutable_directories.update(
        (root / source.snapshot_path).resolve() for source in corpus.sources
    )
    if any(
        immutable_file == parsed or _is_below(immutable_file, parsed)
        for immutable_file in immutable_files
    ) or any(
        _paths_overlap(parsed, immutable_directory)
        for immutable_directory in immutable_directories
    ):
        raise ValueError("parsed output overlaps an immutable corpus input")
    for output_file in (chunks, report):
        if output_file in immutable_files or any(
            _is_below(output_file, immutable_directory)
            for immutable_directory in immutable_directories
        ):
            raise ValueError("output path targets an immutable corpus input")


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or _is_below(first, second) or _is_below(second, first)


def _is_below(path: Path, directory: Path) -> bool:
    return path != directory and path.is_relative_to(directory)


def _publish_artifacts(artifacts: tuple[tuple[Path, bytes], ...]) -> None:
    staged_artifacts: list[_StagedArtifact] = []
    published: list[_StagedArtifact] = []
    try:
        for target, contents in artifacts:
            staged_artifacts.append(_stage_artifact(target, contents))
        for artifact in staged_artifacts:
            os.replace(artifact.staged, artifact.target)
            published.append(artifact)
    except BaseException as error:
        rollback_errors: list[OSError] = []
        for artifact in reversed(published):
            try:
                if artifact.backup is None:
                    artifact.target.unlink(missing_ok=True)
                else:
                    os.replace(artifact.backup, artifact.target)
            except OSError as rollback_error:
                rollback_errors.append(rollback_error)
        if rollback_errors:
            raise RuntimeError("corpus artifact rollback failed") from error
        raise
    finally:
        for artifact in staged_artifacts:
            for temporary_path in (artifact.staged, artifact.backup):
                if temporary_path is not None and temporary_path.exists():
                    temporary_path.unlink()


def _stage_artifact(target: Path, contents: bytes) -> _StagedArtifact:
    staged = _stage_file(target, contents)
    try:
        backup = (
            _stage_file(target, target.read_bytes())
            if target.is_file()
            else None
        )
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return _StagedArtifact(target=target, staged=staged, backup=backup)


def _stage_file(path: Path, contents: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(contents)
            temporary.flush()
            os.fsync(temporary.fileno())
        return temporary_path
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
