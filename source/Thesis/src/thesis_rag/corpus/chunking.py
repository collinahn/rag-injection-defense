from dataclasses import dataclass

from thesis_rag.canonical import JsonValue, canonical_sha256
from thesis_rag.corpus.html_tree import normalize_source_text
from thesis_rag.corpus.models import (
    ChunkingConfig,
    HeadingBlock,
    ParsedDocument,
    ParsedTable,
    ReviewRule,
    TableBlock,
    TextBlock,
)
from thesis_rag.corpus.tables import render_logical_row
from thesis_rag.domain import Chunk, MetadataValue

_SPLIT_BOUNDARIES = ("\n", "다. ", "요. ", ". ", "; ")


@dataclass(frozen=True, slots=True)
class _TextUnit:
    text: str
    block_ordinal: int
    source_locator: str


@dataclass(frozen=True, slots=True)
class _ChunkDraft:
    text: str
    heading_path: tuple[str, ...]
    page_number: int | None
    block_start: int
    block_end: int
    source_locator: str
    is_table: bool
    oversize: bool


def chunk_document(
    document: ParsedDocument,
    config: ChunkingConfig,
) -> tuple[Chunk, ...]:
    drafts: list[_ChunkDraft] = []
    pending: list[TextBlock] = []
    pending_key: tuple[tuple[str, ...], int | None] | None = None

    def flush_pending() -> None:
        nonlocal pending, pending_key
        if pending:
            drafts.extend(_chunk_text_group(document, tuple(pending), config))
        pending = []
        pending_key = None

    for block in document.blocks:
        if isinstance(block, HeadingBlock):
            flush_pending()
            continue
        if isinstance(block, TableBlock):
            flush_pending()
            drafts.extend(_chunk_table(document, block, config))
            continue

        key = (block.heading_path, block.page_number)
        if pending_key is not None and key != pending_key:
            flush_pending()
        pending_key = key
        pending.append(block)
    flush_pending()

    if not drafts:
        raise ValueError(f"document produced no chunks: {document.document_id}")

    chunks = tuple(
        _materialize_chunk(document, draft, ordinal)
        for ordinal, draft in enumerate(drafts, start=1)
    )
    chunk_ids = tuple(chunk.chunk_id for chunk in chunks)
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError(f"duplicate chunk IDs for document: {document.document_id}")
    return chunks


def _chunk_text_group(
    document: ParsedDocument,
    blocks: tuple[TextBlock, ...],
    config: ChunkingConfig,
) -> tuple[_ChunkDraft, ...]:
    heading_path = blocks[0].heading_path
    page_number = blocks[0].page_number
    prefix = _text_prefix(document, heading_path, page_number)
    max_body_chars = config.max_chars - len(prefix)

    units = _text_units(
        blocks,
        max_body_chars=max_body_chars,
        overlap_chars=config.overlap_chars,
    )

    packed: list[tuple[_TextUnit, ...]] = []
    current: list[_TextUnit] = []
    for unit in units:
        if not current:
            current.append(unit)
            continue
        candidate = (*current, unit)
        candidate_length = len(_render_text_units(prefix, candidate))
        current_length = len(_render_text_units(prefix, tuple(current)))
        if candidate_length <= config.target_chars or (
            current_length < config.min_chars
            and candidate_length <= config.max_chars
        ):
            current.append(unit)
            continue

        packed.append(tuple(current))
        current = list(_overlap_units(tuple(current), config.overlap_chars))
        while current and len(
            _render_text_units(prefix, (*current, unit))
        ) > config.max_chars:
            current.pop(0)
        current.append(unit)
    if current:
        packed.append(tuple(current))

    return tuple(
        _text_draft(prefix, heading_path, page_number, group, config.max_chars)
        for group in packed
    )


def _text_units(
    blocks: tuple[TextBlock, ...],
    *,
    max_body_chars: int,
    overlap_chars: int,
) -> tuple[_TextUnit, ...]:
    units: list[_TextUnit] = []
    for block in blocks:
        segments = _split_long_text(
            block.text,
            max_body_chars=max_body_chars,
            overlap_chars=overlap_chars,
        )
        locators = _segment_source_locators(
            block.source_locator,
            len(segments),
        )
        units.extend(
            _TextUnit(
                text=segment,
                block_ordinal=block.ordinal,
                source_locator=locator,
            )
            for segment, locator in zip(segments, locators, strict=True)
        )
    return tuple(units)


def _segment_source_locators(
    source_locator: str,
    segment_count: int,
) -> tuple[str, ...]:
    if segment_count == 1:
        return (source_locator,)
    return tuple(
        _locator_with_part(source_locator, part_number)
        for part_number in range(1, segment_count + 1)
    )


def _locator_with_part(source_locator: str, part_number: int) -> str:
    first = _locator_start(source_locator)
    last = _locator_end(source_locator)
    first_part = f"{first}#part:{part_number}"
    if first == last:
        return first_part
    return f"{first_part}..{last}#part:{part_number}"


def _split_long_text(
    text: str,
    *,
    max_body_chars: int,
    overlap_chars: int,
) -> tuple[str, ...]:
    normalized = normalize_source_text(text)
    if max_body_chars <= 0 or len(normalized) <= max_body_chars:
        return (normalized,)

    overlap = min(overlap_chars, max_body_chars // 2)
    parts: list[str] = []
    start = 0
    while start < len(normalized):
        hard_end = min(len(normalized), start + max_body_chars)
        end = hard_end
        if hard_end < len(normalized):
            relative = normalized[start:hard_end]
            boundary_end = 0
            for boundary in _SPLIT_BOUNDARIES:
                position = relative.rfind(boundary)
                if position >= 0:
                    boundary_end = max(boundary_end, position + len(boundary))
            if boundary_end >= max_body_chars // 2:
                end = start + boundary_end
        part = normalized[start:end].strip()
        if not part:
            raise ValueError("long-text splitting produced an empty segment")
        parts.append(part)
        if end == len(normalized):
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start
    return tuple(parts)


def _overlap_units(
    units: tuple[_TextUnit, ...],
    overlap_chars: int,
) -> tuple[_TextUnit, ...]:
    if overlap_chars <= 0:
        return ()
    selected: list[_TextUnit] = []
    used = 0
    for unit in reversed(units):
        added = len(unit.text) + (1 if selected else 0)
        if used + added > overlap_chars:
            break
        selected.append(unit)
        used += added
    selected.reverse()
    return tuple(selected)


def _render_text_units(
    prefix: str,
    units: tuple[_TextUnit, ...],
) -> str:
    return prefix + "\n".join(unit.text for unit in units)


def _text_draft(
    prefix: str,
    heading_path: tuple[str, ...],
    page_number: int | None,
    units: tuple[_TextUnit, ...],
    max_chars: int,
) -> _ChunkDraft:
    text = _render_text_units(prefix, units)
    return _ChunkDraft(
        text=text,
        heading_path=heading_path,
        page_number=page_number,
        block_start=min(unit.block_ordinal for unit in units),
        block_end=max(unit.block_ordinal for unit in units),
        source_locator=_locator_range(units),
        is_table=False,
        oversize=len(text) > max_chars,
    )


def _chunk_table(
    document: ParsedDocument,
    block: TableBlock,
    config: ChunkingConfig,
) -> tuple[_ChunkDraft, ...]:
    table = block.table
    table_title = table.caption or (
        block.heading_path[-1] if block.heading_path else "표"
    )
    prefix_lines = [
        f"문서: {document.title}",
        f"섹션: {_heading_label(block.heading_path)}",
    ]
    if block.page_number is not None:
        prefix_lines.append(f"페이지: {block.page_number}")
    prefix_lines.extend(
        (
            f"표: {table_title}",
            f"열: {' | '.join(table.logical_headers)}",
        )
    )
    prefix = "\n".join(prefix_lines) + "\n"
    row_lines = tuple(
        f"행: {render_logical_row(table, row_index)}"
        for row_index in range(len(table.logical_rows))
    )
    atomic_row_groups: list[tuple[int, int]] = []
    for unit_start, unit_end in _table_atomic_row_groups(table):
        unit_text = prefix + "\n".join(row_lines[unit_start:unit_end])
        if unit_end - unit_start > 1 and len(unit_text) > config.max_chars:
            atomic_row_groups.extend(
                (row_index, row_index + 1)
                for row_index in range(unit_start, unit_end)
            )
        else:
            atomic_row_groups.append((unit_start, unit_end))

    groups: list[tuple[int, int]] = []
    start: int | None = None
    current: list[str] = []
    current_end = 0
    for unit_start, unit_end in atomic_row_groups:
        unit_lines = row_lines[unit_start:unit_end]
        candidate = (*current, *unit_lines)
        candidate_text = prefix + "\n".join(candidate)
        if not current or len(candidate_text) <= config.target_chars:
            if start is None:
                start = unit_start
            current.extend(unit_lines)
            current_end = unit_end
            continue
        current_text = prefix + "\n".join(current)
        if (
            len(current_text) < config.min_chars
            and len(candidate_text) <= config.max_chars
        ):
            current.extend(unit_lines)
            current_end = unit_end
            continue
        if start is None:
            raise ValueError("table chunking lost its row-group start")
        groups.append((start, current_end))
        start = unit_start
        current = list(unit_lines)
        current_end = unit_end
    if current:
        if start is None:
            raise ValueError("table chunking lost its final row-group start")
        groups.append((start, current_end))

    drafts = []
    for start_index, end_index in groups:
        text = prefix + "\n".join(row_lines[start_index:end_index])
        drafts.append(
            _ChunkDraft(
                text=text,
                heading_path=block.heading_path,
                page_number=block.page_number,
                block_start=block.ordinal,
                block_end=block.ordinal,
                source_locator=(
                    f"{block.source_locator}#rows:{start_index + 1}-{end_index}"
                ),
                is_table=True,
                oversize=len(text) > config.max_chars,
            )
        )
    return tuple(drafts)


def _table_atomic_row_groups(
    table: ParsedTable,
) -> tuple[tuple[int, int], ...]:
    source_rows = table.source_rows
    logical_row_count = len(table.logical_rows)
    header_row_count = len(source_rows) - logical_row_count
    blocked_boundaries: set[int] = set()
    for logical_row_index, source_row in enumerate(
        source_rows[header_row_count:]
    ):
        for cell in source_row.cells:
            span_end = min(logical_row_count, logical_row_index + cell.rowspan)
            blocked_boundaries.update(
                range(logical_row_index + 1, span_end)
            )

    groups: list[tuple[int, int]] = []
    start = 0
    for boundary in range(1, logical_row_count + 1):
        if boundary in blocked_boundaries:
            continue
        groups.append((start, boundary))
        start = boundary
    return tuple(groups)


def _materialize_chunk(
    document: ParsedDocument,
    draft: _ChunkDraft,
    ordinal: int,
) -> Chunk:
    review_codes = _matching_review_codes(document.review_rules, draft.text)
    source_locator = draft.source_locator
    payload: JsonValue = {
        "document_id": document.document_id,
        "ordinal": ordinal,
        "text": draft.text,
        "source_locator": source_locator,
        "page_number": draft.page_number,
        "heading_path": list(draft.heading_path),
        "review_codes": list(review_codes),
    }
    chunk_id = (
        f"CH_{document.document_id}_{ordinal:04d}_"
        f"{canonical_sha256(payload)[:12]}"
    )
    metadata: dict[str, MetadataValue] = {
        "candidate_id": document.candidate_id,
        "source_url": document.source_url,
        "source_title": document.title,
        "authority": document.authority,
        "source_type": document.source_type.value,
        "published_or_revised": document.published_or_revised,
        "snapshot_commitment": document.snapshot_commitment,
        "source_locator": source_locator,
        "page_number": draft.page_number,
        "heading_path": " > ".join(draft.heading_path),
        "block_start": draft.block_start,
        "block_end": draft.block_end,
        "is_table": draft.is_table,
        "review_required": bool(review_codes),
        "review_codes": ",".join(review_codes),
        "oversize": draft.oversize,
    }
    return Chunk(
        chunk_id=chunk_id,
        document_id=document.document_id,
        text=draft.text,
        language=document.language,
        metadata=metadata,
    )


def _matching_review_codes(
    rules: tuple[ReviewRule, ...],
    text: str,
) -> tuple[str, ...]:
    normalized_text = normalize_source_text(text)
    codes = [
        rule.code
        for rule in rules
        if all(
            normalize_source_text(term) in normalized_text
            for term in rule.all_terms
        )
    ]
    return tuple(sorted(codes))


def _text_prefix(
    document: ParsedDocument,
    heading_path: tuple[str, ...],
    page_number: int | None,
) -> str:
    lines = [
        f"문서: {document.title}",
        f"섹션: {_heading_label(heading_path)}",
    ]
    if page_number is not None:
        lines.append(f"페이지: {page_number}")
    return "\n".join(lines) + "\n\n"


def _heading_label(heading_path: tuple[str, ...]) -> str:
    return " > ".join(heading_path) if heading_path else "문서 전체"


def _locator_range(units: tuple[_TextUnit, ...]) -> str:
    first = _locator_start(units[0].source_locator)
    last = _locator_end(units[-1].source_locator)
    return first if first == last else f"{first}..{last}"


def _locator_start(locator: str) -> str:
    return locator.split("..", maxsplit=1)[0]


def _locator_end(locator: str) -> str:
    return locator.rsplit("..", maxsplit=1)[-1]
