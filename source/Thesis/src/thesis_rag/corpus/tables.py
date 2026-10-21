import re
from dataclasses import dataclass

from thesis_rag.corpus.html_tree import (
    HtmlElement,
    attribute,
    direct_child_elements,
    element_lines,
    element_text,
)
from thesis_rag.corpus.models import (
    ParsedTable,
    SourceTableCell,
    SourceTableRow,
)

_POSITIVE_INTEGER_PATTERN = re.compile(r"^[1-9][0-9]*$")


@dataclass(frozen=True, slots=True)
class _PlacedCell:
    text: str
    is_header: bool


def parse_table(element: HtmlElement) -> ParsedTable:
    if element.tag != "table":
        raise ValueError("parse_table requires a table element")

    caption_elements = direct_child_elements(element, tag="caption")
    caption = element_text(caption_elements[0]) if caption_elements else None
    source_rows = tuple(_parse_row(row) for row in _table_rows(element))
    if not source_rows:
        raise ValueError("table must contain at least one row")

    header_row_count = 0
    for row in source_rows:
        if all(cell.is_header for cell in row.cells):
            header_row_count += 1
        else:
            break
    if header_row_count == len(source_rows):
        raise ValueError("table must contain at least one data row")

    grid = _expand_spans(source_rows)
    headers = _logical_headers(grid, header_row_count)
    logical_rows = tuple(
        tuple(cell.text if cell is not None else "" for cell in row)
        for row in grid[header_row_count:]
    )
    return ParsedTable(
        caption=caption,
        source_rows=source_rows,
        logical_headers=headers,
        logical_rows=logical_rows,
    )


def render_logical_row(table: ParsedTable, row_index: int) -> str:
    if row_index < 0 or row_index >= len(table.logical_rows):
        raise IndexError(f"logical row index is out of range: {row_index}")
    row = table.logical_rows[row_index]
    return " | ".join(
        f"{header}={value}"
        for header, value in zip(table.logical_headers, row, strict=True)
    )


def _table_rows(table: HtmlElement) -> tuple[HtmlElement, ...]:
    rows: list[HtmlElement] = []

    def visit(element: HtmlElement) -> None:
        for child in direct_child_elements(element):
            if child.tag == "table":
                continue
            if child.tag == "tr":
                rows.append(child)
                continue
            visit(child)

    visit(table)
    return tuple(rows)


def _parse_row(element: HtmlElement) -> SourceTableRow:
    cells = tuple(
        SourceTableCell(
            text=" / ".join(element_lines(child)),
            is_header=child.tag == "th",
            rowspan=_parse_span(attribute(child, "rowspan")),
            colspan=_parse_span(attribute(child, "colspan")),
        )
        for child in direct_child_elements(element)
        if child.tag in {"th", "td"}
    )
    if not cells:
        raise ValueError("table row must contain direct cells")
    return SourceTableRow(cells=cells)


def _parse_span(value: str | None) -> int:
    if value is None or value == "":
        return 1
    stripped = value.strip()
    if _POSITIVE_INTEGER_PATTERN.fullmatch(stripped) is None:
        raise ValueError(f"invalid table span: {value}")
    return int(stripped)


def _expand_spans(
    rows: tuple[SourceTableRow, ...],
) -> tuple[tuple[_PlacedCell | None, ...], ...]:
    grid: list[list[_PlacedCell | None]] = [[] for _ in rows]
    for row_index, source_row in enumerate(rows):
        column_index = 0
        for cell in source_row.cells:
            while _is_occupied(grid[row_index], column_index):
                column_index += 1
            if row_index + cell.rowspan > len(rows):
                raise ValueError("table rowspan extends past the final row")
            _place_cell(
                grid,
                row_index=row_index,
                column_index=column_index,
                cell=cell,
            )
            column_index += cell.colspan

    width = max(len(row) for row in grid)
    return tuple(
        tuple(row + [None] * (width - len(row)))
        for row in grid
    )


def _is_occupied(row: list[_PlacedCell | None], index: int) -> bool:
    return index < len(row) and row[index] is not None


def _place_cell(
    grid: list[list[_PlacedCell | None]],
    *,
    row_index: int,
    column_index: int,
    cell: SourceTableCell,
) -> None:
    for target_row_index in range(row_index, row_index + cell.rowspan):
        target_row = grid[target_row_index]
        required_width = column_index + cell.colspan
        if len(target_row) < required_width:
            target_row.extend([None] * (required_width - len(target_row)))
        for target_column_index in range(
            column_index,
            column_index + cell.colspan,
        ):
            if target_row[target_column_index] is not None:
                raise ValueError("overlapping table spans")

    placed = _PlacedCell(text=cell.text, is_header=bool(cell.is_header))
    for target_row_index in range(row_index, row_index + cell.rowspan):
        for target_column_index in range(
            column_index,
            column_index + cell.colspan,
        ):
            grid[target_row_index][target_column_index] = placed


def _logical_headers(
    grid: tuple[tuple[_PlacedCell | None, ...], ...],
    header_row_count: int,
) -> tuple[str, ...]:
    width = len(grid[0])
    headers: list[str] = []
    for column_index in range(width):
        labels: list[str] = []
        for row in grid[:header_row_count]:
            cell = row[column_index]
            label = cell.text if cell is not None else ""
            if label and (not labels or labels[-1] != label):
                labels.append(label)
        headers.append(" > ".join(labels) if labels else f"열 {column_index + 1}")
    return tuple(headers)
