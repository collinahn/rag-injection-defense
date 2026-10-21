import hashlib
import re
from pathlib import Path
from typing import Literal

from thesis_rag.corpus.html_tree import (
    HtmlElement,
    HtmlText,
    attribute,
    element_lines,
    element_text,
    find_by_id,
    find_first_by_class,
    find_first_tag,
    iter_elements,
    normalize_source_text,
    parse_html,
)
from thesis_rag.corpus.models import (
    ApprovedSource,
    ContentBlock,
    CorpusSourceType,
    EbookLineNormalizationAction,
    EbookTableReplacementRule,
    EbookTextReplacementRule,
    HeadingBlock,
    ParsedDocument,
    ParsedTable,
    SourceTableCell,
    SourceTableRow,
    TableBlock,
    TextBlock,
    VerifiedSnapshot,
)
from thesis_rag.corpus.tables import parse_table

_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_TEXT_TAGS = frozenset({"p"})
_LIST_TAGS = frozenset({"li", "dt", "dd"})
_CONTAINER_TAGS = frozenset(
    {
        "[document]",
        "article",
        "aside",
        "blockquote",
        "div",
        "dl",
        "fieldset",
        "figure",
        "form",
        "main",
        "nav",
        "ol",
        "section",
        "ul",
    }
)
_DATE_PATTERN = re.compile(r"\b([0-9]{4})[.]([0-9]{2})[.]([0-9]{2})\b")
_ARTICLE_PATTERN = re.compile(
    r"^(제\s*[0-9]+\s*조(?:\s*의\s*[0-9]+)?(?:\s*\([^)]*\))?)\s*(.*)$"
)
_EBOOK_PAGE_NAME_PATTERN = re.compile(r"page_([0-9]{3})[.]html$")


def parse_source(
    repository_root: Path,
    source: ApprovedSource,
    snapshot: VerifiedSnapshot,
) -> ParsedDocument:
    if snapshot.document_id != source.document_id:
        raise ValueError("snapshot document_id must match approved source")
    if snapshot.root_path != source.snapshot_path:
        raise ValueError("snapshot root_path must match approved source")

    if source.source_type is CorpusSourceType.K2WEB_CONTENT:
        return _parse_k2web_content(repository_root, source, snapshot)
    if source.source_type is CorpusSourceType.K2WEB_BOARD:
        return _parse_k2web_board(repository_root, source, snapshot)
    return _parse_ebook(repository_root, source, snapshot)


class _BlockCollector:
    def __init__(self, locator_prefix: str) -> None:
        self._locator_prefix = locator_prefix
        self._headings: list[tuple[int, str]] = []
        self.blocks: list[ContentBlock] = []

    def collect(self, root: HtmlElement) -> tuple[ContentBlock, ...]:
        self._visit_container(root)
        if not self.blocks:
            raise ValueError("source body produced no content blocks")
        return tuple(self.blocks)

    def _visit_container(self, element: HtmlElement) -> None:
        if _has_html_class(element, "people-info"):
            return
        buffered_inline: list[str] = []

        def flush_inline() -> None:
            raw_text = "".join(buffered_inline)
            buffered_inline.clear()
            for raw_line in raw_text.splitlines():
                text = normalize_source_text(raw_line)
                if text:
                    self._append_text("paragraph", text)

        for child in element.children:
            if isinstance(child, HtmlText):
                buffered_inline.append(child.value)
                continue
            if child.tag in _HEADING_TAGS | _TEXT_TAGS | _LIST_TAGS | {"table"}:
                flush_inline()
                self._emit_semantic(child)
                continue
            if child.tag == "br":
                buffered_inline.append("\n")
                continue
            if child.tag in _CONTAINER_TAGS:
                flush_inline()
                self._visit_container(child)
                continue
            inline_text = element_text(child)
            if inline_text:
                buffered_inline.append(inline_text)
        flush_inline()

    def _emit_semantic(self, element: HtmlElement) -> None:
        if element.tag in _HEADING_TAGS:
            text = element_text(element)
            if text:
                self._append_heading(int(element.tag[1]), text)
            return
        if element.tag in _TEXT_TAGS:
            text = element_text(element)
            bold = find_first_tag(element, "b")
            bold_text = element_text(bold) if bold is not None else ""
            if (
                bold_text
                and re.match(r"^[0-9]+[.]\s+", bold_text)
                and text.startswith(bold_text)
            ):
                self._append_heading(3, bold_text)
                remainder = normalize_source_text(text[len(bold_text) :])
                if remainder:
                    self._append_text("paragraph", remainder)
            else:
                for line in element_lines(element):
                    self._append_text("paragraph", line)
            return
        if element.tag in _LIST_TAGS:
            text = _list_item_text(element)
            if text:
                self._append_text("list_item", text)
            return
        if element.tag == "table":
            self.blocks.append(
                TableBlock(
                    ordinal=self._next_ordinal(),
                    heading_path=self._heading_path(),
                    source_locator=self._next_locator(),
                    table=parse_table(element),
                )
            )

    def _append_heading(self, level: int, text: str) -> None:
        self._headings = [
            (existing_level, existing_text)
            for existing_level, existing_text in self._headings
            if existing_level < level
        ]
        self._headings.append((level, text))
        self.blocks.append(
            HeadingBlock(
                ordinal=self._next_ordinal(),
                heading_path=self._heading_path(),
                source_locator=self._next_locator(),
                text=text,
                level=level,
            )
        )

    def _append_text(
        self,
        kind: Literal["paragraph", "list_item"],
        text: str,
    ) -> None:
        if text == "*참고사항":
            self._append_heading(3, text)
            return
        self.blocks.append(
            TextBlock(
                kind=kind,
                ordinal=self._next_ordinal(),
                heading_path=self._heading_path(),
                source_locator=self._next_locator(),
                text=text,
            )
        )

    def _heading_path(self) -> tuple[str, ...]:
        return tuple(text for _, text in self._headings)

    def _next_ordinal(self) -> int:
        return len(self.blocks) + 1

    def _next_locator(self) -> str:
        return f"{self._locator_prefix}/block:{self._next_ordinal()}"


def _list_item_text(element: HtmlElement) -> str:
    text = element_text(element)
    marker_element = find_first_by_class(element, "num")
    if marker_element is None:
        return text
    marker = element_text(marker_element)
    if not marker.isdecimal() or not text.startswith(marker):
        return text
    return normalize_source_text(f"{marker}. {text[len(marker) :]}")


def _has_html_class(element: HtmlElement, class_name: str) -> bool:
    classes = attribute(element, "class")
    return classes is not None and class_name in classes.split()


def _parse_k2web_content(
    repository_root: Path,
    source: ApprovedSource,
    snapshot: VerifiedSnapshot,
) -> ParsedDocument:
    tree = _load_k2web_tree(repository_root, source, snapshot)
    body = find_by_id(tree, "_contentBuilder")
    if body is None:
        raise ValueError("K2Web content builder is missing")
    blocks = _BlockCollector("html:#_contentBuilder").collect(body)
    return _document(source, snapshot, title=source.title, blocks=blocks)


def _parse_k2web_board(
    repository_root: Path,
    source: ApprovedSource,
    snapshot: VerifiedSnapshot,
) -> ParsedDocument:
    tree = _load_k2web_tree(repository_root, source, snapshot)
    title_element = find_first_by_class(tree, "view-title")
    if title_element is None:
        raise ValueError("K2Web board title is missing")
    title = element_text(title_element)
    if title != normalize_source_text(source.title):
        raise ValueError(
            f"board title mismatch for {source.document_id}: {title!r}"
        )
    body = find_first_by_class(tree, "view-con")
    if body is None:
        raise ValueError("K2Web board body is missing")

    modified = _date_from_class(tree, "modify")
    published = _date_from_class(tree, "write")
    observed_date = modified or published
    if (
        source.published_or_revised is not None
        and observed_date != source.published_or_revised
    ):
        raise ValueError(
            f"board date mismatch for {source.document_id}: {observed_date!r}"
        )
    blocks = _BlockCollector("html:.view-con").collect(body)
    return _document(
        source,
        snapshot,
        title=title,
        blocks=blocks,
        published_or_revised=source.published_or_revised or observed_date,
    )


def _load_k2web_tree(
    repository_root: Path,
    source: ApprovedSource,
    snapshot: VerifiedSnapshot,
) -> HtmlElement:
    relative = f"{source.snapshot_path}/source.html"
    if relative not in snapshot.files:
        raise ValueError(f"verified K2Web source file is missing: {relative}")
    try:
        html = _verified_snapshot_bytes(
            repository_root,
            snapshot,
            relative,
        ).decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"unable to decode K2Web source as UTF-8: {relative}") from error
    return parse_html(html)


def _date_from_class(tree: HtmlElement, class_name: str) -> str | None:
    element = find_first_by_class(tree, class_name)
    if element is None:
        return None
    match = _DATE_PATTERN.search(element_text(element))
    if match is None:
        return None
    return "-".join(match.groups())


def _parse_ebook(
    repository_root: Path,
    source: ApprovedSource,
    snapshot: VerifiedSnapshot,
) -> ParsedDocument:
    if source.page_count is None:
        raise ValueError("e-book source requires page_count")
    expected_pages = tuple(
        f"{source.snapshot_path}/page_{page_number:03d}.html"
        for page_number in range(1, source.page_count + 1)
    )
    actual_pages = tuple(
        relative
        for relative in snapshot.files
        if _EBOOK_PAGE_NAME_PATTERN.search(Path(relative).name)
    )
    missing = tuple(page for page in expected_pages if page not in actual_pages)
    if missing:
        raise ValueError(f"missing e-book page: {','.join(missing)}")
    if tuple(sorted(actual_pages)) != expected_pages:
        raise ValueError("e-book page count does not match approved source")

    blocks: list[ContentBlock] = []
    current_heading: tuple[str, ...] = ()
    for page_number, relative in enumerate(expected_pages, start=1):
        try:
            html = _verified_snapshot_bytes(
                repository_root,
                snapshot,
                relative,
            ).decode("euc-kr", errors="strict")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"unable to decode e-book page as EUC-KR: {relative}"
            ) from error
        tree = parse_html(html)
        text_bodies = tuple(
            element
            for element in iter_elements(tree, tag="div")
            if attribute(element, "id") == "text"
        )
        if len(text_bodies) != 1:
            raise ValueError(f"e-book page must contain one text body: {relative}")
        lines = element_lines(text_bodies[0])
        if not lines:
            raise ValueError(f"empty e-book text body: {relative}")
        before_page = len(blocks)
        for line_number, line in enumerate(lines, start=1):
            article_match = _ARTICLE_PATTERN.fullmatch(line)
            if article_match is None:
                blocks.append(
                    TextBlock(
                        kind="paragraph",
                        ordinal=len(blocks) + 1,
                        heading_path=current_heading,
                        page_number=page_number,
                        source_locator=f"page:{page_number}#line:{line_number}",
                        text=line,
                    )
                )
                continue
            heading = _normalize_article_heading(article_match.group(1))
            trailing = normalize_source_text(article_match.group(2))
            current_heading = (heading,)
            blocks.append(
                HeadingBlock(
                    ordinal=len(blocks) + 1,
                    heading_path=current_heading,
                    page_number=page_number,
                    source_locator=(
                        f"page:{page_number}#line:{line_number}#heading"
                    ),
                    text=heading,
                    level=2,
                )
            )
            if trailing:
                blocks.append(
                    TextBlock(
                        kind="paragraph",
                        ordinal=len(blocks) + 1,
                        heading_path=current_heading,
                        page_number=page_number,
                        source_locator=f"page:{page_number}#line:{line_number}#body",
                        text=trailing,
                    )
                )
        if len(blocks) == before_page:
            raise ValueError(f"e-book page produced no blocks: {relative}")

    normalized_blocks = _normalize_ebook_line_rules(
        tuple(blocks),
        source,
    )
    normalized_blocks = _apply_ebook_replacement_rules(
        normalized_blocks,
        source,
    )
    normalized_blocks = _rebuild_ebook_hierarchy(normalized_blocks)
    return _document(
        source,
        snapshot,
        title=source.title,
        blocks=normalized_blocks,
    )


def _normalize_ebook_line_rules(
    blocks: tuple[ContentBlock, ...],
    source: ApprovedSource,
) -> tuple[ContentBlock, ...]:
    if not source.ebook_line_rules:
        return blocks

    rules = {
        (rule.left_locator, rule.right_locator): rule
        for rule in source.ebook_line_rules
    }
    applied: set[tuple[str, str]] = set()
    working = list(blocks)

    for index in range(len(working) - 1):
        left = working[index]
        right = working[index + 1]
        boundary = (
            _locator_end(left.source_locator),
            _locator_start(right.source_locator),
        )
        rule = rules.get(boundary)
        if (
            rule is None
            or rule.action
            is not EbookLineNormalizationAction.MOVE_LEFT_SUFFIX_TO_RIGHT
        ):
            continue
        if not isinstance(left, TextBlock) or not isinstance(right, TextBlock):
            raise ValueError(f"line move requires text blocks: {boundary}")
        fragment = rule.fragment
        if fragment is None or not left.text.endswith(fragment):
            raise ValueError(f"line move fragment does not match: {boundary}")
        left_text = left.text[: -len(fragment)].rstrip()
        if not left_text:
            raise ValueError(f"line move empties left block: {boundary}")
        working[index] = left.model_copy(update={"text": left_text})
        working[index + 1] = right.model_copy(
            update={
                "source_locator": _locator_span(
                    left.source_locator,
                    right.source_locator,
                ),
                "text": fragment + right.text.lstrip(),
            }
        )
        applied.add(boundary)

    normalized: list[ContentBlock] = []
    index = 0
    while index < len(working):
        current = working[index]
        while index + 1 < len(working):
            right = working[index + 1]
            boundary = (
                _locator_end(current.source_locator),
                _locator_start(right.source_locator),
            )
            rule = rules.get(boundary)
            if rule is None or rule.action is (
                EbookLineNormalizationAction.MOVE_LEFT_SUFFIX_TO_RIGHT
            ):
                break
            if not isinstance(current, TextBlock) or not isinstance(
                right, TextBlock
            ):
                raise ValueError(f"line join requires text blocks: {boundary}")
            if (
                current.page_number != right.page_number
                or current.heading_path != right.heading_path
            ):
                raise ValueError(
                    f"line join requires matching page and heading: {boundary}"
                )
            separator = (
                ""
                if rule.action
                is EbookLineNormalizationAction.JOIN_WITHOUT_SPACE
                else " "
            )
            current = TextBlock(
                kind=current.kind,
                ordinal=current.ordinal,
                heading_path=current.heading_path,
                page_number=current.page_number,
                source_locator=_locator_span(
                    current.source_locator,
                    right.source_locator,
                ),
                text=current.text + separator + right.text,
            )
            applied.add(boundary)
            index += 1
        normalized.append(current)
        index += 1

    missing = tuple(sorted(set(rules) - applied))
    if missing:
        raise ValueError(f"e-book line rule did not match source: {missing[0]}")

    return tuple(
        block.model_copy(update={"ordinal": ordinal})
        for ordinal, block in enumerate(normalized, start=1)
    )


def _apply_ebook_replacement_rules(
    blocks: tuple[ContentBlock, ...],
    source: ApprovedSource,
) -> tuple[ContentBlock, ...]:
    if not source.ebook_replacement_rules:
        return blocks

    working = list(blocks)
    applied: set[tuple[str, str]] = set()
    for rule in source.ebook_replacement_rules:
        replacement_range = (rule.start_locator, rule.end_locator)
        exact_start_matches = [
            index
            for index, block in enumerate(working)
            if block.source_locator == rule.start_locator
        ]
        start_matches = exact_start_matches or [
            index
            for index, block in enumerate(working)
            if _locator_start(block.source_locator) == rule.start_locator
        ]
        if len(start_matches) != 1:
            raise ValueError(
                f"e-book replacement range did not match source: {replacement_range}"
            )
        start_index = start_matches[0]
        end_matches = [
            index
            for index in range(start_index, len(working))
            if _locator_end(working[index].source_locator) == rule.end_locator
        ]
        if len(end_matches) != 1:
            raise ValueError(
                f"e-book replacement range did not match source: {replacement_range}"
            )
        end_index = end_matches[0]
        selected = tuple(working[start_index : end_index + 1])
        if any(isinstance(block, TableBlock) for block in selected):
            raise ValueError(
                f"e-book replacement cannot consume a table: {replacement_range}"
            )
        page_numbers = {block.page_number for block in selected}
        if len(page_numbers) != 1 and isinstance(
            rule, EbookTableReplacementRule
        ):
            raise ValueError(
                "e-book table replacement must stay on one page: "
                f"{replacement_range}"
            )
        observed_text = "\n".join(
            block.text
            for block in selected
            if isinstance(block, HeadingBlock | TextBlock)
        )
        if observed_text != rule.expected_text:
            raise ValueError(
                f"e-book replacement text does not match source: {replacement_range}"
            )

        source_locator = (
            selected[0].source_locator
            if len(selected) == 1
            else _locator_span(
                selected[0].source_locator,
                selected[-1].source_locator,
            )
        )
        page_number = selected[0].page_number
        if isinstance(rule, EbookTextReplacementRule):
            if rule.output_kind == "heading":
                replacement: ContentBlock = HeadingBlock(
                    ordinal=selected[0].ordinal,
                    heading_path=(rule.replacement_text,),
                    page_number=page_number,
                    source_locator=source_locator,
                    text=rule.replacement_text,
                    level=2,
                )
            else:
                replacement = TextBlock(
                    kind="paragraph",
                    ordinal=selected[0].ordinal,
                    heading_path=selected[0].heading_path,
                    page_number=page_number,
                    source_locator=source_locator,
                    text=rule.replacement_text,
                )
        elif isinstance(rule, EbookTableReplacementRule):
            replacement = TableBlock(
                ordinal=selected[0].ordinal,
                heading_path=selected[0].heading_path,
                page_number=page_number,
                source_locator=source_locator,
                table=_replacement_table(rule),
            )
        else:
            raise TypeError(f"unsupported e-book replacement rule: {type(rule)}")

        working[start_index : end_index + 1] = [replacement]
        applied.add(replacement_range)

    expected = {
        (rule.start_locator, rule.end_locator)
        for rule in source.ebook_replacement_rules
    }
    missing = tuple(sorted(expected - applied))
    if missing:
        raise ValueError(
            f"e-book replacement range did not match source: {missing[0]}"
        )
    return tuple(working)


def _replacement_table(rule: EbookTableReplacementRule) -> ParsedTable:
    source_rows = (
        SourceTableRow(
            cells=tuple(
                SourceTableCell(text=header, is_header=True)
                for header in rule.headers
            )
        ),
        *(
            SourceTableRow(
                cells=tuple(
                    SourceTableCell(text=cell, is_header=False)
                    for cell in row
                )
            )
            for row in rule.rows
        ),
    )
    return ParsedTable(
        caption=rule.caption,
        source_rows=source_rows,
        logical_headers=rule.headers,
        logical_rows=rule.rows,
    )


def _rebuild_ebook_hierarchy(
    blocks: tuple[ContentBlock, ...],
) -> tuple[ContentBlock, ...]:
    rebuilt: list[ContentBlock] = []
    current_heading: tuple[str, ...] = ()
    for ordinal, block in enumerate(blocks, start=1):
        if isinstance(block, HeadingBlock):
            current_heading = (block.text,)
        rebuilt.append(
            block.model_copy(
                update={
                    "ordinal": ordinal,
                    "heading_path": current_heading,
                }
            )
        )
    return tuple(rebuilt)


def _normalize_article_heading(value: str) -> str:
    heading = normalize_source_text(value)
    heading = re.sub(r"^제\s*([0-9]+)\s*조", r"제\1조", heading)
    return re.sub(
        r"^(제[0-9]+조)\s*의\s*([0-9]+)",
        r"\1의\2",
        heading,
    )


def _locator_start(locator: str) -> str:
    return locator.split("..", maxsplit=1)[0]


def _locator_end(locator: str) -> str:
    return locator.rsplit("..", maxsplit=1)[-1]


def _locator_span(left: str, right: str) -> str:
    return f"{_locator_start(left)}..{_locator_end(right)}"


def _document(
    source: ApprovedSource,
    snapshot: VerifiedSnapshot,
    *,
    title: str,
    blocks: tuple[ContentBlock, ...],
    published_or_revised: str | None = None,
) -> ParsedDocument:
    return ParsedDocument(
        document_id=source.document_id,
        candidate_id=source.candidate_id,
        title=title,
        source_url=source.source_url,
        authority=source.authority,
        source_type=source.source_type,
        language=source.language,
        published_or_revised=(
            source.published_or_revised
            if published_or_revised is None
            else published_or_revised
        ),
        snapshot_commitment=snapshot.commitment,
        snapshot_files=snapshot.files,
        blocks=blocks,
        review_rules=source.review_rules,
        review_required=bool(source.review_rules),
    )


def _resolve_source_path(repository_root: Path, relative: str) -> Path:
    root = repository_root.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("source path resolves outside repository root") from error
    return resolved


def _verified_snapshot_bytes(
    repository_root: Path,
    snapshot: VerifiedSnapshot,
    relative: str,
) -> bytes:
    if relative not in snapshot.files:
        raise ValueError(f"verified snapshot file is missing: {relative}")
    path = _resolve_source_path(repository_root, relative)
    try:
        contents = path.read_bytes()
    except OSError as error:
        raise ValueError(f"unable to read verified snapshot file: {relative}") from error
    if hashlib.sha256(contents).hexdigest() != snapshot.file_hashes[relative]:
        raise ValueError(f"snapshot hash mismatch: {relative}")
    return contents
