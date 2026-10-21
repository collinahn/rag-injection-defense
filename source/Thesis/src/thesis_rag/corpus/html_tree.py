import re
import unicodedata
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import TypeAlias

_VOID_TAGS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)
_IGNORED_TEXT_TAGS = frozenset({"script", "style", "template", "noscript"})
_BLOCK_TAGS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "caption",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
    }
)
_SPACING_VOID_TAGS = frozenset({"area", "embed", "hr", "img", "input", "param"})
_LAYOUT_CHARACTER_PATTERN = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")
_WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class HtmlText:
    value: str


@dataclass(frozen=True, slots=True)
class HtmlElement:
    tag: str
    attrs: tuple[tuple[str, str], ...]
    children: tuple["HtmlNode", ...]


HtmlNode: TypeAlias = HtmlText | HtmlElement


@dataclass(slots=True)
class _MutableElement:
    tag: str
    attrs: dict[str, str]
    children: list["_MutableNode"] = field(default_factory=list)


_MutableNode: TypeAlias = str | _MutableElement


class _TreeBuilder(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _MutableElement(tag="[document]", attrs={})
        self._stack = [self.root]

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        normalized_attrs: dict[str, str] = {}
        for name, value in attrs:
            normalized_attrs[name.casefold()] = value or ""
        element = _MutableElement(tag=tag.casefold(), attrs=normalized_attrs)
        self._stack[-1].children.append(element)
        if element.tag not in _VOID_TAGS:
            self._stack.append(element)

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        normalized_attrs: dict[str, str] = {}
        for name, value in attrs:
            normalized_attrs[name.casefold()] = value or ""
        self._stack[-1].children.append(
            _MutableElement(tag=tag.casefold(), attrs=normalized_attrs)
        )

    def handle_endtag(self, tag: str) -> None:
        normalized_tag = tag.casefold()
        for index in range(len(self._stack) - 1, 0, -1):
            if self._stack[index].tag == normalized_tag:
                del self._stack[index:]
                return

    def handle_data(self, data: str) -> None:
        if data:
            self._stack[-1].children.append(data)


def parse_html(value: str) -> HtmlElement:
    parser = _TreeBuilder()
    parser.feed(value)
    parser.close()
    return _freeze_element(parser.root)


def attribute(element: HtmlElement, name: str) -> str | None:
    normalized_name = name.casefold()
    for key, value in element.attrs:
        if key == normalized_name:
            return value
    return None


def direct_child_elements(
    element: HtmlElement,
    *,
    tag: str | None = None,
) -> tuple[HtmlElement, ...]:
    normalized_tag = tag.casefold() if tag is not None else None
    return tuple(
        child
        for child in element.children
        if isinstance(child, HtmlElement)
        and (normalized_tag is None or child.tag == normalized_tag)
    )


def iter_elements(
    element: HtmlElement,
    *,
    tag: str | None = None,
) -> tuple[HtmlElement, ...]:
    normalized_tag = tag.casefold() if tag is not None else None
    matches: list[HtmlElement] = []

    def visit(current: HtmlElement) -> None:
        if normalized_tag is None or current.tag == normalized_tag:
            matches.append(current)
        for child in current.children:
            if isinstance(child, HtmlElement):
                visit(child)

    visit(element)
    return tuple(matches)


def find_by_id(element: HtmlElement, value: str) -> HtmlElement | None:
    for candidate in iter_elements(element):
        if attribute(candidate, "id") == value:
            return candidate
    return None


def find_first_by_class(element: HtmlElement, value: str) -> HtmlElement | None:
    for candidate in iter_elements(element):
        classes = attribute(candidate, "class")
        if classes is not None and value in classes.split():
            return candidate
    return None


def find_first_tag(element: HtmlElement, tag: str) -> HtmlElement | None:
    normalized_tag = tag.casefold()
    for candidate in iter_elements(element):
        if candidate.tag == normalized_tag:
            return candidate
    return None


def normalize_source_text(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value)
    normalized = normalized.replace("\u00a0", " ")
    normalized = _LAYOUT_CHARACTER_PATTERN.sub("", normalized)
    return _WHITESPACE_PATTERN.sub(" ", normalized).strip()


def element_text(element: HtmlElement) -> str:
    parts: list[str] = []
    _append_visible_text(element, parts)
    return normalize_source_text("".join(parts))


def element_lines(element: HtmlElement) -> tuple[str, ...]:
    parts: list[str] = []
    _append_visible_text(element, parts)
    lines = []
    for line in "".join(parts).splitlines():
        normalized = normalize_source_text(line)
        if normalized:
            lines.append(normalized)
    return tuple(lines)


def _freeze_element(element: _MutableElement) -> HtmlElement:
    children: list[HtmlNode] = []
    for child in element.children:
        if isinstance(child, str):
            children.append(HtmlText(value=child))
        else:
            children.append(_freeze_element(child))
    return HtmlElement(
        tag=element.tag,
        attrs=tuple(sorted(element.attrs.items())),
        children=tuple(children),
    )


def _append_visible_text(node: HtmlNode, parts: list[str]) -> None:
    if isinstance(node, HtmlText):
        parts.append(node.value)
        return
    if node.tag in _IGNORED_TEXT_TAGS:
        return
    if node.tag == "br":
        parts.append("\n")
        return
    if node.tag in _SPACING_VOID_TAGS:
        parts.append(" ")
        return

    is_block = node.tag in _BLOCK_TAGS
    if is_block:
        parts.append("\n")
    for child in node.children:
        _append_visible_text(child, parts)
    if is_block:
        parts.append("\n")
