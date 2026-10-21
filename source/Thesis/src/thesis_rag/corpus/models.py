from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import Field, StrictBool, field_validator, model_validator

from thesis_rag.domain import FrozenRecord, Language, Sha256Hex


def _validate_repository_path(value: str) -> str:
    if not value.strip() or "\\" in value:
        raise ValueError("path must be a relative repository path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
        raise ValueError("path must be a relative repository path")
    return path.as_posix()


class CorpusSourceType(StrEnum):
    K2WEB_CONTENT = "k2web_content"
    K2WEB_BOARD = "k2web_board"
    EBOOK_HTML_COLLECTION = "ebook_html_collection"


class EbookLineNormalizationAction(StrEnum):
    JOIN_WITHOUT_SPACE = "join_without_space"
    JOIN_WITH_SPACE = "join_with_space"
    MOVE_LEFT_SUFFIX_TO_RIGHT = "move_left_suffix_to_right"


class EbookLineRule(FrozenRecord):
    left_locator: str = Field(min_length=1)
    right_locator: str = Field(min_length=1)
    action: EbookLineNormalizationAction
    fragment: str | None = None

    @field_validator("left_locator", "right_locator")
    @classmethod
    def strip_locator(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("line locator must be nonblank")
        return stripped

    @field_validator("fragment")
    @classmethod
    def strip_fragment(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("fragment must be nonblank")
        return stripped

    @model_validator(mode="after")
    def validate_fragment_for_action(self) -> Self:
        moves_fragment = (
            self.action
            is EbookLineNormalizationAction.MOVE_LEFT_SUFFIX_TO_RIGHT
        )
        if moves_fragment != (self.fragment is not None):
            raise ValueError(
                "fragment is required only for move_left_suffix_to_right"
            )
        return self


class EbookReplacementRuleBase(FrozenRecord):
    start_locator: str = Field(min_length=1)
    end_locator: str = Field(min_length=1)
    expected_text: str = Field(min_length=1)

    @field_validator("start_locator", "end_locator")
    @classmethod
    def strip_locator(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("replacement locator must be nonblank")
        return stripped

    @field_validator("expected_text")
    @classmethod
    def require_nonblank_expected_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("expected_text must be nonblank")
        return value


class EbookTextReplacementRule(EbookReplacementRuleBase):
    kind: Literal["text"] = "text"
    replacement_text: str = Field(min_length=1)
    output_kind: Literal["paragraph", "heading"] = "paragraph"

    @field_validator("replacement_text")
    @classmethod
    def strip_replacement_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("replacement_text must be nonblank")
        return stripped


class EbookTableReplacementRule(EbookReplacementRuleBase):
    kind: Literal["table"] = "table"
    caption: str = Field(min_length=1)
    headers: tuple[str, ...] = Field(min_length=1)
    rows: tuple[tuple[str, ...], ...] = Field(min_length=1)

    @field_validator("caption")
    @classmethod
    def strip_caption(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("caption must be nonblank")
        return stripped

    @field_validator("headers")
    @classmethod
    def validate_headers(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(header.strip() for header in value)
        if any(not header for header in normalized):
            raise ValueError("headers must contain nonblank values")
        return normalized

    @field_validator("rows")
    @classmethod
    def validate_rows(
        cls,
        value: tuple[tuple[str, ...], ...],
    ) -> tuple[tuple[str, ...], ...]:
        normalized = tuple(
            tuple(cell.strip() for cell in row)
            for row in value
        )
        if any(not row or any(not cell for cell in row) for row in normalized):
            raise ValueError("rows must contain nonblank cells")
        return normalized

    @model_validator(mode="after")
    def validate_row_width(self) -> Self:
        if any(len(row) != len(self.headers) for row in self.rows):
            raise ValueError("table rows must match header width")
        return self


EbookReplacementRule = Annotated[
    EbookTextReplacementRule | EbookTableReplacementRule,
    Field(discriminator="kind"),
]


class ReviewRule(FrozenRecord):
    code: str = Field(pattern=r"^[A-Z][A-Z0-9_]+$")
    all_terms: tuple[str, ...] = Field(min_length=1)
    note: str = Field(min_length=1)

    @field_validator("all_terms")
    @classmethod
    def validate_terms(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(term.strip() for term in value)
        if any(not term for term in normalized) or len(set(normalized)) != len(normalized):
            raise ValueError("all_terms must contain unique nonblank terms")
        return normalized

    @field_validator("note")
    @classmethod
    def strip_note(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("note must be nonblank")
        return stripped


class ApprovedSource(FrozenRecord):
    document_id: str = Field(pattern=r"^DOC_[0-9]{3}$")
    candidate_id: str = Field(pattern=r"^CAND_[0-9]{3}$")
    title: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    authority: str = Field(min_length=1)
    source_type: CorpusSourceType
    snapshot_path: str = Field(min_length=1)
    language: Language
    page_count: int | None = Field(default=None, strict=True, ge=1)
    published_or_revised: str | None = None
    review_rules: tuple[ReviewRule, ...] = ()
    ebook_line_rules: tuple[EbookLineRule, ...] = ()
    ebook_replacement_rules: tuple[EbookReplacementRule, ...] = ()

    @field_validator("snapshot_path")
    @classmethod
    def validate_snapshot_path(cls, value: str) -> str:
        return _validate_repository_path(value)

    @field_validator("title", "authority")
    @classmethod
    def strip_nonblank_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must be nonblank")
        return stripped

    @model_validator(mode="after")
    def validate_ebook_options(self) -> Self:
        if (
            self.source_type is CorpusSourceType.EBOOK_HTML_COLLECTION
            and self.page_count is None
        ):
            raise ValueError("e-book source requires page_count")
        if (
            self.source_type is not CorpusSourceType.EBOOK_HTML_COLLECTION
            and self.page_count is not None
        ):
            raise ValueError("page_count is only valid for e-book sources")
        if (
            self.source_type is not CorpusSourceType.EBOOK_HTML_COLLECTION
            and self.ebook_line_rules
        ):
            raise ValueError(
                "ebook_line_rules are only valid for e-book sources"
            )
        if (
            self.source_type is not CorpusSourceType.EBOOK_HTML_COLLECTION
            and self.ebook_replacement_rules
        ):
            raise ValueError(
                "ebook_replacement_rules are only valid for e-book sources"
            )
        boundaries = tuple(
            (rule.left_locator, rule.right_locator)
            for rule in self.ebook_line_rules
        )
        if len(set(boundaries)) != len(boundaries):
            raise ValueError("ebook_line_rules must use unique boundaries")
        replacement_ranges = tuple(
            (rule.start_locator, rule.end_locator)
            for rule in self.ebook_replacement_rules
        )
        if len(set(replacement_ranges)) != len(replacement_ranges):
            raise ValueError("ebook_replacement_rules must use unique ranges")
        return self


class ApprovedCorpus(FrozenRecord):
    schema_version: Literal[1]
    collection_status: Literal["approved"]
    approved_at: str = Field(min_length=1)
    corpus_id: str = Field(min_length=1)
    allowed_domains: tuple[str, ...] = Field(min_length=1)
    hash_manifest: str = Field(min_length=1)
    sources: tuple[ApprovedSource, ...]

    @field_validator("hash_manifest")
    @classmethod
    def validate_hash_manifest(cls, value: str) -> str:
        return _validate_repository_path(value)

    @field_validator("allowed_domains")
    @classmethod
    def validate_allowed_domains(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(domain.strip().casefold() for domain in value)
        if any(
            not domain
            or "/" in domain
            or ":" in domain
            or domain.startswith(".")
            for domain in normalized
        ):
            raise ValueError("allowed_domains must contain hostnames")
        if len(set(normalized)) != len(normalized):
            raise ValueError("allowed_domains must be unique")
        return normalized

    @model_validator(mode="after")
    def validate_unique_source_ids(self) -> Self:
        document_ids: set[str] = set()
        candidate_ids: set[str] = set()
        for source in self.sources:
            if source.document_id in document_ids:
                raise ValueError(f"duplicate document_id: {source.document_id}")
            if source.candidate_id in candidate_ids:
                raise ValueError(f"duplicate candidate_id: {source.candidate_id}")
            document_ids.add(source.document_id)
            candidate_ids.add(source.candidate_id)
        return self


class VerifiedSnapshot(FrozenRecord):
    document_id: str = Field(pattern=r"^DOC_[0-9]{3}$")
    root_path: str = Field(min_length=1)
    files: tuple[str, ...] = Field(min_length=1)
    file_hashes: dict[str, Sha256Hex]
    commitment: Sha256Hex

    @field_validator("root_path")
    @classmethod
    def validate_root_path(cls, value: str) -> str:
        return _validate_repository_path(value)

    @field_validator("files")
    @classmethod
    def validate_files(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(_validate_repository_path(item) for item in value)
        if normalized != tuple(sorted(normalized)):
            raise ValueError("snapshot files must use sorted path order")
        if len(set(normalized)) != len(normalized):
            raise ValueError("snapshot files must be unique")
        return normalized

    @model_validator(mode="after")
    def validate_file_hash_keys(self) -> Self:
        if set(self.files) != set(self.file_hashes):
            raise ValueError("snapshot files must match file_hashes keys")
        return self


class SourceTableCell(FrozenRecord):
    text: str
    is_header: StrictBool
    rowspan: int = Field(default=1, strict=True, ge=1)
    colspan: int = Field(default=1, strict=True, ge=1)


class SourceTableRow(FrozenRecord):
    cells: tuple[SourceTableCell, ...] = Field(min_length=1)


class ParsedTable(FrozenRecord):
    caption: str | None
    source_rows: tuple[SourceTableRow, ...] = Field(min_length=1)
    logical_headers: tuple[str, ...] = Field(min_length=1)
    logical_rows: tuple[tuple[str, ...], ...] = Field(min_length=1)

    @field_validator("caption")
    @classmethod
    def normalize_caption(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @model_validator(mode="after")
    def validate_logical_width(self) -> Self:
        width = len(self.logical_headers)
        if any(len(row) != width for row in self.logical_rows):
            raise ValueError("logical table rows must match header width")
        return self


class BlockBase(FrozenRecord):
    ordinal: int = Field(strict=True, ge=1)
    heading_path: tuple[str, ...] = ()
    page_number: int | None = Field(default=None, strict=True, ge=1)
    source_locator: str = Field(min_length=1)


class HeadingBlock(BlockBase):
    kind: Literal["heading"] = "heading"
    text: str = Field(min_length=1)
    level: int = Field(strict=True, ge=1, le=6)


class TextBlock(BlockBase):
    kind: Literal["paragraph", "list_item"]
    text: str = Field(min_length=1)


class TableBlock(BlockBase):
    kind: Literal["table"] = "table"
    table: ParsedTable


ContentBlock = Annotated[
    HeadingBlock | TextBlock | TableBlock,
    Field(discriminator="kind"),
]


class ParsedDocument(FrozenRecord):
    document_id: str = Field(pattern=r"^DOC_[0-9]{3}$")
    candidate_id: str = Field(pattern=r"^CAND_[0-9]{3}$")
    title: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    authority: str = Field(min_length=1)
    source_type: CorpusSourceType
    language: Language
    published_or_revised: str | None
    snapshot_commitment: Sha256Hex
    snapshot_files: tuple[str, ...] = Field(min_length=1)
    blocks: tuple[ContentBlock, ...] = Field(min_length=1)
    review_rules: tuple[ReviewRule, ...] = ()
    review_required: StrictBool

    @model_validator(mode="after")
    def validate_blocks_and_review_state(self) -> Self:
        expected_ordinals = tuple(range(1, len(self.blocks) + 1))
        if tuple(block.ordinal for block in self.blocks) != expected_ordinals:
            raise ValueError("parsed block ordinals must be consecutive from one")
        locators = tuple(block.source_locator for block in self.blocks)
        if len(set(locators)) != len(locators):
            raise ValueError("parsed block source locators must be unique")
        if self.review_required != bool(self.review_rules):
            raise ValueError("review_required must match presence of review_rules")
        if self.source_type is CorpusSourceType.EBOOK_HTML_COLLECTION:
            if any(block.page_number is None for block in self.blocks):
                raise ValueError("e-book blocks require page numbers")
        elif any(block.page_number is not None for block in self.blocks):
            raise ValueError("non-e-book blocks must not have page numbers")
        return self


class ChunkingConfig(FrozenRecord):
    target_chars: int = Field(default=650, strict=True, ge=1)
    min_chars: int = Field(default=250, strict=True, ge=1)
    max_chars: int = Field(default=900, strict=True, ge=1)
    overlap_chars: int = Field(default=100, strict=True, ge=0)

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if not self.min_chars <= self.target_chars <= self.max_chars:
            raise ValueError("chunking bounds must satisfy min <= target <= max")
        if self.overlap_chars >= self.max_chars:
            raise ValueError("chunking bounds require overlap < max")
        return self


class CorpusValidationReport(FrozenRecord):
    corpus_id: str = Field(min_length=1)
    document_count: int = Field(strict=True, ge=0)
    block_count: int = Field(strict=True, ge=0)
    table_count: int = Field(strict=True, ge=0)
    chunk_count: int = Field(strict=True, ge=0)
    character_count: int = Field(strict=True, ge=0)
    review_document_count: int = Field(strict=True, ge=0)
    review_chunk_count: int = Field(strict=True, ge=0)
    oversize_chunk_count: int = Field(strict=True, ge=0)
    document_commitments: dict[str, Sha256Hex]
    output_hashes: dict[str, Sha256Hex]
