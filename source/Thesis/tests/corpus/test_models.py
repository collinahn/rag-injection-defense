from pathlib import Path

import pytest
from pydantic import ValidationError

from thesis_rag.corpus.models import (
    ApprovedSource,
    CorpusSourceType,
    ReviewRule,
)
from thesis_rag.domain import Language


def _source(**overrides: object) -> ApprovedSource:
    values: dict[str, object] = {
        "document_id": "DOC_001",
        "candidate_id": "CAND_001",
        "title": "융복합소프트웨어전공 이수요건",
        "source_url": "https://sw.hufs.ac.kr/sw/14991/subview.do",
        "authority": "융복합소프트웨어전공 공식 홈페이지",
        "source_type": CorpusSourceType.K2WEB_CONTENT,
        "snapshot_path": "data/candidates/raw/CAND_001",
        "language": Language.KO,
    }
    values.update(overrides)
    return ApprovedSource.model_validate(values)


def test_corpus_source_type_values_are_explicit() -> None:
    assert CorpusSourceType.K2WEB_CONTENT.value == "k2web_content"
    assert CorpusSourceType.K2WEB_BOARD.value == "k2web_board"
    assert CorpusSourceType.EBOOK_HTML_COLLECTION.value == "ebook_html_collection"


def test_approved_source_rejects_parent_path() -> None:
    with pytest.raises(ValidationError, match="relative repository path"):
        _source(snapshot_path="../outside")


def test_approved_source_rejects_absolute_path() -> None:
    with pytest.raises(ValidationError, match="relative repository path"):
        _source(snapshot_path=str(Path("/tmp/raw")))


def test_ebook_source_requires_page_count() -> None:
    with pytest.raises(ValidationError, match="page_count"):
        _source(source_type=CorpusSourceType.EBOOK_HTML_COLLECTION)


def test_non_ebook_source_rejects_page_count() -> None:
    with pytest.raises(ValidationError, match="page_count"):
        _source(page_count=3)


def test_ebook_source_accepts_explicit_line_normalization_rules() -> None:
    try:
        source = _source(
            source_type=CorpusSourceType.EBOOK_HTML_COLLECTION,
            page_count=1,
            ebook_line_rules=(
                {
                    "left_locator": "page:1#line:1",
                    "right_locator": "page:1#line:2",
                    "action": "join_without_space",
                },
                {
                    "left_locator": "page:1#line:2",
                    "right_locator": "page:2#line:1",
                    "action": "move_left_suffix_to_right",
                    "fragment": "입",
                },
            ),
        )
    except ValidationError as error:
        pytest.fail(f"e-book line normalization rules must be accepted: {error}")

    assert [rule.action.value for rule in source.ebook_line_rules] == [
        "join_without_space",
        "move_left_suffix_to_right",
    ]


def test_line_normalization_rules_are_limited_to_ebook_sources() -> None:
    with pytest.raises(ValidationError, match="only valid for e-book"):
        _source(
            ebook_line_rules=(
                {
                    "left_locator": "page:1#line:1",
                    "right_locator": "page:1#line:2",
                    "action": "join_without_space",
                },
            )
        )


def test_line_normalization_rule_validates_fragment_by_action() -> None:
    with pytest.raises(ValidationError, match="fragment"):
        _source(
            source_type=CorpusSourceType.EBOOK_HTML_COLLECTION,
            page_count=1,
            ebook_line_rules=(
                {
                    "left_locator": "page:1#line:1",
                    "right_locator": "page:2#line:1",
                    "action": "move_left_suffix_to_right",
                },
            ),
        )
    with pytest.raises(ValidationError, match="fragment"):
        _source(
            source_type=CorpusSourceType.EBOOK_HTML_COLLECTION,
            page_count=1,
            ebook_line_rules=(
                {
                    "left_locator": "page:1#line:1",
                    "right_locator": "page:1#line:2",
                    "action": "join_without_space",
                    "fragment": "입",
                },
            ),
        )


def test_ebook_source_rejects_duplicate_line_rule_boundaries() -> None:
    rule = {
        "left_locator": "page:1#line:1",
        "right_locator": "page:1#line:2",
        "action": "join_without_space",
    }
    with pytest.raises(ValidationError, match="unique boundaries"):
        _source(
            source_type=CorpusSourceType.EBOOK_HTML_COLLECTION,
            page_count=1,
            ebook_line_rules=(rule, rule),
        )


def test_ebook_source_accepts_curated_text_and_table_replacements() -> None:
    source = _source(
        source_type=CorpusSourceType.EBOOK_HTML_COLLECTION,
        page_count=1,
        ebook_replacement_rules=(
            {
                "kind": "text",
                "start_locator": "page:1#line:1",
                "end_locator": "page:1#line:2",
                "expected_text": "시각\n순서",
                "replacement_text": "시각 순서",
                "output_kind": "heading",
            },
            {
                "kind": "table",
                "start_locator": "page:1#line:3",
                "end_locator": "page:1#line:8",
                "expected_text": "표\n텍스트",
                "caption": "제한학과",
                "headers": ("구분", "서울", "글로벌"),
                "rows": (("학과", "가", "나"),),
            },
        ),
    )

    assert [rule.kind for rule in source.ebook_replacement_rules] == [
        "text",
        "table",
    ]


def test_ebook_replacement_rules_are_limited_to_ebook_sources() -> None:
    with pytest.raises(ValidationError, match="only valid for e-book"):
        _source(
            ebook_replacement_rules=(
                {
                    "kind": "text",
                    "start_locator": "page:1#line:1",
                    "end_locator": "page:1#line:1",
                    "expected_text": "원문",
                    "replacement_text": "복원",
                },
            )
        )


def test_ebook_table_replacement_requires_matching_row_width() -> None:
    with pytest.raises(ValidationError, match="header width"):
        _source(
            source_type=CorpusSourceType.EBOOK_HTML_COLLECTION,
            page_count=1,
            ebook_replacement_rules=(
                {
                    "kind": "table",
                    "start_locator": "page:1#line:1",
                    "end_locator": "page:1#line:3",
                    "expected_text": "표",
                    "caption": "제한학과",
                    "headers": ("구분", "서울", "글로벌"),
                    "rows": (("학과", "가"),),
                },
            ),
        )


def test_ebook_source_rejects_duplicate_replacement_ranges() -> None:
    rule = {
        "kind": "text",
        "start_locator": "page:1#line:1",
        "end_locator": "page:1#line:2",
        "expected_text": "원문",
        "replacement_text": "복원",
    }
    with pytest.raises(ValidationError, match="unique ranges"):
        _source(
            source_type=CorpusSourceType.EBOOK_HTML_COLLECTION,
            page_count=1,
            ebook_replacement_rules=(rule, rule),
        )


def test_review_rule_rejects_duplicate_or_blank_terms() -> None:
    with pytest.raises(ValidationError, match="unique nonblank"):
        ReviewRule(
            code="CAPSTONE_THESIS_STAGE_TIMING_REVIEW",
            all_terms=("종합설계", " 종합설계 "),
            note="사람 검토 필요",
        )


def test_source_rejects_invalid_document_and_candidate_ids() -> None:
    with pytest.raises(ValidationError):
        _source(document_id="DOC_1")
    with pytest.raises(ValidationError):
        _source(candidate_id="CAND_1")
