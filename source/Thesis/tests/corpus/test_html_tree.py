from thesis_rag.corpus.html_tree import (
    HtmlElement,
    attribute,
    direct_child_elements,
    element_lines,
    element_text,
    find_by_id,
    find_first_by_class,
    find_first_tag,
    iter_elements,
    normalize_source_text,
    parse_html,
)


def test_tree_selects_exact_source_body_and_decodes_entities() -> None:
    tree = parse_html(
        '<nav id="menu">삭제할 메뉴</nav>'
        '<article id="_contentBuilder"><p>졸업&nbsp;요건</p></article>'
    )
    body = find_by_id(tree, "_contentBuilder")

    assert body is not None
    assert element_text(body) == "졸업 요건"
    assert "삭제할 메뉴" not in element_text(body)


def test_line_extraction_preserves_br_boundaries() -> None:
    tree = parse_html(
        '<div id="text">제1조 목적<br>다음 줄<br><br>제2조 범위</div>'
    )
    body = find_by_id(tree, "text")

    assert body is not None
    assert element_lines(body) == ("제1조 목적", "다음 줄", "제2조 범위")


def test_normalization_preserves_words_but_cleans_layout_characters() -> None:
    assert normalize_source_text("  종합\u00a0설계\u200b\r\n 수강  ") == "종합 설계 수강"
    assert normalize_source_text("Pass/Fail과 SW·AI") == "Pass/Fail과 SW·AI"


def test_class_matching_uses_exact_tokens() -> None:
    tree = parse_html(
        '<div class="view-container">wrong</div>'
        '<div class="wide view-con selected">right</div>'
    )

    match = find_first_by_class(tree, "view-con")

    assert match is not None
    assert element_text(match) == "right"


def test_tree_ignores_noncontent_elements_during_text_extraction() -> None:
    tree = parse_html(
        "<article id=body>"
        "<style>.secret { display:block }</style>"
        "<script>alert('secret')</script>"
        "<template>template secret</template>"
        "<noscript>fallback secret</noscript>"
        "<p>보이는 본문</p>"
        "</article>"
    )
    body = find_by_id(tree, "body")

    assert body is not None
    assert element_text(body) == "보이는 본문"


def test_parser_recovers_from_unmatched_nested_end_tags_deterministically() -> None:
    html = "<div id=target><p>첫째 <b>둘째</div><p>바깥</p>"

    first = parse_html(html)
    second = parse_html(html)
    target = find_by_id(first, "target")

    assert first == second
    assert target is not None
    assert element_text(target) == "첫째 둘째"
    assert element_text(first) == "첫째 둘째 바깥"


def test_element_iteration_and_direct_children_preserve_source_order() -> None:
    tree = parse_html(
        "<section id=root>"
        "<h2 data-k=z>제목</h2>"
        "<div><p>중첩</p></div>"
        "<p>직접</p>"
        "</section>"
    )
    root = find_by_id(tree, "root")

    assert root is not None
    assert [element.tag for element in direct_child_elements(root)] == [
        "h2",
        "div",
        "p",
    ]
    assert [element_text(element) for element in iter_elements(root, tag="p")] == [
        "중첩",
        "직접",
    ]
    heading = find_first_tag(root, "h2")
    assert heading is not None
    assert attribute(heading, "data-k") == "z"


def test_void_elements_do_not_capture_following_siblings() -> None:
    tree = parse_html("<div id=root>앞<img src=x>뒤<input value=y><br>끝</div>")
    root = find_by_id(tree, "root")

    assert root is not None
    assert element_text(root) == "앞 뒤 끝"
    assert element_lines(root) == ("앞 뒤", "끝")


def test_duplicate_attributes_have_deterministic_last_value() -> None:
    tree = parse_html('<div id="first" id="second">본문</div>')

    assert find_by_id(tree, "first") is None
    second = find_by_id(tree, "second")
    assert isinstance(second, HtmlElement)
    assert element_text(second) == "본문"
