import math

import pytest

from thesis_rag.canonical import canonical_json, sha256_text, stable_id
from thesis_rag.domain import Chunk


def test_canonical_json_is_key_order_independent() -> None:
    left = {"question": "졸업 요건", "rank": 2}
    right = {"rank": 2, "question": "졸업 요건"}

    assert canonical_json(left) == canonical_json(right)
    assert stable_id("JOB", left) == stable_id("JOB", right)


def test_canonical_json_normalizes_unicode_to_nfc() -> None:
    assert canonical_json({"text": "가"}) == canonical_json({"text": "가"})


def test_canonical_json_normalizes_nested_dictionary_keys_and_values() -> None:
    composed = {"outer": [{"가": "값"}]}
    decomposed = {"o\u0075ter": [{"가": "값"}]}

    assert canonical_json(composed) == canonical_json(decomposed)


@pytest.mark.parametrize(
    "value",
    [
        {"é": 1, "e\u0301": 2},
        {"e\u0301": 2, "é": 1},
    ],
)
def test_canonical_json_rejects_nfc_normalized_key_collisions(value: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="NFC-normalized key collision"):
        canonical_json(value)


def test_canonical_json_serializes_base_model_deterministically() -> None:
    first = Chunk(
        chunk_id="chunk-1",
        document_id="document-1",
        text="가",
        language="ko",
        metadata={"z": "마지막", "a": "첫째"},
    )
    second = Chunk(
        document_id="document-1",
        chunk_id="chunk-1",
        text="가",
        language="ko",
        metadata={"a": "첫째", "z": "마지막"},
    )

    assert canonical_json(first) == canonical_json(second)
    assert stable_id("CHUNK", first) == stable_id("CHUNK", second)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_canonical_json_rejects_non_finite_floats(value: float) -> None:
    with pytest.raises(ValueError):
        canonical_json({"value": value})


def test_sha256_text_uses_utf8_bytes() -> None:
    assert sha256_text("abc") == (
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad"
    )


def test_stable_id_uses_prefix_and_first_24_digest_characters() -> None:
    value = {"question": "졸업 요건", "rank": 2}

    assert stable_id("JOB", value) == "JOB_" + sha256_text(canonical_json(value))[:24]
