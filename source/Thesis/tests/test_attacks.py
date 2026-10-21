import json
import re
from pathlib import Path
from urllib.parse import urlparse

import pytest

from thesis_rag.attacks import insert_payload
from thesis_rag.domain import (
    AttackCase,
    AttackGoal,
    AttackPosition,
    Chunk,
    Question,
    Split,
)


def _attack(
    *, position: AttackPosition = AttackPosition.MIDDLE, payload: str = "PAYLOAD_SYN"
) -> AttackCase:
    return AttackCase(
        attack_id="ATK_SYN_001",
        split=Split.DEV,
        base_context_id="CTX_SYN_001",
        goal=AttackGoal.HIJACK,
        language="en",
        form="plain",
        position=position,
        payload=payload,
        success_markers=("ATTACK_SUCCESS_ATK_SYN_001",),
        template_id="TPL_SYN_G1_EN_PLAIN_V1",
        safety_class="sentinel_only",
    )


@pytest.fixture
def gold_chunk() -> Chunk:
    return Chunk(
        chunk_id="CH_SYN_GOLD",
        document_id="DOC_SYN_GOLD",
        text="Synthetic original evidence.",
        language="en",
        metadata={
            "origin": "synthetic",
            "is_malicious": False,
            "attack_id": "ATK_OLD",
            "clean_chunk_id": "CH_OLD",
        },
    )


@pytest.fixture
def middle_attack() -> AttackCase:
    return _attack()


def test_middle_insertion_keeps_original_text(middle_attack: AttackCase, gold_chunk: Chunk) -> None:
    changed = insert_payload(gold_chunk, middle_attack)

    split_at = len(gold_chunk.text) // 2
    assert changed.text.startswith(gold_chunk.text[:split_at])
    assert changed.text.endswith(gold_chunk.text[split_at:])
    assert middle_attack.payload in changed.text
    assert changed.chunk_id != gold_chunk.chunk_id


@pytest.mark.parametrize(
    ("position", "expected_text"),
    [
        (AttackPosition.HEAD, "PAYLOAD_SYN\n\nSynthetic original evidence."),
        (AttackPosition.TAIL, "Synthetic original evidence.\n\nPAYLOAD_SYN"),
    ],
)
def test_insert_payload_uses_requested_edge_position(
    gold_chunk: Chunk, position: AttackPosition, expected_text: str
) -> None:
    changed = insert_payload(gold_chunk, _attack(position=position))

    assert changed.text == expected_text


def test_insert_payload_preserves_source_metadata_except_malicious_keys(gold_chunk: Chunk) -> None:
    attack = _attack()

    changed = insert_payload(gold_chunk, attack)

    assert changed.document_id == gold_chunk.document_id
    assert changed.metadata == {
        "origin": "synthetic",
        "is_malicious": True,
        "attack_id": attack.attack_id,
        "clean_chunk_id": gold_chunk.chunk_id,
    }


def test_insert_payload_id_changes_when_complete_canonical_inputs_change(gold_chunk: Chunk) -> None:
    first = insert_payload(gold_chunk, _attack())
    changed_source = gold_chunk.model_copy(update={"text": "Different synthetic evidence."})

    second = insert_payload(changed_source, _attack())

    assert first.chunk_id != second.chunk_id


@pytest.mark.parametrize(
    ("filename", "record_type"),
    [
        ("chunks.jsonl", Chunk),
        ("questions.jsonl", Question),
        ("attacks.jsonl", AttackCase),
    ],
)
def test_synthetic_jsonl_fixtures_parse_as_domain_records(
    filename: str, record_type: type[Chunk] | type[Question] | type[AttackCase]
) -> None:
    path = Path(__file__).parent / "fixtures" / filename
    lines = path.read_text(encoding="utf-8").splitlines()

    records = [record_type.model_validate(json.loads(line)) for line in lines]

    assert records
    assert all("SYN" in record.model_dump_json() for record in records)
    for record in records:
        serialized = record.model_dump_json()
        scheme_positions = [match.start() for match in re.finditer(r"https?://", serialized, re.I)]
        for index, start in enumerate(scheme_positions):
            end = (
                scheme_positions[index + 1]
                if index + 1 < len(scheme_positions)
                else len(serialized)
            )
            hostname = urlparse(serialized[start:end]).hostname
            assert hostname == "invalid" or hostname is not None and hostname.endswith(".invalid")
