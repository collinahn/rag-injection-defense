from typing import cast

from thesis_rag.canonical import JsonValue, stable_id
from thesis_rag.domain import AttackCase, AttackPosition, Chunk, MetadataValue

_ATTACK_METADATA_KEYS = frozenset({"is_malicious", "attack_id", "clean_chunk_id"})


def insert_payload(chunk: Chunk, attack: AttackCase) -> Chunk:
    if attack.position is AttackPosition.HEAD:
        text = f"{attack.payload}\n\n{chunk.text}"
    elif attack.position is AttackPosition.MIDDLE:
        split_at = len(chunk.text) // 2
        text = f"{chunk.text[:split_at]}\n\n{attack.payload}\n\n{chunk.text[split_at:]}"
    else:
        text = f"{chunk.text}\n\n{attack.payload}"

    metadata: dict[str, MetadataValue] = {
        **chunk.metadata,
        "is_malicious": True,
        "attack_id": attack.attack_id,
        "clean_chunk_id": chunk.chunk_id,
    }
    return Chunk(
        chunk_id=stable_id(
            "MCH",
            cast(
                JsonValue,
                {
                    "clean_chunk": chunk.model_dump(mode="json"),
                    "attack": attack.model_dump(mode="json"),
                    "text": text,
                    "metadata": metadata,
                },
            ),
        ),
        document_id=chunk.document_id,
        text=text,
        language=chunk.language,
        metadata=metadata,
    )


def reconstruct_clean_chunk(malicious: Chunk, attack: AttackCase) -> Chunk:
    clean_chunk_id = malicious.metadata.get("clean_chunk_id")
    if not isinstance(clean_chunk_id, str) or not clean_chunk_id.strip():
        raise ValueError("malicious chunk clean_chunk_id must be a non-empty string")
    if clean_chunk_id == malicious.chunk_id:
        raise ValueError("malicious chunk clean_chunk_id must not be self-referential")

    candidate_texts: tuple[str, ...]
    if attack.position is AttackPosition.HEAD:
        prefix = f"{attack.payload}\n\n"
        candidate_texts = (
            (malicious.text.removeprefix(prefix),)
            if malicious.text.startswith(prefix)
            else ()
        )
    elif attack.position is AttackPosition.TAIL:
        suffix = f"\n\n{attack.payload}"
        candidate_texts = (
            (malicious.text.removesuffix(suffix),)
            if malicious.text.endswith(suffix)
            else ()
        )
    else:
        delimiter = f"\n\n{attack.payload}\n\n"
        candidate_texts = tuple(
            malicious.text[:index] + malicious.text[index + len(delimiter) :]
            for index in _occurrence_indices(malicious.text, delimiter)
        )

    clean_metadata = {
        key: value
        for key, value in malicious.metadata.items()
        if key not in _ATTACK_METADATA_KEYS
    }
    canonical_candidates: list[Chunk] = []
    for text in candidate_texts:
        try:
            clean = Chunk(
                chunk_id=clean_chunk_id,
                document_id=malicious.document_id,
                text=text,
                language=malicious.language,
                metadata=clean_metadata,
            )
        except ValueError:
            continue
        if insert_payload(clean, attack) == malicious and clean not in canonical_candidates:
            canonical_candidates.append(clean)

    if len(canonical_candidates) != 1:
        raise ValueError("malicious chunk is not one canonical payload insertion")
    return canonical_candidates[0]


def _occurrence_indices(text: str, delimiter: str) -> tuple[int, ...]:
    indices: list[int] = []
    start = 0
    while True:
        index = text.find(delimiter, start)
        if index < 0:
            return tuple(indices)
        indices.append(index)
        start = index + 1
