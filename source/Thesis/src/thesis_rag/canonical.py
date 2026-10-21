import hashlib
import json
import unicodedata
from typing import TypeAlias, cast

from pydantic import BaseModel

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def _normalize_strings(value: JsonValue) -> JsonValue:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, list):
        return [_normalize_strings(item) for item in value]
    if isinstance(value, dict):
        normalized_dict: dict[str, JsonValue] = {}
        for key, item in value.items():
            normalized_key = unicodedata.normalize("NFC", key)
            if normalized_key in normalized_dict:
                raise ValueError(f"NFC-normalized key collision for {normalized_key!r}")
            normalized_dict[normalized_key] = _normalize_strings(item)
        return normalized_dict
    return value


def canonical_json(value: JsonValue | BaseModel) -> str:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    normalized = _normalize_strings(cast(JsonValue, payload))
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_sha256(value: JsonValue | BaseModel) -> str:
    return sha256_text(canonical_json(value))


def stable_id(prefix: str, value: JsonValue | BaseModel) -> str:
    digest = canonical_sha256(value)
    return f"{prefix}_{digest[:24]}"
