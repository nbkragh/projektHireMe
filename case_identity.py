"""Stable identities for source case folders."""

import hashlib
import re
import unicodedata


def normalize_case_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Case name must be a non-empty string.")
    return unicodedata.normalize("NFC", name).strip().casefold()


def case_id_from_name(name: str) -> str:
    normalized = normalize_case_name(name)
    ascii_name = unicodedata.normalize("NFKD", normalized).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name).strip("-") or "case"
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
    return f"case-{slug[:48]}-{digest}"
