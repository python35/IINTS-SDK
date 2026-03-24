from __future__ import annotations

from typing import Any


def card_to_prov(card: dict[str, Any]) -> dict[str, Any]:
    """
    Export an MDMP card as W3C PROV-compatible JSON-LD.

    The output is intentionally minimal and stable for interoperability.
    """
    return {
        "@context": {
            "prov": "https://www.w3.org/ns/prov#",
            "mdmp": "https://mdmp.dev/ns#",
            "label": "prov:label",
            "generatedAtTime": "prov:generatedAtTime",
            "wasAttributedTo": "prov:wasAttributedTo",
        },
        "@type": "prov:Entity",
        "label": card.get("dataset") or card.get("source") or "mdmp_dataset",
        "generatedAtTime": card.get("signed_at") or card.get("created"),
        "wasAttributedTo": {
            "@type": "prov:Agent",
            "label": card.get("signed_by", "unknown"),
        },
        "mdmp:grade": card.get("grade"),
        "mdmp:fingerprint": card.get("fingerprint") or card.get("dataset_fingerprint"),
        "mdmp:specVersion": card.get("spec_version"),
        "mdmp:keyId": card.get("key_id"),
    }
