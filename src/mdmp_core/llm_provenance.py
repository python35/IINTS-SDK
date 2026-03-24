from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_llm_training_card(
    *,
    model_name: str,
    corpora: List[Dict[str, Any]],
    tokenizer: str,
    pretraining_tokens: int,
    fine_tune_tokens: int = 0,
) -> Dict[str, Any]:
    return {
        "spec_version": "1.0",
        "mdmp_object": "llm_training_card",
        "model": {
            "name": model_name,
            "created_utc": _now_iso(),
            "tokenizer": tokenizer,
            "pretraining_tokens": int(pretraining_tokens),
            "fine_tune_tokens": int(fine_tune_tokens),
            "training_corpora": corpora,
        },
        "provenance": {
            "schema": "mdmp-llm-provenance-v0",
            "status": "draft",
        },
    }
