from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from iints.research.jetson_hf_trainer import _checkpoint_config


def test_checkpoint_inspection_never_falls_back_to_unsafe_pickle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "legacy.pt"
    checkpoint.write_bytes(b"not-used")

    def unsupported(*args, **kwargs):
        assert kwargs.get("weights_only") is True
        raise TypeError("weights_only unsupported")

    monkeypatch.setattr(torch, "load", unsupported)

    with pytest.raises(RuntimeError, match="weights_only=True"):
        _checkpoint_config(checkpoint)
