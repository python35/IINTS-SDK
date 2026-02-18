from __future__ import annotations

import argparse
from pathlib import Path

try:
    import torch
except Exception as exc:
    raise SystemExit(
        "Torch is required for export. Install with `pip install iints-sdk-python35[research]`."
    ) from exc

from iints.research.predictor import load_predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path, help="Model checkpoint (.pt)")
    parser.add_argument("--out", required=True, type=Path, help="Output ONNX path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, cfg = load_predictor(args.model)
    model.eval()

    dummy = torch.zeros(1, cfg["history_steps"], cfg["input_size"], dtype=torch.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=["inputs"],
        output_names=["predictions"],
        dynamic_axes={"inputs": {0: "batch"}, "predictions": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported ONNX: {args.out}")


if __name__ == "__main__":
    main()
