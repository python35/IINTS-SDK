# Running IINTS Neural Forecasting v1 on NVIDIA Jetson Orin Nano Super (JetPack 6.2.1)

> **Research & Education Only (`clinical_use=false`):** The Jetson acts strictly as an isolated AI research compute node. Never connect outputs to an insulin pump or clinical dosing workflow.

---

## 1. Architecture Overview

1. **Training & Benchmark Host (PC or Jetson):** Trains `Persistence`, `Ridge`, `GRU-32`, and `Causal TCN-16` on embargoed 12-sample CGM windows and exports `.iints_edge.json` + static `(1, 12, 1)` `.onnx` (opset 17).
2. **NVIDIA Jetson Orin Nano Super (JetPack 6.2.1):**
   - Loads `.onnx` via `onnxruntime-gpu` (`TensorrtExecutionProvider` / `CUDAExecutionProvider`) or compiles `.plan` engines directly via `/usr/src/tensorrt/bin/trtexec --fp16` (or runs the zero-dependency `.iints_edge.json` bundle).
   - Samples real-time rail power (`VDD_IN` mW), GPU utilization (`GR3D_FREQ %`), RAM, and die temperatures via `tegrastats --interval 100`.
   - Serves standardized `PredictionRecord` + telemetry over HTTP (`iints forecasting serve-jetson-worker`).

---

## 2. Setup on Jetson Orin Nano Super (JetPack 6.2.1)

```bash
git clone https://github.com/python35/IINTS-SDK.git
cd IINTS-SDK
python3 -m venv .venv-jetson --system-site-packages
source .venv-jetson/bin/activate
pip install -e .
# Optional if not already in system-site-packages on JetPack 6.2.1:
# pip install onnx onnxruntime-gpu
```

### Lock Jetson Clocks for Reproducible Latency & Power Benchmarking
Before comparing GRU vs TCN latency/energy on the Orin Nano Super, lock the power mode (e.g. `15W` or `25W MAXN_SUPER`) so dynamic frequency scaling does not confound the comparison:

```bash
sudo nvpmodel -q
# Optional reproducible benchmark lock:
sudo jetson_clocks
```

---

## 3. Export ONNX, Compile TensorRT FP16 & Verify Numerical Parity

```bash
export PYTHONPATH=src

# 1. Train & export models (or scp the exported folder from your Windows PC)
python3 tools/forecasting/run_full_neural_jetson_pipeline.py --out-dir results/jetson_study

# 2. Optional direct TensorRT FP16 engine build on JetPack 6.2.1
/usr/src/tensorrt/bin/trtexec \
    --onnx=results/jetson_study/edge_exports/iints-tcn-glucose-v1.onnx \
    --saveEngine=results/jetson_study/edge_exports/iints-tcn-glucose-v1_fp16.plan \
    --fp16

# 3. Run 500-iteration batch-size-1 latency + tegrastats VDD_IN power benchmark
python3 -m iints.cli.cli forecasting edge-benchmark \
    results/jetson_study/edge_exports/iints-tcn-glucose-v1.iints_edge.json \
    --warmup 50 --runs 500 \
    --out-json results/jetson_study/jetson_tcn_telemetry.json

python3 -m iints.cli.cli forecasting edge-benchmark \
    results/jetson_study/edge_exports/iints-gru-glucose-v1.iints_edge.json \
    --warmup 50 --runs 500 \
    --out-json results/jetson_study/jetson_gru_telemetry.json
```

> **Note on Power & Energy Measurement:** `tegrastats` reads the onboard INA3221 `VDD_IN` rail (total module + carrier power). Always record an **idle baseline** (`tegrastats` for 30 s with no inference running) to separate dynamic inference power from idle board draw, and verify with an external USB-C/DC inline wattmeter when publishing hardware energy claims.

---

## 4. Live Network Inference Worker (Jetson ↔ PC)

On the **Jetson Orin Nano Super**:
```bash
export PYTHONPATH=src
python3 -m iints.cli.cli forecasting serve-jetson-worker \
    results/jetson_study/edge_exports/iints-tcn-glucose-v1.iints_edge.json \
    --host 0.0.0.0 --port 8770
```

From your **Windows PC** (replace `<JETSON_IP>`):
```powershell
Invoke-RestMethod -Method Post -Uri "http://<JETSON_IP>:8770/predict" `
    -ContentType "application/json" `
    -Body '{"timestamp":"2027-01-10T12:00:00Z","cgm_window_mgdl":[118,121,125,129,133,136,140,142,145,148,150,151],"prediction_horizon_minutes":30}'
```
