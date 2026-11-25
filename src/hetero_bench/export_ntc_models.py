#!/usr/bin/env python3
"""Export trained PA-AHNTS NTC MLPs (16 -> H -> H -> 8) as static-shape NCHW 1x1 Conv2D
ONNX models in FP32, FP16, and calibrated INT8 QDQ formats for DirectML, OpenVINO NPU/iGPU/CPU,
and ONNX Runtime benchmarking.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "phase3" / "models"


def load_trained_mlp_weights(manifest_path: Path, pack_path: Path) -> list[dict[str, np.ndarray]]:
    """Load real FP16 MLP weights [0..31] from pbr_ntc_pack_v1.bin."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mlp_tbl = manifest["mlp_table"]
    mlp_off = int(mlp_tbl["offset"])
    mlp_stride = int(mlp_tbl["stride_bytes"])
    blob = pack_path.read_bytes()

    weights: list[dict[str, np.ndarray]] = []
    for mat in manifest["materials"]:
        mid = int(mat["material_id"])
        base = mlp_off + mid * mlp_stride
        w1 = np.frombuffer(blob, dtype=np.float16, count=16 * 32, offset=base + 0).reshape(16, 32).astype(np.float32)
        b1 = np.frombuffer(blob, dtype=np.float16, count=32, offset=base + 1024).astype(np.float32)
        w2 = np.frombuffer(blob, dtype=np.float16, count=32 * 32, offset=base + 1088).reshape(32, 32).astype(np.float32)
        b2 = np.frombuffer(blob, dtype=np.float16, count=32, offset=base + 3136).astype(np.float32)
        w3 = np.frombuffer(blob, dtype=np.float16, count=32 * 8, offset=base + 3200).reshape(32, 8).astype(np.float32)
        b3 = np.frombuffer(blob, dtype=np.float16, count=8, offset=base + 3712).astype(np.float32)
        weights.append({"w1": w1, "b1": b1, "w2": w2, "b2": b2, "w3": w3, "b3": b3})
    return weights


def make_synthetic_weights(hidden_dim: int, seed: int = 42) -> dict[str, np.ndarray]:
    """Deterministic Kaiming-scaled FP16 weights for H=16 / H=64 architecture sweep."""
    rng = np.random.default_rng(seed + hidden_dim)
    w1 = (rng.standard_normal((16, hidden_dim)) / math.sqrt(16.0)).astype(np.float16).astype(np.float32)
    b1 = np.zeros((hidden_dim,), dtype=np.float32)
    w2 = (rng.standard_normal((hidden_dim, hidden_dim)) / math.sqrt(hidden_dim)).astype(np.float16).astype(np.float32)
    b2 = np.zeros((hidden_dim,), dtype=np.float32)
    w3 = (rng.standard_normal((hidden_dim, 8)) / math.sqrt(hidden_dim)).astype(np.float16).astype(np.float32)
    b3 = np.zeros((8,), dtype=np.float32)
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "w3": w3, "b3": b3}


def build_real_calib_tiles(manifest_path: Path, pack_path: Path, mat_id: int = 0, tile_dim: int = 256) -> np.ndarray:
    """Build 4 real [1, 16, 256, 256] float32 input tiles from material mat_id latent grids (512x512 = 4 tiles)."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    blob = pack_path.read_bytes()
    mat = manifest["materials"][mat_id]

    g0_s = mat["g0_mips"][0]
    g0 = np.frombuffer(blob, dtype=np.uint8, count=256 * 256 * 4, offset=int(g0_s["offset"])).reshape(256, 256, 4)
    g0_f = g0.astype(np.float32) / 255.0

    g1_s = mat["g1_mips"][0]
    g1 = np.frombuffer(blob, dtype=np.float16, count=64 * 64 * 4, offset=int(g1_s["offset"])).reshape(64, 64, 4)
    g1_f = g1.astype(np.float32)

    def sample_bilinear(grid: np.ndarray, uv_x: np.ndarray, uv_y: np.ndarray) -> np.ndarray:
        h, w, c = grid.shape
        u = np.mod(uv_x, 1.0) * w - 0.5
        v = np.mod(uv_y, 1.0) * h - 0.5
        x0 = np.floor(u).astype(np.int64)
        y0 = np.floor(v).astype(np.int64)
        fx = (u - x0)[..., None]
        fy = (v - y0)[..., None]
        x0 = np.mod(x0, w)
        y0 = np.mod(y0, h)
        x1 = np.mod(x0 + 1, w)
        y1 = np.mod(y0 + 1, h)
        v00 = grid[y0, x0]
        v10 = grid[y0, x1]
        v01 = grid[y1, x0]
        v11 = grid[y1, x1]
        top = v00 * (1.0 - fx) + v10 * fx
        bot = v01 * (1.0 - fx) + v11 * fx
        return (top * (1.0 - fy) + bot * fy).astype(np.float32)

    tiles = np.zeros((4, 16, tile_dim, tile_dim), dtype=np.float32)
    two_pi = 2.0 * math.pi
    idx = 0
    for ty in range(2):
        for tx in range(2):
            px = tx * tile_dim + np.arange(tile_dim, dtype=np.float32) + 0.5
            py = ty * tile_dim + np.arange(tile_dim, dtype=np.float32) + 0.5
            yy, xx = np.meshgrid(py, px, indexing="ij")
            u = xx / 512.0
            v = yy / 512.0
            g0_s_val = sample_bilinear(g0_f, u, v) * 2.0 - 1.0
            g1_s_val = sample_bilinear(g1_f, u, v)
            tile_u = np.mod(u * 16.0, 1.0)
            tile_v = np.mod(v * 16.0, 1.0)
            pe = np.stack(
                [
                    np.sin(two_pi * 8.0 * tile_u),
                    np.cos(two_pi * 8.0 * tile_u),
                    np.sin(two_pi * 8.0 * tile_v),
                    np.cos(two_pi * 8.0 * tile_v),
                    np.sin(two_pi * 16.0 * tile_u),
                    np.cos(two_pi * 16.0 * tile_u),
                    np.sin(two_pi * 16.0 * tile_v),
                    np.cos(two_pi * 16.0 * tile_v),
                ],
                axis=-1,
            ).astype(np.float32)
            feat = np.concatenate([g0_s_val, g1_s_val, pe], axis=-1)  # [256, 256, 16]
            tiles[idx] = np.transpose(feat, (2, 0, 1))
            idx += 1
    return tiles


def build_ntc_onnx(
    weights: dict[str, np.ndarray],
    batch_size: int,
    hidden_dim: int,
    fp16: bool = False,
    tile_dim: int = 256,
) -> onnx.ModelProto:
    """Create static-shape [B, 16, 256, 256] -> [B, 8, 256, 256] 1x1 Conv2D ONNX model."""
    dtype = np.float16 if fp16 else np.float32
    tensor_dtype = TensorProto.FLOAT16 if fp16 else TensorProto.FLOAT

    w1_c = np.ascontiguousarray(weights["w1"].T.reshape(hidden_dim, 16, 1, 1).astype(dtype))
    b1_c = np.ascontiguousarray(weights["b1"].astype(dtype))
    w2_c = np.ascontiguousarray(weights["w2"].T.reshape(hidden_dim, hidden_dim, 1, 1).astype(dtype))
    b2_c = np.ascontiguousarray(weights["b2"].astype(dtype))
    w3_c = np.ascontiguousarray(weights["w3"].T.reshape(8, hidden_dim, 1, 1).astype(dtype))
    b3_c = np.ascontiguousarray(weights["b3"].astype(dtype))

    in_t = helper.make_tensor_value_info("features", tensor_dtype, [batch_size, 16, tile_dim, tile_dim])
    out_t = helper.make_tensor_value_info("pbr_out", tensor_dtype, [batch_size, 8, tile_dim, tile_dim])

    inits = [
        numpy_helper.from_array(w1_c, name="W1"),
        numpy_helper.from_array(b1_c, name="B1"),
        numpy_helper.from_array(w2_c, name="W2"),
        numpy_helper.from_array(b2_c, name="B2"),
        numpy_helper.from_array(w3_c, name="W3"),
        numpy_helper.from_array(b3_c, name="B3"),
    ]

    nodes = [
        helper.make_node("Conv", ["features", "W1", "B1"], ["c1"], kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0], name="conv1"),
        helper.make_node("LeakyRelu", ["c1"], ["a1"], alpha=0.125, name="lrelu1"),
        helper.make_node("Conv", ["a1", "W2", "B2"], ["c2"], kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0], name="conv2"),
        helper.make_node("LeakyRelu", ["c2"], ["a2"], alpha=0.125, name="lrelu2"),
        helper.make_node("Conv", ["a2", "W3", "B3"], ["c3"], kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0], name="conv3"),
        helper.make_node("Sigmoid", ["c3"], ["pbr_out"], name="sigmoid_out"),
    ]

    graph = helper.make_graph(nodes, f"NTC_MLP_H{hidden_dim}_B{batch_size}", [in_t], [out_t], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def quantize_ntc_int8(
    fp32_path: Path,
    int8_path: Path,
    calib_tiles: np.ndarray,
    target_batch: int | None = None,
) -> None:
    """Static QDQ INT8 quantization calibrated with real latent tiles, with optional batch-dim rewrite."""
    from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static

    samples = [calib_tiles[i : i + 1] for i in range(calib_tiles.shape[0])]

    class NtcCalibReader(CalibrationDataReader):
        def __init__(self, data_list: list[np.ndarray]) -> None:
            self._data = iter([{"features": x} for x in data_list])

        def get_next(self) -> dict[str, np.ndarray] | None:
            return next(self._data, None)

    quantize_static(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        calibration_data_reader=NtcCalibReader(samples),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
    )

    if target_batch is not None:
        m = onnx.load(str(int8_path))
        m.graph.input[0].type.tensor_type.shape.dim[0].dim_value = int(target_batch)
        m.graph.output[0].type.tensor_type.shape.dim[0].dim_value = int(target_batch)
        for vi in m.graph.value_info:
            if len(vi.type.tensor_type.shape.dim) == 4:
                vi.type.tensor_type.shape.dim[0].dim_value = int(target_batch)
        onnx.checker.check_model(m)
        onnx.save(m, str(int8_path))


def export_all_models(out_dir: Path, skip_per_mat: bool = False) -> dict[str, Any]:
    manifest_path = REPO_ROOT / "results" / "assets_cache" / "manifest.json"
    pack_path = REPO_ROOT / "results" / "assets_cache" / "pbr_ntc_pack_v1.bin"
    all_weights = load_trained_mlp_weights(manifest_path, pack_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    calib_4_m0 = build_real_calib_tiles(manifest_path, pack_path, mat_id=0)

    # 1) Primary H=32 batch sweep B in [1, 4, 16, 64, 256]
    batches = [1, 4, 16, 64, 256]
    fp32_b1_path = out_dir / "ntc_h32_b1_fp32.onnx"
    for b in batches:
        m_fp32 = build_ntc_onnx(all_weights[0], batch_size=b, hidden_dim=32, fp16=False)
        m_fp16 = build_ntc_onnx(all_weights[0], batch_size=b, hidden_dim=32, fp16=True)
        fp32_p = out_dir / f"ntc_h32_b{b}_fp32.onnx"
        fp16_p = out_dir / f"ntc_h32_b{b}_fp16.onnx"
        int8_p = out_dir / f"ntc_h32_b{b}_int8.onnx"
        onnx.save(m_fp32, str(fp32_p))
        onnx.save(m_fp16, str(fp16_p))

        # Calibrate QDQ scales on 4 real tiles (B=1) and rewrite batch dim to b (saves huge RAM for B=64/256)
        quantize_ntc_int8(fp32_b1_path, int8_p, calib_4_m0, target_batch=b)
        print(f"[export_ntc_models] H=32 B={b}: fp32, fp16, int8 -> {out_dir}")

    # 2) Hidden-dim sweep H in [16, 64] at B=16 (H=32 B=16 already exported above)
    for h in [16, 64]:
        w_h = make_synthetic_weights(h)
        m32_1 = build_ntc_onnx(w_h, batch_size=1, hidden_dim=h, fp16=False)
        b1_tmp = out_dir / f"_tmp_h{h}_b1_fp32.onnx"
        onnx.save(m32_1, str(b1_tmp))

        m32_16 = build_ntc_onnx(w_h, batch_size=16, hidden_dim=h, fp16=False)
        m16_16 = build_ntc_onnx(w_h, batch_size=16, hidden_dim=h, fp16=True)
        onnx.save(m32_16, str(out_dir / f"ntc_h{h}_b16_fp32.onnx"))
        onnx.save(m16_16, str(out_dir / f"ntc_h{h}_b16_fp16.onnx"))
        quantize_ntc_int8(b1_tmp, out_dir / f"ntc_h{h}_b16_int8.onnx", calib_4_m0, target_batch=16)
        b1_tmp.unlink(missing_ok=True)
        print(f"[export_ntc_models] H={h} B=16: fp32, fp16, int8 -> {out_dir}")

    # 3) Per-material (0..31) B=4 models for exact 32-material Quantization Loss audit on NPU INT8
    if not skip_per_mat:
        pm_dir = out_dir / "per_mat"
        pm_dir.mkdir(parents=True, exist_ok=True)
        for mid in range(32):
            c4 = build_real_calib_tiles(manifest_path, pack_path, mat_id=mid)
            m_b1_fp32 = build_ntc_onnx(all_weights[mid], batch_size=1, hidden_dim=32, fp16=False)
            tmp_b1 = pm_dir / f"_tmp_mat_{mid:02d}_b1.onnx"
            onnx.save(m_b1_fp32, str(tmp_b1))

            m_b4_fp16 = build_ntc_onnx(all_weights[mid], batch_size=4, hidden_dim=32, fp16=True)
            onnx.save(m_b4_fp16, str(pm_dir / f"mat_{mid:02d}_b4_fp16.onnx"))
            quantize_ntc_int8(tmp_b1, pm_dir / f"mat_{mid:02d}_b4_int8.onnx", c4, target_batch=4)
            tmp_b1.unlink(missing_ok=True)
        print(f"[export_ntc_models] Exported 32 per-material FP16 + calibrated INT8 models -> {pm_dir}")

    return {"out_dir": str(out_dir), "batches": batches, "hidden_dims": [16, 32, 64]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Export PA-AHNTS NTC ONNX FP32/FP16/INT8 models")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--skip-per-mat", action="store_true")
    args = parser.parse_args()
    export_all_models(args.out_dir, skip_per_mat=args.skip_per_mat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
