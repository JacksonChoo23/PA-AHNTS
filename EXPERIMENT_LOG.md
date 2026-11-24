# PA-AHNTS Experiment & Engineering Log

## Phase 0 (Gate 0) — Hardware & Environment Probe
- **Timestamp (UTC):** 2026-11-24T13:02:24Z
- **Status:** `PASSED` (`configs/hardware_profile.json` generated from live hardware/OS APIs; zero mocked metrics)

### 1. Host Platform & Compute Topology
- **System:** LENOVO `83F3`, Microsoft Windows 11 Pro (`10.0.26300`, 64-bit)
- **CPU:** `Intel(R) Core(TM) Ultra 7 255HX` (20 cores / 20 threads)
- **System RAM:** `31.43 GiB` (`33,752,997,888` bytes; `17.36 GiB` available at probe time)
- **dGPU (Adapter 0):** `NVIDIA GeForce RTX 5060 Laptop GPU` (`0x10DE:0x2D59`, driver `621.35`, CUDA CC `12.0` Blackwell)
  - Total VRAM (NVML): `8151.0 MiB` (`8,546,942,976` bytes) -> **confirms 8GB target laptop envelope**
  - DXGI Local Memory Budget: `7131.0 MiB`; Non-Local Shared Budget: `17579.94 MiB`
  - PCIe Topology: Max `Gen 5 x16`, active laptop link width `x8` (`Gen 5 x8` under active query; scales down to `Gen 2` at idle)
  - D3D12 Capabilities: Feature Level `12_2`, Shader Model `6.8`, Tiled Resources `Tier 4`, Sampler Feedback `Tier 0.9 (90)`, Native 16-bit Shader Ops `true`, Direct & Copy Queue Timestamps `1,000,000,000 Hz`
  - DirectML: `DMLCreateDevice` succeeded (`0x00000000`, `DirectML.dll 1.15.6`)
- **iGPU (Adapter 1):** `Intel(R) Graphics` (`0x8086:0x7D67`, driver `32.0.101.8860`)
  - D3D12 Feature Level `12_2`, SM `6.8`, UMA `true`, Native 16-bit Shader Ops `true`, DirectML `DMLCreateDevice` succeeded (`0x00000000`), Timestamp Freq `19.2 MHz`
- **NPU:** `Intel(R) AI Boost` (`PCI\VEN_8086&DEV_AD1D`, driver `32.0.100.4841`, PnP Status `OK`)

### 2. Telemetry Subsystem Verification
- **NVML (`nvml.dll 13.621.35`):** Live readings verified without elevation (`25.223 W`, default limit `50.0 W`, enforced limit `115.0 W`, `52 C`, GFX `1395 MHz` / Mem `12001 MHz`).
- **Windows PDH (`pdh.dll`) RAPL Energy Meter:** Verified **without Administrator privileges** (`is_admin: false`):
  - `\Energy Meter(RAPL_Package0_PKG)\Power` = `11.738 W`
  - `\Energy Meter(RAPL_Package0_PP0)\Power` = `5.255 W`
  - GPU PDH counter sets (`GPU Engine`, `GPU Process Memory`, `GPU Local Adapter Memory`, `GPU Non Local Adapter Memory`) all present.

### 3. Toolchain & Dependency Audit
- **Present:** `CMake 4.4.4`, `Ninja 1.13.2`, `MinGW-w64 g++ 16.2.0`, `Python 3.12.10`, `PyTorch 2.9.1+cu130` (CUDA `13.0` active).
- **Missing / Needed for Phases 1–3:**
  - `MSVC (VS 2022 Build Tools cl.exe)` & `Windows SDK` (`dxc.exe` / `dxcompiler.dll` for HLSL SM 6.6+ FP16 & Sampler Feedback compilation)
  - `third_party/agility_sdk` (for consistent DX12 Sampler Feedback Tier 1.0 & Enhanced Barriers)
  - `flip-evaluator` (Python package for NVIDIA FLIP visual quality evaluation in Phase 1)
  - `onnxruntime-directml` and/or `openvino` (for Phase 3 NPU/iGPU/CPU decode micro-benchmarks)

## Phase 1 (Gate 1) — Baseline Reproduction & Telemetry Verification
- **Timestamp (UTC):** 2026-11-24T13:58:48Z
- **Status:** `PASSED` (`results/phase1/{frame_metrics,memory_power,quality_metrics}.csv`, `results/phase1/phase1_summary.json`)
- **Testbed:** Native C++17 DX12 Ultimate (`1920x1080`, `ALLOW_TEARING`, Agility SDK `1.620.2`, SM 6.6 HLSL via `dxc 1.9.2609.5`, Sampler Feedback `Tier 0.9` `{256,256,1}`, `16`-bay architectural gallery, `32` `8192x8192` PBR materials = `8,192 MiB` raw BC7/BC5 footprint).
- **Protocol:** `N = 5` repeated runs per baseline, `120` warm-up + `1000` measured frames per run (`20,000` measured frame rows; `624` 20Hz NVML + PDH-RAPL + DXGI samples; `5` deterministic camera checkpoints evaluated with PSNR & NVIDIA `flip-evaluator 1.7`).

### Phase 1 Summary (`N = 5`, mean ± std)
| Config | Avg FPS | CPU Frame (ms) | GPU Frame (ms) | P99 CPU (ms) | Peak Local VRAM (MB) | Non-Local VRAM (MB) | PCIe Upload (MB) | dGPU Decoded Tiles | Avg dGPU / Total Power (W) | PSNR (dB) | FLIP (mean / max P99) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `B0_BCn_Full` | `2769.9 ± 113.1` | `0.36 ± 0.01` | `0.33 ± 0.01` | `1.06 ± 0.06` | `8247.6 ± 0.4` | `50.3 ± 0.1` | `0.0` | `0` | `22.17 / 40.48` | `99.00` (ref) | `0.0000 / 0.0000` |
| `B1_SFS_BCn` | `781.9 ± 71.6` | `1.29 ± 0.11` | `0.44 ± 0.00` | `9.69 ± 1.43` | `1605.6 ± 0.0` | `80.3 ± 0.4` | `481.7` | `0` | `36.18 / 60.54` | `68.61` | `0.0002 / 0.0194` |
| `B2_NTC_OnSample` | `342.7 ± 5.5` | `2.92 ± 0.05` | `2.88 ± 0.04` | `4.62 ± 0.07` | `3126.7 ± 0.0` | `50.1 ± 0.0` | `0.0` | `0` | `81.80 / 94.50` | `32.45` | `0.0510 / 0.4446` |
| `B3_NTC_OnFeedback` | `537.8 ± 39.5` | `1.87 ± 0.13` | `0.53 ± 0.00` | `14.23 ± 1.63` | `4762.3 ± 0.2` | `54.7 ± 0.3` | `0.0` | `2569` | `39.60 / 58.36` | `31.62` | `0.0541 / 0.5064` |

### Key Empirical & Physical Findings
1. **Multi-Scale NTC Fix:** Initial run yielded ~23–24 dB PSNR because the NTC MLP was trained only at `mip=0` (`lod_atten=1.0`) and `G1` (`1024x1024`) had an unaligned mip offset in `ntc_decode_bcn.cs.hlsl`. Switching to multi-scale LOD-aware training (`mips 0..4`) and aligning `G1` mip bias (`max(0, lod - 2)`) improved rendered quality to `32.45 dB` (`B2`, FLIP `0.0510`) and `31.62 dB` (`B3`, FLIP `0.0541`).
2. **NTC-on-Sample Power/Shader Penalty (`B2`):** Per-pixel `16->32->32->8` FP16 MLP inference inflates GPU frame time `8.7x` (`0.33 -> 2.88 ms`) and drives dGPU power from `22.2 W` to `81.8 W` (`94.5 W` total).
3. **NTC-on-Feedback Bottleneck (`B3`):** Moving NTC decode to Async Compute on Sampler Feedback slashes GPU frame time back to `0.53 ms` and total power to `58.36 W`, **but still consumes `4,762.3 MB` Local VRAM** (`3,072 MB` VRAM-resident NTC latents + `1,536 MB` BCn tile cache) and incurs `14.23 ms` P99 CPU stalls from synchronous `UpdateTileMappings` + `fb_readback` fences. Offloading cold/warm NTC latents to System RAM (`Phase 2`) directly targets this `~3.0 GB` latent footprint and P99 jitter.
