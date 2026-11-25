# PA-AHNTS — Phase 2 & Phase 3 Reinforcement

Suspend Phase 4 development. Reinforce the existing Phase 2 and Phase 3 implementations without rebuilding the project.

Read existing source code, raw CSVs, hardware profiles, and experiment logs first.

## Mandatory Rules

* **Real hardware telemetry only:** No mocking, fabricated data, or hardcoded results.

* **Preserve history:** Preserve historical results and existing implementations.

* **Statistical rigor:** Use identical workloads and at least $N=5$ independent runs. Report mean, standard deviation, and 95% CI.

* **Language conventions:** Code and logs in English; user-facing reports in Simplified Chinese.

* **Scientific honesty:** Negative results are valid. Report unsupported hypotheses honestly.

## Phase 2 Reinforcement

### 1. Memory Accounting

Unify `B0`–`B3` and `R1`–`R5` memory measurement definitions. Separate Local VRAM, Non-Local memory, allocated resources, and observable residency.

Create a resource-level memory breakdown explaining:

* Why a 0.31 GiB RAM latent pool corresponds to up to 2.74 GiB VRAM savings.

* Why `R1`–`R5` follow:
  $$\text{VRAM} = 1.66\text{ GiB} + \text{L1 cache capacity}$$

* How much memory is saved by RAM tiering versus cache resizing.

Avoid double-counting resources and backing heaps.

### 2. Fair Baselines and Ablation

Implement a controlled $2 \times 2$ ablation:

* **C0:** VRAM-resident + synchronous.

* **C1:** RAM-tiered + synchronous.

* **C2:** VRAM-resident + asynchronous.

* **C3:** RAM-tiered + asynchronous.

Use identical assets, cache capacity, mip requirements, camera paths, and rendering settings.

Add an optimized asynchronous SFS-BCn baseline with RAM caching.

Isolate the contributions of RAM tiering and asynchronous batching. Prioritize `R1` and `R2` configurations.

### 3. End-to-End Validation

Measure:

* Present FPS, 1% Low, and P99 frame time.

* Peak Local VRAM.

* Package-sum power and Joules/frame.

* PSNR, FLIP, and texture pop-in.

* Cache hit rate and tile deadline miss rate.

* PCIe upload traffic.

Verify Copy Queue timestamp frequency through `GetTimestampFrequency()` and use consistent memory units.

Validate whether the reported VRAM savings and latency improvements persist at equivalent visual quality.

Do not equate CPU/GPU execution time with displayed frame time, or package power with whole-system power.

## Phase 3 Reinforcement

### 4. Operation Count and Backend Audit

Verify actual execution devices, precision, model architecture, and CPU fallback.

For the $16 \rightarrow 32 \rightarrow 32 \rightarrow 8$ MLP:

* **MACs/texel:** $1792$

* **Ops/texel:** $3584$

* **$B=16$ total:** $3.758096384\text{ GOP}$

Recalculate effective GOP/s consistently across all backends.

Do not confuse effective TOPS with hardware occupancy. Treat SRAM and memory-bandwidth bottlenecks as hypotheses unless profiling confirms them.

### 5. Host Feature Gather Optimization

Directly measure the approximately $19.8\text{ ms}$ preparation overhead for $B=16$.

Optimize using contiguous memory, precomputed indices, SIMD/vectorization, and persistent buffers where appropriate.

Compare original and optimized implementations using:

* Gather latency.

* End-to-end latency.

* RAM overhead.

* Numerical equivalence.

Target $\le 5\text{ ms}$ gather latency if achievable, without sacrificing correctness.

### 6. Correct Break-Even Model

Replace inference-only comparisons with full end-to-end costs:

$$
T_{\text{complete}} = T_{\text{queue}} + T_{\text{batch\_wait}} + T_{\text{gather}} + T_{\text{inference}} + T_{\text{encode}} + T_{\text{upload}} + T_{\text{sync}}
$$

Account for overlapping asynchronous stages using the actual critical path.

Verify the preliminary $B=16$ results:

* **dGPU:** $3.48\text{ ms}$

* **NPU:** $67.53\text{ ms}$

* **iGPU:** $42.87\text{ ms}$

* **NPU queue-delay break-even:** $64.05\text{ ms}$

* **iGPU queue-delay break-even:** $39.39\text{ ms}$

Recalculate thresholds after optimization.

Measure actual GPU queue congestion under controlled background workloads. Clearly distinguish measured results from analytical extrapolation.

Do not impose a fixed 40 TOPS NPU eligibility threshold.

### 7. Energy and Concurrent Contention

Repeat dGPU-only and concurrent NPU/iGPU/CPU benchmarks under comparable thermal conditions.

Measure:

* Completed valid tiles per second.

* Incremental package power.

* Joules per tile.

* Concurrent aggregate throughput.

* dGPU clock, power, and throughput changes.

Determine whether performance degradation originates from CPU submission contention, shared resources, thermal limits, or actual Dynamic Boost power redistribution.

Do not claim NPU energy superiority solely from lower instantaneous power.

### 8. Scheduler Cost Model

Generate a hardware-specific cost model for Phase 4 using validated measurements:

* Backend and batch size.

* End-to-end P50/P99 latency.

* Queue delay.

* Energy per tile.

* Concurrent dGPU performance impact.

* RAM/VRAM cost.

* Visual error.

Validate predictions against separate benchmark runs.

## Deliverables and Verification

Store reinforced results separately:

* `results/phase2_reinforced/`

* `results/phase3_reinforced/`

Produce:

* Raw CSV/JSON data and reproducible plots.

* `PHASE_2_REINFORCEMENT_REPORT.md`

* `PHASE_3_REINFORCEMENT_REPORT.md`

* `scheduler_cost_model.json`

Update `EXPERIMENT_LOG.md`.

Report original versus corrected results, statistical uncertainty, unresolved limitations, and supported/rejected/inconclusive hypotheses.

Complete Phase 2 reinforcement before Phase 3. Stop after both verification gates and report findings in Simplified Chinese.

**Do not begin Phase 4 without explicit user approval.**