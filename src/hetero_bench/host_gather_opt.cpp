#include <immintrin.h>
#include <malloc.h>
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

constexpr int kTileDim = 256;
constexpr int kPlaneElems = kTileDim * kTileDim; // 65536
constexpr int kChannels = 16;
constexpr int kTileElems = kChannels * kPlaneElems; // 1048576
constexpr int kMaxMaterials = 32;
constexpr int kRowsPerStrip = 16;
constexpr int kStripsPerTile = kTileDim / kRowsPerStrip; // 16

inline uint16_t f32_to_f16_scalar(float x) {
    __m128 v = _mm_set_ss(x);
    __m128i h = _mm_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    return static_cast<uint16_t>(_mm_extract_epi16(h, 0));
}

struct BilinearLut1D {
    int i0[2][kTileDim]{};
    int i1[2][kTileDim]{};
    float f0[2][kTileDim]{};
    float f1[2][kTileDim]{};
};

struct GatherEngine {
    int max_batch = 0;
    int num_threads = 0;
    bool initialized = false;

    float* persistent_fp32 = nullptr;
    uint16_t* persistent_fp16 = nullptr;
    float* pe_lut_fp32 = nullptr;    // [8, 256, 256]
    uint16_t* pe_lut_fp16 = nullptr; // [8, 256, 256]

    float g0_u8_to_f32[256]{};
    BilinearLut1D g0_lut{};
    BilinearLut1D g1_lut{};

    const uint8_t* reg_g0[kMaxMaterials]{};
    const uint16_t* reg_g1[kMaxMaterials]{};

    // Persistent Win32 worker pool (zero libwinpthread DLL dependency)
    std::vector<HANDLE> workers;
    SRWLOCK mtx = SRWLOCK_INIT;
    CONDITION_VARIABLE cv_work = CONDITION_VARIABLE_INIT;
    CONDITION_VARIABLE cv_done = CONDITION_VARIABLE_INIT;
    bool stop_pool = false;
    uint64_t generation = 0;
    int active_jobs = 0;
    std::atomic<int> next_job{0};
    std::atomic<int> finished_workers{0};

    // Active dispatch parameters
    bool cur_is_fp16 = true;
    int cur_batch = 0;
    const int* cur_mat_ids = nullptr;
    const uint8_t* const* cur_g0_ptrs = nullptr;
    const uint16_t* const* cur_g1_ptrs = nullptr;
    const int* cur_txs = nullptr;
    const int* cur_tys = nullptr;
    float* cur_out_fp32 = nullptr;
    uint16_t* cur_out_fp16 = nullptr;

    void init_luts() {
        for (int v = 0; v < 256; ++v) {
            g0_u8_to_f32[v] = (static_cast<float>(v) / 255.0f) * 2.0f - 1.0f;
        }

        auto fill_1d = [](BilinearLut1D& lut, int grid_dim) {
            for (int q = 0; q < 2; ++q) {
                for (int idx = 0; idx < kTileDim; ++idx) {
                    double px = static_cast<double>(q * kTileDim + idx) + 0.5;
                    double u = px / 512.0; // in [0, 1)
                    double coord = u * static_cast<double>(grid_dim) - 0.5;
                    double f0_fl = std::floor(coord);
                    int x0 = static_cast<int>(static_cast<long long>(f0_fl));
                    double fx = coord - f0_fl;
                    x0 = ((x0 % grid_dim) + grid_dim) % grid_dim;
                    int x1 = (x0 + 1) % grid_dim;
                    lut.i0[q][idx] = x0;
                    lut.i1[q][idx] = x1;
                    lut.f0[q][idx] = static_cast<float>(1.0 - fx);
                    lut.f1[q][idx] = static_cast<float>(fx);
                }
            }
        };
        fill_1d(g0_lut, 256);
        fill_1d(g1_lut, 64);

        const double two_pi = 2.0 * 3.14159265358979323846;
        for (int iy = 0; iy < kTileDim; ++iy) {
            double v = (static_cast<double>(iy) + 0.5) / 512.0;
            double tv = std::fmod(v * 16.0, 1.0);
            for (int ix = 0; ix < kTileDim; ++ix) {
                double u = (static_cast<double>(ix) + 0.5) / 512.0;
                double tu = std::fmod(u * 16.0, 1.0);
                double vals[8] = {
                    std::sin(two_pi * 8.0 * tu),
                    std::cos(two_pi * 8.0 * tu),
                    std::sin(two_pi * 8.0 * tv),
                    std::cos(two_pi * 8.0 * tv),
                    std::sin(two_pi * 16.0 * tu),
                    std::cos(two_pi * 16.0 * tu),
                    std::sin(two_pi * 16.0 * tv),
                    std::cos(two_pi * 16.0 * tv),
                };
                int off = iy * kTileDim + ix;
                for (int c = 0; c < 8; ++c) {
                    float fv = static_cast<float>(vals[c]);
                    pe_lut_fp32[c * kPlaneElems + off] = fv;
                    pe_lut_fp16[c * kPlaneElems + off] = f32_to_f16_scalar(fv);
                }
            }
        }
    }

    void process_strip(int job_idx) const {
        int b = job_idx / kStripsPerTile;
        int strip = job_idx % kStripsPerTile;
        int y_beg = strip * kRowsPerStrip;
        int y_end = y_beg + kRowsPerStrip;

        const uint8_t* g0_base = cur_g0_ptrs ? cur_g0_ptrs[b] : reg_g0[cur_mat_ids[b] & 31];
        const uint16_t* g1_base = cur_g1_ptrs ? cur_g1_ptrs[b] : reg_g1[cur_mat_ids[b] & 31];
        int qx = cur_txs[b] & 1;
        int qy = cur_tys[b] & 1;

        alignas(32) float g0_r0[256], g0_r1[256], g0_r2[256], g0_r3[256];
        alignas(32) float g0_s0[256], g0_s1[256], g0_s2[256], g0_s3[256];
        alignas(32) float g1_r0[64], g1_r1[64], g1_r2[64], g1_r3[64];
        alignas(32) float g1_s0[64], g1_s1[64], g1_s2[64], g1_s3[64];

        const __m256i v_g0_x0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(g0_lut.i0[qx]));
        // Load LUT offsets in loop for each 8-element chunk

        for (int iy = y_beg; iy < y_end; ++iy) {
            // 1. Decode G0 row y0 and y1 (256 RGBA8 -> planar FP32 [-1, 1])
            int gy0 = g0_lut.i0[qy][iy];
            int gy1 = g0_lut.i1[qy][iy];
            const uint8_t* src0 = g0_base + static_cast<size_t>(gy0) * 256u * 4u;
            const uint8_t* src1 = g0_base + static_cast<size_t>(gy1) * 256u * 4u;
            for (int x = 0; x < 256; ++x) {
                g0_r0[x] = g0_u8_to_f32[src0[x * 4 + 0]];
                g0_r1[x] = g0_u8_to_f32[src0[x * 4 + 1]];
                g0_r2[x] = g0_u8_to_f32[src0[x * 4 + 2]];
                g0_r3[x] = g0_u8_to_f32[src0[x * 4 + 3]];
                g0_s0[x] = g0_u8_to_f32[src1[x * 4 + 0]];
                g0_s1[x] = g0_u8_to_f32[src1[x * 4 + 1]];
                g0_s2[x] = g0_u8_to_f32[src1[x * 4 + 2]];
                g0_s3[x] = g0_u8_to_f32[src1[x * 4 + 3]];
            }

            // 2. Decode G1 row y0 and y1 (64 RGBA16F -> planar FP32) via F16C
            int hy0 = g1_lut.i0[qy][iy];
            int hy1 = g1_lut.i1[qy][iy];
            const uint16_t* hsrc0 = g1_base + static_cast<size_t>(hy0) * 64u * 4u;
            const uint16_t* hsrc1 = g1_base + static_cast<size_t>(hy1) * 64u * 4u;
            for (int x = 0; x < 64; x += 2) {
                __m128i ph0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(hsrc0 + x * 4));
                __m128i ph1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(hsrc1 + x * 4));
                __m256 f0 = _mm256_cvtph_ps(ph0);
                __m256 f1 = _mm256_cvtph_ps(ph1);
                alignas(32) float t0[8], t1[8];
                _mm256_store_ps(t0, f0);
                _mm256_store_ps(t1, f1);
                g1_r0[x + 0] = t0[0]; g1_r1[x + 0] = t0[1]; g1_r2[x + 0] = t0[2]; g1_r3[x + 0] = t0[3];
                g1_r0[x + 1] = t0[4]; g1_r1[x + 1] = t0[5]; g1_r2[x + 1] = t0[6]; g1_r3[x + 1] = t0[7];
                g1_s0[x + 0] = t1[0]; g1_s1[x + 0] = t1[1]; g1_s2[x + 0] = t1[2]; g1_s3[x + 0] = t1[3];
                g1_s0[x + 1] = t1[4]; g1_s1[x + 1] = t1[5]; g1_s2[x + 1] = t1[6]; g1_s3[x + 1] = t1[7];
            }
            (void)v_g0_x0;

            __m256 gwy0 = _mm256_set1_ps(g0_lut.f0[qy][iy]);
            __m256 gwy1 = _mm256_set1_ps(g0_lut.f1[qy][iy]);
            __m256 hwy0 = _mm256_set1_ps(g1_lut.f0[qy][iy]);
            __m256 hwy1 = _mm256_set1_ps(g1_lut.f1[qy][iy]);

            const float* g0_rows0[4] = {g0_r0, g0_r1, g0_r2, g0_r3};
            const float* g0_rows1[4] = {g0_s0, g0_s1, g0_s2, g0_s3};
            const float* g1_rows0[4] = {g1_r0, g1_r1, g1_r2, g1_r3};
            const float* g1_rows1[4] = {g1_s0, g1_s1, g1_s2, g1_s3};

            size_t tile_base = static_cast<size_t>(b) * kTileElems;
            size_t row_off = static_cast<size_t>(iy) * kTileDim;

            if (cur_is_fp16) {
                uint16_t* dst_tile = cur_out_fp16 + tile_base;
                for (int ix = 0; ix < kTileDim; ix += 8) {
                    __m256i gx0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(g0_lut.i0[qx] + ix));
                    __m256i gx1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(g0_lut.i1[qx] + ix));
                    __m256 gwx0 = _mm256_loadu_ps(g0_lut.f0[qx] + ix);
                    __m256 gwx1 = _mm256_loadu_ps(g0_lut.f1[qx] + ix);

                    for (int c = 0; c < 4; ++c) {
                        __m256 v00 = _mm256_i32gather_ps(g0_rows0[c], gx0, 4);
                        __m256 v10 = _mm256_i32gather_ps(g0_rows0[c], gx1, 4);
                        __m256 v01 = _mm256_i32gather_ps(g0_rows1[c], gx0, 4);
                        __m256 v11 = _mm256_i32gather_ps(g0_rows1[c], gx1, 4);
                        __m256 top = _mm256_fmadd_ps(v10, gwx1, _mm256_mul_ps(v00, gwx0));
                        __m256 bot = _mm256_fmadd_ps(v11, gwx1, _mm256_mul_ps(v01, gwx0));
                        __m256 val = _mm256_fmadd_ps(bot, gwy1, _mm256_mul_ps(top, gwy0));
                        __m128i h8 = _mm256_cvtps_ph(val, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst_tile + static_cast<size_t>(c) * kPlaneElems + row_off + ix), h8);
                    }

                    __m256i hx0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(g1_lut.i0[qx] + ix));
                    __m256i hx1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(g1_lut.i1[qx] + ix));
                    __m256 hwx0 = _mm256_loadu_ps(g1_lut.f0[qx] + ix);
                    __m256 hwx1 = _mm256_loadu_ps(g1_lut.f1[qx] + ix);

                    for (int c = 0; c < 4; ++c) {
                        __m256 v00 = _mm256_i32gather_ps(g1_rows0[c], hx0, 4);
                        __m256 v10 = _mm256_i32gather_ps(g1_rows0[c], hx1, 4);
                        __m256 v01 = _mm256_i32gather_ps(g1_rows1[c], hx0, 4);
                        __m256 v11 = _mm256_i32gather_ps(g1_rows1[c], hx1, 4);
                        __m256 top = _mm256_fmadd_ps(v10, hwx1, _mm256_mul_ps(v00, hwx0));
                        __m256 bot = _mm256_fmadd_ps(v11, hwx1, _mm256_mul_ps(v01, hwx0));
                        __m256 val = _mm256_fmadd_ps(bot, hwy1, _mm256_mul_ps(top, hwy0));
                        __m128i h8 = _mm256_cvtps_ph(val, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst_tile + static_cast<size_t>(4 + c) * kPlaneElems + row_off + ix), h8);
                    }
                }
                for (int c = 0; c < 8; ++c) {
                    std::memcpy(
                        dst_tile + static_cast<size_t>(8 + c) * kPlaneElems + row_off,
                        pe_lut_fp16 + static_cast<size_t>(c) * kPlaneElems + row_off,
                        static_cast<size_t>(kTileDim) * sizeof(uint16_t)
                    );
                }
            } else {
                float* dst_tile = cur_out_fp32 + tile_base;
                for (int ix = 0; ix < kTileDim; ix += 8) {
                    __m256i gx0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(g0_lut.i0[qx] + ix));
                    __m256i gx1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(g0_lut.i1[qx] + ix));
                    __m256 gwx0 = _mm256_loadu_ps(g0_lut.f0[qx] + ix);
                    __m256 gwx1 = _mm256_loadu_ps(g0_lut.f1[qx] + ix);

                    for (int c = 0; c < 4; ++c) {
                        __m256 v00 = _mm256_i32gather_ps(g0_rows0[c], gx0, 4);
                        __m256 v10 = _mm256_i32gather_ps(g0_rows0[c], gx1, 4);
                        __m256 v01 = _mm256_i32gather_ps(g0_rows1[c], gx0, 4);
                        __m256 v11 = _mm256_i32gather_ps(g0_rows1[c], gx1, 4);
                        __m256 top = _mm256_fmadd_ps(v10, gwx1, _mm256_mul_ps(v00, gwx0));
                        __m256 bot = _mm256_fmadd_ps(v11, gwx1, _mm256_mul_ps(v01, gwx0));
                        __m256 val = _mm256_fmadd_ps(bot, gwy1, _mm256_mul_ps(top, gwy0));
                        _mm256_storeu_ps(dst_tile + static_cast<size_t>(c) * kPlaneElems + row_off + ix, val);
                    }

                    __m256i hx0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(g1_lut.i0[qx] + ix));
                    __m256i hx1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(g1_lut.i1[qx] + ix));
                    __m256 hwx0 = _mm256_loadu_ps(g1_lut.f0[qx] + ix);
                    __m256 hwx1 = _mm256_loadu_ps(g1_lut.f1[qx] + ix);

                    for (int c = 0; c < 4; ++c) {
                        __m256 v00 = _mm256_i32gather_ps(g1_rows0[c], hx0, 4);
                        __m256 v10 = _mm256_i32gather_ps(g1_rows0[c], hx1, 4);
                        __m256 v01 = _mm256_i32gather_ps(g1_rows1[c], hx0, 4);
                        __m256 v11 = _mm256_i32gather_ps(g1_rows1[c], hx1, 4);
                        __m256 top = _mm256_fmadd_ps(v10, hwx1, _mm256_mul_ps(v00, hwx0));
                        __m256 bot = _mm256_fmadd_ps(v11, hwx1, _mm256_mul_ps(v01, hwx0));
                        __m256 val = _mm256_fmadd_ps(bot, hwy1, _mm256_mul_ps(top, hwy0));
                        _mm256_storeu_ps(dst_tile + static_cast<size_t>(4 + c) * kPlaneElems + row_off + ix, val);
                    }
                }
                for (int c = 0; c < 8; ++c) {
                    std::memcpy(
                        dst_tile + static_cast<size_t>(8 + c) * kPlaneElems + row_off,
                        pe_lut_fp32 + static_cast<size_t>(c) * kPlaneElems + row_off,
                        static_cast<size_t>(kTileDim) * sizeof(float)
                    );
                }
            }
        }
    }

    static DWORD WINAPI worker_entry(LPVOID param) {
        reinterpret_cast<GatherEngine*>(param)->worker_loop();
        return 0;
    }

    void worker_loop() {
        uint64_t seen_gen = 0;
        while (true) {
            AcquireSRWLockExclusive(&mtx);
            while (!stop_pool && generation == seen_gen) {
                SleepConditionVariableSRW(&cv_work, &mtx, INFINITE, 0);
            }
            bool do_stop = stop_pool;
            seen_gen = generation;
            ReleaseSRWLockExclusive(&mtx);
            if (do_stop) return;

            while (true) {
                int j = next_job.fetch_add(1, std::memory_order_relaxed);
                if (j >= active_jobs) break;
                process_strip(j);
            }
            if (finished_workers.fetch_add(1, std::memory_order_acq_rel) + 1 == static_cast<int>(workers.size())) {
                AcquireSRWLockExclusive(&mtx);
                WakeConditionVariable(&cv_done);
                ReleaseSRWLockExclusive(&mtx);
            }
        }
    }

    void run_parallel_gather() {
        int total_jobs = cur_batch * kStripsPerTile;
        if (workers.empty() || total_jobs <= 16) {
            for (int j = 0; j < total_jobs; ++j) process_strip(j);
            return;
        }
        AcquireSRWLockExclusive(&mtx);
        active_jobs = total_jobs;
        next_job.store(0, std::memory_order_relaxed);
        finished_workers.store(0, std::memory_order_relaxed);
        ++generation;
        WakeAllConditionVariable(&cv_work);
        while (finished_workers.load(std::memory_order_acquire) < static_cast<int>(workers.size())) {
            SleepConditionVariableSRW(&cv_done, &mtx, INFINITE, 0);
        }
        ReleaseSRWLockExclusive(&mtx);
    }
};

GatherEngine g_engine;

} // namespace

extern "C" {

__declspec(dllexport) int pahnts_gather_init(int max_batch, int num_threads) {
    if (max_batch <= 0) max_batch = 64;
    if (num_threads <= 0) {
        SYSTEM_INFO si{};
        GetSystemInfo(&si);
        int hc = static_cast<int>(si.dwNumberOfProcessors);
        num_threads = std::max(1, std::min(12, hc > 2 ? hc - 2 : hc));
    }

    if (g_engine.initialized) {
        AcquireSRWLockExclusive(&g_engine.mtx);
        g_engine.stop_pool = true;
        WakeAllConditionVariable(&g_engine.cv_work);
        ReleaseSRWLockExclusive(&g_engine.mtx);
        if (!g_engine.workers.empty()) {
            WaitForMultipleObjects(static_cast<DWORD>(g_engine.workers.size()), g_engine.workers.data(), TRUE, INFINITE);
            for (HANDLE h : g_engine.workers) CloseHandle(h);
        }
        g_engine.workers.clear();
        _aligned_free(g_engine.persistent_fp32);
        _aligned_free(g_engine.persistent_fp16);
        _aligned_free(g_engine.pe_lut_fp32);
        _aligned_free(g_engine.pe_lut_fp16);
        g_engine.persistent_fp32 = nullptr;
        g_engine.persistent_fp16 = nullptr;
        g_engine.pe_lut_fp32 = nullptr;
        g_engine.pe_lut_fp16 = nullptr;
        g_engine.stop_pool = false;
        g_engine.initialized = false;
    }

    size_t elems = static_cast<size_t>(max_batch) * kTileElems;
    g_engine.persistent_fp32 = static_cast<float*>(_aligned_malloc(elems * sizeof(float), 64));
    g_engine.persistent_fp16 = static_cast<uint16_t*>(_aligned_malloc(elems * sizeof(uint16_t), 64));
    g_engine.pe_lut_fp32 = static_cast<float*>(_aligned_malloc(8ULL * kPlaneElems * sizeof(float), 64));
    g_engine.pe_lut_fp16 = static_cast<uint16_t*>(_aligned_malloc(8ULL * kPlaneElems * sizeof(uint16_t), 64));
    if (!g_engine.persistent_fp32 || !g_engine.persistent_fp16 || !g_engine.pe_lut_fp32 || !g_engine.pe_lut_fp16) {
        return -1;
    }

    g_engine.max_batch = max_batch;
    g_engine.num_threads = num_threads;
    g_engine.init_luts();

    for (int t = 0; t < num_threads; ++t) {
        HANDLE h = CreateThread(nullptr, 0, &GatherEngine::worker_entry, &g_engine, 0, nullptr);
        if (h) g_engine.workers.push_back(h);
    }
    g_engine.initialized = true;
    return 0;
}

__declspec(dllexport) void pahnts_gather_register_material(int mat_id, const uint8_t* g0_rgba8, const uint16_t* g1_rgba16f) {
    if (mat_id >= 0 && mat_id < kMaxMaterials) {
        g_engine.reg_g0[mat_id] = g0_rgba8;
        g_engine.reg_g1[mat_id] = g1_rgba16f;
    }
}

__declspec(dllexport) void pahnts_gather_batch_fp16(
    int batch_size, const int* mat_ids, const int* tile_xs, const int* tile_ys, uint16_t* out_override
) {
    if (!g_engine.initialized || batch_size <= 0 || batch_size > g_engine.max_batch) return;
    g_engine.cur_is_fp16 = true;
    g_engine.cur_batch = batch_size;
    g_engine.cur_mat_ids = mat_ids;
    g_engine.cur_g0_ptrs = nullptr;
    g_engine.cur_g1_ptrs = nullptr;
    g_engine.cur_txs = tile_xs;
    g_engine.cur_tys = tile_ys;
    g_engine.cur_out_fp16 = out_override ? out_override : g_engine.persistent_fp16;
    g_engine.run_parallel_gather();
}

__declspec(dllexport) void pahnts_gather_batch_fp32(
    int batch_size, const int* mat_ids, const int* tile_xs, const int* tile_ys, float* out_override
) {
    if (!g_engine.initialized || batch_size <= 0 || batch_size > g_engine.max_batch) return;
    g_engine.cur_is_fp16 = false;
    g_engine.cur_batch = batch_size;
    g_engine.cur_mat_ids = mat_ids;
    g_engine.cur_g0_ptrs = nullptr;
    g_engine.cur_g1_ptrs = nullptr;
    g_engine.cur_txs = tile_xs;
    g_engine.cur_tys = tile_ys;
    g_engine.cur_out_fp32 = out_override ? out_override : g_engine.persistent_fp32;
    g_engine.run_parallel_gather();
}

__declspec(dllexport) float* pahnts_gather_get_fp32_ptr() {
    return g_engine.persistent_fp32;
}

__declspec(dllexport) uint16_t* pahnts_gather_get_fp16_ptr() {
    return g_engine.persistent_fp16;
}

__declspec(dllexport) uint64_t pahnts_gather_ram_bytes() {
    if (!g_engine.initialized) return 0;
    size_t elems = static_cast<size_t>(g_engine.max_batch) * kTileElems;
    return static_cast<uint64_t>(
        elems * (sizeof(float) + sizeof(uint16_t)) +
        8ULL * kPlaneElems * (sizeof(float) + sizeof(uint16_t))
    );
}

__declspec(dllexport) void pahnts_gather_shutdown() {
    if (!g_engine.initialized) return;
    AcquireSRWLockExclusive(&g_engine.mtx);
    g_engine.stop_pool = true;
    WakeAllConditionVariable(&g_engine.cv_work);
    ReleaseSRWLockExclusive(&g_engine.mtx);
    if (!g_engine.workers.empty()) {
        WaitForMultipleObjects(static_cast<DWORD>(g_engine.workers.size()), g_engine.workers.data(), TRUE, INFINITE);
        for (HANDLE h : g_engine.workers) CloseHandle(h);
    }
    g_engine.workers.clear();
    _aligned_free(g_engine.persistent_fp32);
    _aligned_free(g_engine.persistent_fp16);
    _aligned_free(g_engine.pe_lut_fp32);
    _aligned_free(g_engine.pe_lut_fp16);
    g_engine.persistent_fp32 = nullptr;
    g_engine.persistent_fp16 = nullptr;
    g_engine.pe_lut_fp32 = nullptr;
    g_engine.pe_lut_fp16 = nullptr;
    g_engine.stop_pool = false;
    g_engine.initialized = false;
}

} // extern "C"
