#!/usr/bin/env python3
"""Generate 32 seamless 512x512 PBR base materials (virtual 8192x8192 via 16x16 tiling),
train batched CUDA NTC latent pyramids (G0 256x256 uint8, G1 64x64 fp16) + FP16 MLPs,
and pack bit-exact BC7 Mode 6 / BC5_UNORM mip pyramids into results/assets_cache/pbr_ntc_pack_v1.bin.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "assets_cache"
PACK_PATH = OUT_DIR / "pbr_ntc_pack_v1.bin"
MANIFEST_PATH = OUT_DIR / "manifest.json"


def make_grid(size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    lin = torch.linspace(0.0, 1.0 - 1.0 / size, size, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")
    return xx, yy


def periodic_fbm(
    xx: torch.Tensor,
    yy: torch.Tensor,
    seed_offset: int,
    octaves: int = 5,
    base_freq: int = 4,
) -> torch.Tensor:
    """Deterministic seamless periodic multi-octave noise on [0,1)^2."""
    gen = torch.Generator(device=xx.device)
    gen.manual_seed(1000 + seed_offset)
    acc = torch.zeros_like(xx)
    amp = 0.5
    norm = 0.0
    for oct_i in range(octaves):
        freq = base_freq * (2 ** oct_i)
        # Sum 4 oriented plane waves per octave for rich anisotropic detail
        oct_val = torch.zeros_like(xx)
        for _ in range(4):
            kx = int(torch.randint(-freq, freq + 1, (1,), generator=gen, device=xx.device).item())
            ky = int(torch.randint(-freq, freq + 1, (1,), generator=gen, device=xx.device).item())
            if kx == 0 and ky == 0:
                kx = freq
            phase = float(torch.rand((1,), generator=gen, device=xx.device).item() * 2.0 * math.pi)
            oct_val = oct_val + torch.sin(2.0 * math.pi * (kx * xx + ky * yy) + phase)
        acc = acc + amp * (oct_val * 0.25)
        norm += amp
        amp *= 0.52
    return (acc / max(norm, 1e-6)) * 0.5 + 0.5


def generate_32_pbr_materials(device: torch.device, size: int = 512) -> torch.Tensor:
    """Synthesize [32, 8, 512, 512] float32 seamless PBR maps in [0, 1].
    Channels: 0:R, 1:G, 2:B, 3:AO, 4:Nx_enc, 5:Ny_enc, 6:Roughness, 7:Metallic
    """
    xx, yy = make_grid(size, device)
    mats = torch.zeros((32, 8, size, size), device=device, dtype=torch.float32)

    palettes = [
        # (colA, colB, base_rough, base_metal, pattern_kind, scale)
        ((0.58, 0.24, 0.16), (0.36, 0.34, 0.31), 0.78, 0.00, "brick", 8.0),
        ((0.68, 0.65, 0.59), (0.48, 0.45, 0.40), 0.72, 0.00, "ashlar", 6.0),
        ((0.84, 0.85, 0.82), (0.28, 0.34, 0.38), 0.28, 0.05, "hex", 10.0),
        ((0.46, 0.30, 0.18), (0.31, 0.19, 0.10), 0.48, 0.00, "parquet", 8.0),
        ((0.42, 0.44, 0.46), (0.25, 0.27, 0.30), 0.80, 0.00, "cobble", 8.0),
        ((0.24, 0.27, 0.31), (0.14, 0.16, 0.19), 0.55, 0.02, "slate", 6.0),
        ((0.72, 0.38, 0.22), (0.50, 0.26, 0.15), 0.66, 0.00, "checker", 8.0),
        ((0.78, 0.75, 0.70), (0.45, 0.42, 0.38), 0.38, 0.00, "terrazzo", 12.0),

        ((0.70, 0.73, 0.77), (0.44, 0.47, 0.52), 0.26, 0.94, "brushed", 16.0),
        ((0.36, 0.40, 0.45), (0.22, 0.25, 0.29), 0.38, 0.88, "panel", 4.0),
        ((0.54, 0.56, 0.58), (0.30, 0.32, 0.34), 0.34, 0.92, "diamond", 16.0),
        ((0.76, 0.44, 0.26), (0.24, 0.58, 0.52), 0.42, 0.76, "patina", 8.0),
        ((0.64, 0.67, 0.70), (0.22, 0.24, 0.26), 0.30, 0.90, "louver", 12.0),
        ((0.48, 0.42, 0.36), (0.28, 0.20, 0.15), 0.62, 0.65, "corrugated", 14.0),
        ((0.83, 0.68, 0.32), (0.50, 0.38, 0.16), 0.24, 0.95, "filigree", 8.0),
        ((0.18, 0.19, 0.21), (0.09, 0.10, 0.12), 0.32, 0.45, "weave", 24.0),

        ((0.88, 0.86, 0.82), (0.38, 0.36, 0.34), 0.20, 0.00, "marble", 6.0),
        ((0.76, 0.69, 0.57), (0.56, 0.48, 0.37), 0.58, 0.00, "travertine", 8.0),
        ((0.55, 0.37, 0.22), (0.34, 0.21, 0.11), 0.52, 0.00, "wood", 6.0),
        ((0.52, 0.53, 0.52), (0.37, 0.38, 0.37), 0.84, 0.00, "concrete", 4.0),
        ((0.62, 0.43, 0.24), (0.36, 0.23, 0.12), 0.35, 0.86, "hammered", 14.0),
        ((0.46, 0.44, 0.42), (0.18, 0.17, 0.16), 0.48, 0.04, "granite", 16.0),
        ((0.28, 0.36, 0.48), (0.17, 0.23, 0.32), 0.86, 0.00, "weave", 20.0),
        ((0.80, 0.78, 0.74), (0.62, 0.60, 0.56), 0.70, 0.00, "fluted", 12.0),

        ((0.20, 0.23, 0.27), (0.08, 0.58, 0.72), 0.30, 0.72, "hex", 12.0),
        ((0.12, 0.15, 0.18), (0.18, 0.75, 0.54), 0.26, 0.80, "circuit", 8.0),
        ((0.82, 0.64, 0.12), (0.15, 0.16, 0.18), 0.46, 0.50, "hazard", 8.0),
        ((0.40, 0.43, 0.47), (0.11, 0.12, 0.14), 0.36, 0.88, "perf", 16.0),
        ((0.52, 0.56, 0.62), (0.26, 0.30, 0.36), 0.29, 0.92, "panel", 8.0),
        ((0.30, 0.32, 0.27), (0.16, 0.18, 0.14), 0.64, 0.12, "hex", 16.0),
        ((0.15, 0.15, 0.16), (0.07, 0.07, 0.08), 0.74, 0.05, "ribbed", 18.0),
        ((0.26, 0.22, 0.20), (0.86, 0.68, 0.28), 0.25, 0.90, "filigree", 6.0),
    ]

    for m_idx, (colA, colB, base_r, base_m, kind, sc) in enumerate(palettes):
        n_fine = periodic_fbm(xx, yy, m_idx * 17 + 1, octaves=5, base_freq=6)
        n_coarse = periodic_fbm(xx, yy, m_idx * 17 + 7, octaves=3, base_freq=2)

        # Build structural height / mask in [0, 1]
        if kind == "brick":
            row = torch.floor(yy * sc)
            ux = (xx * sc + 0.5 * (row % 2)) % 1.0
            uy = (yy * sc) % 1.0
            mortar = ((ux < 0.06) | (ux > 0.94) | (uy < 0.09) | (uy > 0.91)).float()
            height = (1.0 - mortar) * (0.55 + 0.45 * n_fine)
            mask = mortar
        elif kind == "ashlar":
            ux = (xx * sc) % 1.0
            uy = (yy * sc * 0.5) % 1.0
            grout = ((ux < 0.04) | (uy < 0.05)).float()
            height = (1.0 - grout) * (0.65 + 0.35 * n_fine)
            mask = grout
        elif kind in ("hex", "perf"):
            gx = torch.abs(((xx * sc) % 1.0) - 0.5) * 2.0
            gy = torch.abs(((yy * sc) % 1.0) - 0.5) * 2.0
            dist = torch.maximum(gx * 0.866 + gy * 0.5, gy) if kind == "hex" else torch.sqrt(gx * gx + gy * gy)
            edge = torch.clamp((dist - 0.78) * 8.0, 0.0, 1.0)
            height = (1.0 - edge) * 0.75 + 0.25 * n_fine
            mask = edge
        elif kind == "parquet":
            cx = (xx * sc) % 1.0
            cy = (yy * sc) % 1.0
            stripe = torch.sin(2.0 * math.pi * xx * sc * 4.0) * 0.5 + 0.5
            grout = ((cx < 0.03) | (cy < 0.03)).float()
            height = (1.0 - grout) * (0.6 + 0.2 * stripe + 0.2 * n_fine)
            mask = torch.clamp(grout + 0.4 * stripe, 0.0, 1.0)
        elif kind == "cobble":
            dx = ((xx * sc) % 1.0) - 0.5
            dy = ((yy * sc) % 1.0) - 0.5
            r = torch.sqrt(dx * dx + dy * dy)
            stone = torch.clamp(1.0 - r * 2.1, 0.0, 1.0)
            height = stone * 0.8 + 0.2 * n_fine
            mask = 1.0 - stone
        elif kind == "slate":
            bands = torch.sin(2.0 * math.pi * (xx * 3.0 + yy * 2.0 + n_coarse * 1.6)) * 0.5 + 0.5
            height = 0.5 * bands + 0.5 * n_fine
            mask = bands
        elif kind == "checker":
            c = ((torch.floor(xx * sc) + torch.floor(yy * sc)) % 2.0)
            grout = (((xx * sc) % 1.0 < 0.04) | ((yy * sc) % 1.0 < 0.04)).float()
            height = (1.0 - grout) * (0.7 + 0.3 * n_fine)
            mask = torch.clamp(c * 0.8 + grout, 0.0, 1.0)
        elif kind == "terrazzo":
            speck = (n_fine > 0.62).float()
            height = 0.7 + 0.15 * speck + 0.15 * n_coarse
            mask = speck
        elif kind == "brushed":
            aniso = periodic_fbm(xx * 0.25, yy * 4.0, m_idx * 17 + 3, octaves=4, base_freq=8)
            height = 0.6 * aniso + 0.4 * n_fine
            mask = aniso
        elif kind == "panel":
            ux = (xx * sc) % 1.0
            uy = (yy * sc) % 1.0
            seam = ((ux < 0.035) | (uy < 0.035)).float()
            rivet = ((((ux - 0.12) ** 2 + (uy - 0.12) ** 2) < 0.0012) | (((ux - 0.88) ** 2 + (uy - 0.88) ** 2) < 0.0012)).float()
            height = (1.0 - seam) * 0.65 + rivet * 0.25 + 0.1 * n_fine
            mask = torch.clamp(seam + 0.6 * rivet, 0.0, 1.0)
        elif kind == "diamond":
            ux = ((xx * sc) % 1.0) - 0.5
            uy = ((yy * sc) % 1.0) - 0.5
            d1 = torch.exp(-((ux * 3.0) ** 2 + (uy * 9.0) ** 2) * 8.0)
            d2 = torch.exp(-((uy * 3.0) ** 2 + (ux * 9.0) ** 2) * 8.0)
            bump = torch.clamp(d1 + d2, 0.0, 1.0)
            height = 0.3 + 0.6 * bump + 0.1 * n_fine
            mask = bump
        elif kind == "patina":
            mask = torch.clamp((n_coarse * 0.6 + n_fine * 0.4 - 0.35) * 2.2, 0.0, 1.0)
            height = 0.5 * n_fine + 0.5 * mask
        elif kind in ("louver", "corrugated", "fluted", "ribbed"):
            wave = torch.sin(2.0 * math.pi * xx * sc) * 0.5 + 0.5
            height = 0.78 * wave + 0.22 * n_fine
            mask = wave
        elif kind == "filigree":
            pat = torch.abs(torch.sin(2.0 * math.pi * xx * sc) * torch.cos(2.0 * math.pi * yy * sc))
            ring = torch.clamp((pat - 0.38) * 5.0, 0.0, 1.0)
            height = 0.3 + 0.55 * ring + 0.15 * n_fine
            mask = ring
        elif kind == "weave":
            wx = torch.sin(2.0 * math.pi * xx * sc)
            wy = torch.sin(2.0 * math.pi * yy * sc)
            w = 0.5 + 0.5 * torch.where(((torch.floor(xx * sc) + torch.floor(yy * sc)) % 2.0) > 0.5, wx, wy)
            height = 0.75 * w + 0.25 * n_fine
            mask = w
        elif kind == "marble":
            vein = torch.abs(torch.sin(2.0 * math.pi * (xx * 2.0 + yy * 3.0 + n_coarse * 2.5)))
            vein = torch.pow(1.0 - vein, 4.0)
            height = 0.8 - 0.2 * vein + 0.1 * n_fine
            mask = vein
        elif kind == "travertine":
            streaks = periodic_fbm(xx * 0.5, yy * 2.5, m_idx * 17 + 9, octaves=4, base_freq=6)
            pits = (n_fine < 0.26).float()
            height = (1.0 - 0.45 * pits) * (0.6 + 0.4 * streaks)
            mask = torch.clamp(streaks * 0.7 + pits * 0.5, 0.0, 1.0)
        elif kind == "wood":
            grain = torch.sin(2.0 * math.pi * (xx * 18.0 + n_coarse * 2.4)) * 0.5 + 0.5
            plank = ((yy * sc) % 1.0 < 0.03).float()
            height = (1.0 - plank) * (0.65 + 0.2 * grain + 0.15 * n_fine)
            mask = torch.clamp(grain * 0.7 + plank, 0.0, 1.0)
        elif kind == "concrete":
            tie = ((((xx * sc) % 1.0 - 0.5) ** 2 + ((yy * sc) % 1.0 - 0.5) ** 2) < 0.0025).float()
            height = (1.0 - 0.5 * tie) * (0.45 + 0.55 * n_fine)
            mask = torch.clamp(n_coarse * 0.7 + tie, 0.0, 1.0)
        elif kind == "hammered":
            cell = torch.sin(2.0 * math.pi * xx * sc) * torch.sin(2.0 * math.pi * yy * sc)
            height = 0.5 + 0.35 * cell + 0.15 * n_fine
            mask = cell * 0.5 + 0.5
        elif kind == "granite":
            mask = torch.clamp((n_fine - 0.35) * 2.5, 0.0, 1.0)
            height = 0.4 + 0.6 * n_fine
        elif kind == "circuit":
            ux = (xx * sc) % 1.0
            uy = (yy * sc) % 1.0
            trace = ((torch.abs(ux - 0.25) < 0.03) | (torch.abs(uy - 0.65) < 0.03) | (torch.abs(ux - uy) < 0.035)).float()
            height = 0.4 + 0.45 * trace + 0.15 * n_fine
            mask = trace
        elif kind == "hazard":
            stripe = (torch.sin(2.0 * math.pi * (xx + yy) * sc) > 0.0).float()
            height = 0.5 + 0.2 * stripe + 0.3 * n_fine
            mask = stripe
        else:
            height = n_fine
            mask = n_coarse

        # Albedo RGB
        cA = torch.tensor(colA, device=device, dtype=torch.float32).view(3, 1, 1)
        cB = torch.tensor(colB, device=device, dtype=torch.float32).view(3, 1, 1)
        albedo = cA * (1.0 - mask.unsqueeze(0)) + cB * mask.unsqueeze(0)
        albedo = torch.clamp(albedo * (0.88 + 0.24 * n_fine.unsqueeze(0)), 0.02, 0.98)

        # Tangent-space normal via periodic central differences
        dx = (torch.roll(height, -1, dims=1) - torch.roll(height, 1, dims=1)) * 2.4
        dy = (torch.roll(height, -1, dims=0) - torch.roll(height, 1, dims=0)) * 2.4
        nz = torch.ones_like(height)
        inv_len = torch.rsqrt(dx * dx + dy * dy + nz * nz)
        nx = -dx * inv_len
        ny = -dy * inv_len

        ao = torch.clamp(0.45 + 0.55 * height, 0.15, 1.0)
        rough = torch.clamp(base_r + 0.22 * (n_fine - 0.5) + 0.15 * mask, 0.06, 0.98)
        metal = torch.clamp(base_m - 0.35 * mask * (1.0 if base_m > 0.5 else 0.0) + 0.35 * mask * (1.0 if kind in ("circuit", "filigree") else 0.0), 0.0, 1.0)

        mats[m_idx, 0:3] = albedo
        mats[m_idx, 3] = ao
        mats[m_idx, 4] = torch.clamp(nx * 0.5 + 0.5, 0.0, 1.0)
        mats[m_idx, 5] = torch.clamp(ny * 0.5 + 0.5, 0.0, 1.0)
        mats[m_idx, 6] = rough
        mats[m_idx, 7] = metal

    return mats


def sample_periodic_bilinear(grid: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Exact wrap-around bilinear sampling matching D3D12 D3D12_TEXTURE_ADDRESS_MODE_WRAP.
    grid: [B, C, H, W]
    uv:   [B, N, 2] in [0, 1)
    returns [B, N, C]
    """
    b, c, h, w = grid.shape
    u = torch.remainder(uv[..., 0], 1.0) * w - 0.5
    v = torch.remainder(uv[..., 1], 1.0) * h - 0.5

    x0 = torch.floor(u).long()
    y0 = torch.floor(v).long()
    fx = (u - x0.float()).unsqueeze(-1)
    fy = (v - y0.float()).unsqueeze(-1)

    x0 = torch.remainder(x0, w)
    y0 = torch.remainder(y0, h)
    x1 = torch.remainder(x0 + 1, w)
    y1 = torch.remainder(y0 + 1, h)

    # [B, C, H*W] -> gather
    flat = grid.view(b, c, h * w).permute(0, 2, 1)  # [B, H*W, C]
    idx00 = (y0 * w + x0).unsqueeze(-1).expand(-1, -1, c)
    idx10 = (y0 * w + x1).unsqueeze(-1).expand(-1, -1, c)
    idx01 = (y1 * w + x0).unsqueeze(-1).expand(-1, -1, c)
    idx11 = (y1 * w + x1).unsqueeze(-1).expand(-1, -1, c)

    v00 = torch.gather(flat, 1, idx00)
    v10 = torch.gather(flat, 1, idx10)
    v01 = torch.gather(flat, 1, idx01)
    v11 = torch.gather(flat, 1, idx11)

    top = v00 * (1.0 - fx) + v10 * fx
    bot = v01 * (1.0 - fx) + v11 * fx
    return top * (1.0 - fy) + bot * fy


def build_pos_encoding(uv: torch.Tensor, lod_atten: torch.Tensor | float = 1.0) -> torch.Tensor:
    """8-dim Fourier positional encoding matching HLSL pbr_ntc_sample & ntc_decode_bcn."""
    u = uv[..., 0:1]
    v = uv[..., 1:2]
    two_pi = 2.0 * math.pi
    pe = torch.cat(
        [
            torch.sin(two_pi * 8.0 * u),
            torch.cos(two_pi * 8.0 * u),
            torch.sin(two_pi * 8.0 * v),
            torch.cos(two_pi * 8.0 * v),
            torch.sin(two_pi * 16.0 * u),
            torch.cos(two_pi * 16.0 * u),
            torch.sin(two_pi * 16.0 * v),
            torch.cos(two_pi * 16.0 * v),
        ],
        dim=-1,
    )
    return pe * lod_atten


def train_batched_ntc(
    mats: torch.Tensor, steps: int = 550, batch_pts: int = 16384
) -> dict[str, Any]:
    """Train 32 independent NTC (G0 + G1 + 16->32->32->8 MLP) models concurrently on CUDA."""
    device = mats.device
    num_mats = mats.shape[0]
    torch.manual_seed(42)

    g0_logits = torch.nn.Parameter(torch.zeros((num_mats, 4, 256, 256), device=device))
    g1_param = torch.nn.Parameter(torch.randn((num_mats, 4, 64, 64), device=device) * 0.1)

    w1 = torch.nn.Parameter(torch.randn((num_mats, 16, 32), device=device) * (1.0 / math.sqrt(16.0)))
    b1 = torch.nn.Parameter(torch.zeros((num_mats, 1, 32), device=device))
    w2 = torch.nn.Parameter(torch.randn((num_mats, 32, 32), device=device) * (1.0 / math.sqrt(32.0)))
    b2 = torch.nn.Parameter(torch.zeros((num_mats, 1, 32), device=device))
    w3 = torch.nn.Parameter(torch.randn((num_mats, 32, 8), device=device) * (1.0 / math.sqrt(32.0)))
    b3 = torch.nn.Parameter(torch.zeros((num_mats, 1, 8), device=device))

    opt = torch.optim.Adam([g0_logits, g1_param, w1, b1, w2, b2, w3, b3], lr=0.025)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=0.002)

    # Precompute mip pyramid of target mats [0..4] for multi-scale LOD-consistent training
    mats_mips = [mats]
    for _ in range(4):
        mats_mips.append(F.avg_pool2d(mats_mips[-1], kernel_size=2, stride=2))

    t0 = time.perf_counter()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)

        # Quantization-aware STE for G0 (uint8) and G1 (fp16)
        g0_soft = torch.sigmoid(g0_logits)
        g0_q = g0_soft + (torch.round(g0_soft * 255.0) / 255.0 - g0_soft).detach()
        g1_q = g1_param + (g1_param.half().float() - g1_param).detach()

        # Alternate fine mip 0 (60%) and coarser mips 1..4 (40%) so MLP is accurate across all LODs
        if step % 5 < 3:
            mip = 0
        else:
            mip = 1 + (step % 4)  # 1, 2, 3, 4

        g0_m = g0_q
        for _ in range(mip):
            g0_m = F.avg_pool2d(g0_m, kernel_size=2, stride=2)

        g1_m = g1_q
        for _ in range(max(0, mip - 2)):
            g1_m = F.avg_pool2d(g1_m, kernel_size=2, stride=2)

        lod_atten = max(0.0, 1.0 - float(mip) * 0.25)
        uv = torch.rand((num_mats, batch_pts, 2), device=device)

        g0_s = sample_periodic_bilinear(g0_m, uv)
        g1_s = sample_periodic_bilinear(g1_m, uv)
        pe = build_pos_encoding(uv, lod_atten)
        x = torch.cat([g0_s * 2.0 - 1.0, g1_s, pe], dim=-1)

        h1 = F.leaky_relu(torch.bmm(x, w1) + b1, negative_slope=0.125)
        h2 = F.leaky_relu(torch.bmm(h1, w2) + b2, negative_slope=0.125)
        pred = torch.sigmoid(torch.bmm(h2, w3) + b3)

        target = sample_periodic_bilinear(mats_mips[mip], uv)
        loss = F.mse_loss(pred, target) + 0.05 * F.l1_loss(pred, target)
        loss.backward()
        opt.step()
        scheduler.step()

    torch.cuda.synchronize()
    train_s = time.perf_counter() - t0

    # Evaluate exact quantized FP16/UINT8 reconstruction across all 512x512 texels
    with torch.no_grad():
        g0_u8 = torch.clamp(torch.round(torch.sigmoid(g0_logits) * 255.0), 0, 255).to(torch.uint8)
        g0_eval = g0_u8.float() / 255.0
        g1_f16 = g1_param.half()
        g1_eval = g1_f16.float()

        w1_f16, b1_f16 = w1.half(), b1.half()
        w2_f16, b2_f16 = w2.half(), b2.half()
        w3_f16, b3_f16 = w3.half(), b3.half()

        xx, yy = make_grid(512, device)
        # Pixel center UV: (x + 0.5) / 512
        u_grid = (xx + 0.5 / 512.0).reshape(-1)
        v_grid = (yy + 0.5 / 512.0).reshape(-1)
        full_uv = torch.stack([u_grid, v_grid], dim=-1).unsqueeze(0).expand(num_mats, -1, -1)

        preds = []
        chunk = 65536
        for start in range(0, full_uv.shape[1], chunk):
            uv_c = full_uv[:, start : start + chunk, :]
            g0_s = sample_periodic_bilinear(g0_eval, uv_c)
            g1_s = sample_periodic_bilinear(g1_eval, uv_c)
            pe = build_pos_encoding(uv_c, 1.0)
            x = torch.cat([g0_s * 2.0 - 1.0, g1_s, pe], dim=-1).half()
            h1 = F.leaky_relu(torch.bmm(x, w1_f16) + b1_f16, negative_slope=0.125)
            h2 = F.leaky_relu(torch.bmm(h1, w2_f16) + b2_f16, negative_slope=0.125)
            out = torch.sigmoid((torch.bmm(h2, w3_f16) + b3_f16).float())
            preds.append(out)
        pred_full = torch.cat(preds, dim=1).permute(0, 2, 1).view(num_mats, 8, 512, 512)

        mse_per_mat = torch.mean((pred_full - mats) ** 2, dim=(1, 2, 3))
        psnr_per_mat = (10.0 * torch.log10(1.0 / torch.clamp(mse_per_mat, min=1e-10))).cpu().numpy()

    return {
        "train_seconds": round(train_s, 3),
        "g0_u8": g0_u8.permute(0, 2, 3, 1).contiguous().cpu().numpy(),  # [32, 256, 256, 4] uint8
        "g1_f16": g1_f16.permute(0, 2, 3, 1).contiguous().cpu().numpy(),  # [32, 64, 64, 4] float16
        "w1_f16": w1_f16.cpu().numpy(),  # [32, 16, 32]
        "b1_f16": b1_f16.squeeze(1).cpu().numpy(),  # [32, 32]
        "w2_f16": w2_f16.cpu().numpy(),  # [32, 32, 32]
        "b2_f16": b2_f16.squeeze(1).cpu().numpy(),  # [32, 32]
        "w3_f16": w3_f16.cpu().numpy(),  # [32, 32, 8]
        "b3_f16": b3_f16.squeeze(1).cpu().numpy(),  # [32, 8]
        "psnr_per_mat": psnr_per_mat.tolist(),
    }


def encode_bc7_mode6_np(rgba_u8: np.ndarray) -> np.ndarray:
    """Vectorized bit-exact BC7 Mode 6 encoder.
    rgba_u8: [H, W, 4] uint8 (H, W divisible by 4)
    returns [H//4, W//4, 16] uint8
    """
    h, w, _ = rgba_u8.shape
    bh, bw = h // 4, w // 4
    blocks = rgba_u8.reshape(bh, 4, bw, 4, 4).transpose(0, 2, 1, 3, 4).reshape(-1, 16, 4).astype(np.float32)

    # PCA-like principal axis via max-min contrast channel per block, refined by projections
    bmin = blocks.min(axis=1)  # [N, 4]
    bmax = blocks.max(axis=1)  # [N, 4]
    span = bmax - bmin  # [N, 4]
    # Pick endpoint pair along dominant channel sign to preserve cross-channel correlation
    dom_ch = np.argmax(span, axis=1)  # [N]
    n_blocks = blocks.shape[0]
    vals_dom = blocks[np.arange(n_blocks), :, dom_ch]  # [N, 16]
    idx_min = np.argmin(vals_dom, axis=1)
    idx_max = np.argmax(vals_dom, axis=1)
    e0 = blocks[np.arange(n_blocks), idx_min, :]  # [N, 4]
    e1 = blocks[np.arange(n_blocks), idx_max, :]  # [N, 4]

    e0_u8 = np.clip(np.rint(e0), 0, 255).astype(np.uint64)
    e1_u8 = np.clip(np.rint(e1), 0, 255).astype(np.uint64)

    p0 = (((e0_u8[:, 0] & 1) + (e0_u8[:, 1] & 1) + (e0_u8[:, 2] & 1)) >= 2).astype(np.uint64)
    p1 = (((e1_u8[:, 0] & 1) + (e1_u8[:, 1] & 1) + (e1_u8[:, 2] & 1)) >= 2).astype(np.uint64)

    # Reconstruct exact 8-bit endpoints from (top7 << 1) | p
    rec0 = (((e0_u8 >> 1) << 1) | p0[:, None]).astype(np.float32)
    rec1 = (((e1_u8 >> 1) << 1) | p1[:, None]).astype(np.float32)
    diff = rec1 - rec0  # [N, 4]
    denom = np.sum(diff * diff, axis=1, keepdims=True)  # [N, 1]
    proj = np.sum((blocks - rec0[:, None, :]) * diff[:, None, :], axis=2) / np.maximum(denom, 1e-5)
    proj = np.where(denom < 1e-5, 0.0, proj)
    q = np.clip(np.rint(proj * 15.0), 0, 15).astype(np.uint64)  # [N, 16]

    # Enforce BC7 Mode 6 anchor constraint: MSB of q[:, 0] must be 0
    swap = q[:, 0] >= 8
    if np.any(swap):
        tmp_e = e0_u8[swap].copy()
        e0_u8[swap] = e1_u8[swap]
        e1_u8[swap] = tmp_e

        tmp_p = p0[swap].copy()
        p0[swap] = p1[swap]
        p1[swap] = tmp_p

        q[swap] = 15 - q[swap]

    lo64 = (
        np.uint64(0x40)
        | ((e0_u8[:, 0] >> np.uint64(1)) << np.uint64(7))
        | ((e1_u8[:, 0] >> np.uint64(1)) << np.uint64(14))
        | ((e0_u8[:, 1] >> np.uint64(1)) << np.uint64(21))
        | ((e1_u8[:, 1] >> np.uint64(1)) << np.uint64(28))
        | ((e0_u8[:, 2] >> np.uint64(1)) << np.uint64(35))
        | ((e1_u8[:, 2] >> np.uint64(1)) << np.uint64(42))
        | ((e0_u8[:, 3] >> np.uint64(1)) << np.uint64(49))
        | ((e1_u8[:, 3] >> np.uint64(1)) << np.uint64(56))
        | (p0 << np.uint64(63))
    )

    hi64 = p1 | ((q[:, 0] & np.uint64(0x7)) << np.uint64(1))
    for i in range(1, 16):
        hi64 = hi64 | ((q[:, i] & np.uint64(0xF)) << np.uint64(4 * i))

    out_u64 = np.stack([lo64, hi64], axis=1).astype("<u8")
    return out_u64.view(np.uint8).reshape(bh, bw, 16)


def encode_bc4_channel_np(ch_u8: np.ndarray) -> np.ndarray:
    """Vectorized bit-exact BC4_UNORM encoder for one channel [H, W] -> [H//4, W//4] uint64."""
    h, w = ch_u8.shape
    bh, bw = h // 4, w // 4
    blocks = ch_u8.reshape(bh, 4, bw, 4).transpose(0, 2, 1, 3).reshape(-1, 16).astype(np.float32)

    e0 = blocks.max(axis=1)
    e1 = blocks.min(axis=1)
    e0_u8 = np.clip(np.rint(e0), 0, 255).astype(np.uint64)
    e1_u8 = np.clip(np.rint(e1), 0, 255).astype(np.uint64)

    span = np.maximum(e0 - e1, 1.0)
    step = np.clip(np.rint((e0[:, None] - blocks) * 7.0 / span[:, None]), 0, 7).astype(np.uint64)
    idx = np.where(step == 0, np.uint64(0), np.where(step == 7, np.uint64(1), step + np.uint64(1)))
    idx = np.where((e0_u8 == e1_u8)[:, None], np.uint64(0), idx)

    word = e0_u8 | (e1_u8 << np.uint64(8))
    for i in range(16):
        word = word | ((idx[:, i] & np.uint64(0x7)) << np.uint64(16 + 3 * i))
    return word.astype("<u8").reshape(bh, bw)


def encode_bc5_np(rg_u8: np.ndarray) -> np.ndarray:
    """Encode [H, W, 2] uint8 -> [H//4, W//4, 16] uint8 BC5_UNORM."""
    c0 = encode_bc4_channel_np(rg_u8[..., 0])
    c1 = encode_bc4_channel_np(rg_u8[..., 1])
    out = np.stack([c0, c1], axis=-1).astype("<u8")
    return out.view(np.uint8).reshape(rg_u8.shape[0] // 4, rg_u8.shape[1] // 4, 16)


def downsample_box2x(arr: np.ndarray) -> np.ndarray:
    h, w, c = arr.shape
    return (
        arr.reshape(h // 2, 2, w // 2, 2, c)
        .astype(np.float32)
        .mean(axis=(1, 3))
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate PBR + NTC asset pack for PA-AHNTS")
    parser.add_argument("--steps", type=int, default=550, help="Adam training steps for NTC")
    parser.add_argument("--force", action="store_true", help="Regenerate even if pack exists")
    args = parser.parse_args()

    if PACK_PATH.exists() and MANIFEST_PATH.exists() and not args.force:
        print(f"[asset_gen] Pack already exists: {PACK_PATH}")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[asset_gen] Generating 32 seamless 512x512 PBR base materials on {device}...")

    t_start = time.perf_counter()
    mats = generate_32_pbr_materials(device=device, size=512)
    print(f"[asset_gen] Synthesized PBR base maps in {time.perf_counter() - t_start:.2f}s. Training batched NTC...")

    ntc = train_batched_ntc(mats, steps=args.steps)
    psnrs = ntc["psnr_per_mat"]
    print(
        f"[asset_gen] NTC training done in {ntc['train_seconds']:.2f}s | "
        f"Mean PSNR={np.mean(psnrs):.2f} dB (min={np.min(psnrs):.2f}, max={np.max(psnrs):.2f})"
    )

    # Convert reference PBR maps to uint8 and encode 7 base mip levels (512 down to 8) to BC7/BC5
    mats_np = mats.permute(0, 2, 3, 1).contiguous().cpu().numpy()  # [32, 512, 512, 8]
    mip_sizes = [512, 256, 128, 64, 32, 16, 8]

    blob = bytearray()
    def append_bytes(b: bytes) -> tuple[int, int]:
        off = len(blob)
        blob.extend(b)
        # Pad to 16-byte alignment
        pad = (-len(blob)) & 15
        if pad:
            blob.extend(b"\x00" * pad)
        return off, len(b)

    # 1. Pack MLP weights per material as structured binary (1280 + 64 + 2048 + 64 + 512 + 16 = 3984 bytes -> pad to 4096 bytes)
    mlp_records = []
    mlp_blob_off = len(blob)
    for m in range(32):
        rec = bytearray()
        rec.extend(ntc["w1_f16"][m].astype("<f2").tobytes())  # 16*32*2 = 1024 B
        rec.extend(ntc["b1_f16"][m].astype("<f2").tobytes())  # 32*2 = 64 B
        rec.extend(ntc["w2_f16"][m].astype("<f2").tobytes())  # 32*32*2 = 2048 B
        rec.extend(ntc["b2_f16"][m].astype("<f2").tobytes())  # 32*2 = 64 B
        rec.extend(ntc["w3_f16"][m].astype("<f2").tobytes())  # 32*8*2 = 512 B
        rec.extend(ntc["b3_f16"][m].astype("<f2").tobytes())  # 8*2 = 16 B
        # Pad each material record to 3840 bytes (240 float4 / uint4)
        while len(rec) < 3840:
            rec.extend(b"\x00" * (3840 - len(rec)))
        blob.extend(rec)
    mlp_blob_size = len(blob) - mlp_blob_off

    # 2. Pack G0 (256..4, 7 mips or full 9 mips 256..1) and G1 (64..1, 7 mips)
    materials_meta = []
    for m in range(32):
        g0_cur = ntc["g0_u8"][m].astype(np.float32)  # [256,256,4]
        g0_mips = []
        for s in [256, 128, 64, 32, 16, 8, 4, 2, 1]:
            g0_u8_s = np.clip(np.rint(g0_cur), 0, 255).astype(np.uint8)
            off, sz = append_bytes(g0_u8_s.tobytes())
            g0_mips.append({"size": s, "offset": off, "bytes": sz})
            if s > 1:
                g0_cur = downsample_box2x(g0_cur)

        g1_cur = ntc["g1_f16"][m].astype(np.float32)  # [64,64,4]
        g1_mips = []
        for s in [64, 32, 16, 8, 4, 2, 1]:
            g1_f16_s = g1_cur.astype("<f2")
            off, sz = append_bytes(g1_f16_s.tobytes())
            g1_mips.append({"size": s, "offset": off, "bytes": sz})
            if s > 1:
                g1_cur = downsample_box2x(g1_cur)

        # 3. Encode BC7/BC5 base mip pyramid (512 down to 8)
        cur_pbr = mats_np[m]
        bcn_mips = []
        for s in mip_sizes:
            pbr_u8 = np.clip(np.rint(cur_pbr * 255.0), 0, 255).astype(np.uint8)
            albedo_ao = pbr_u8[..., 0:4]
            normal_xy = pbr_u8[..., 4:6]
            orm = np.stack(
                [pbr_u8[..., 6], pbr_u8[..., 7], pbr_u8[..., 3], np.full((s, s), 255, dtype=np.uint8)],
                axis=-1,
            )
            bc_albedo = encode_bc7_mode6_np(albedo_ao)
            bc_normal = encode_bc5_np(normal_xy)
            bc_orm = encode_bc7_mode6_np(orm)

            off_a, sz_a = append_bytes(bc_albedo.tobytes())
            off_n, sz_n = append_bytes(bc_normal.tobytes())
            off_o, sz_o = append_bytes(bc_orm.tobytes())
            bcn_mips.append(
                {
                    "size": s,
                    "albedo_bc7": {"offset": off_a, "bytes": sz_a},
                    "normal_bc5": {"offset": off_n, "bytes": sz_n},
                    "orm_bc7": {"offset": off_o, "bytes": sz_o},
                }
            )
            if s > 8:
                cur_pbr = downsample_box2x(cur_pbr)

        materials_meta.append(
            {
                "material_id": m,
                "ntc_psnr_db": round(float(psnrs[m]), 3),
                "g0_mips": g0_mips,
                "g1_mips": g1_mips,
                "bcn_mips": bcn_mips,
            }
        )

    PACK_PATH.write_bytes(blob)
    manifest = {
        "schema_version": "1.0",
        "pack_file": str(PACK_PATH.relative_to(REPO_ROOT)),
        "pack_total_bytes": len(blob),
        "material_count": 32,
        "base_tile_size": 512,
        "world_repeat": 16,
        "virtual_texture_size": 8192,
        "ntc_train_seconds": ntc["train_seconds"],
        "ntc_mean_psnr_db": round(float(np.mean(psnrs)), 3),
        "mlp_table": {
            "offset": mlp_blob_off,
            "bytes": mlp_blob_size,
            "stride_bytes": 3840,
        },
        "materials": materials_meta,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"[asset_gen] Wrote {PACK_PATH} ({len(blob) / (1024*1024):.2f} MiB) "
        f"and {MANIFEST_PATH} in {time.perf_counter() - t_start:.2f}s total."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
