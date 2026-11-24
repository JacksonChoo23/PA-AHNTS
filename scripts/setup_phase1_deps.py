#!/usr/bin/env python3
"""Download and verify Phase 1 dependencies: flip-evaluator, DXC, and DX12 Agility SDK."""

from __future__ import annotations

import io
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = REPO_ROOT / "third_party"

DXC_VER = "1.9.2609.5"
DXC_URL = f"https://api.nuget.org/v3-flatcontainer/microsoft.direct3d.dxc/{DXC_VER}/microsoft.direct3d.dxc.{DXC_VER}.nupkg"

AGILITY_VER = "1.620.2"
AGILITY_URL = f"https://api.nuget.org/v3-flatcontainer/microsoft.direct3d.d3d12/{AGILITY_VER}/microsoft.direct3d.d3d12.{AGILITY_VER}.nupkg"


def install_pip_package(pkg: str) -> None:
    print(f"[deps] Installing Python package: {pkg}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


def download_and_extract_nupkg(url: str, dest_dir: Path, required_rel_path: str) -> None:
    marker = dest_dir / required_rel_path
    if marker.exists():
        print(f"[deps] Already present: {marker}")
        return

    print(f"[deps] Downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "PA-AHNTS-Setup/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()

    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(dest_dir)

    if not marker.exists():
        raise RuntimeError(f"Expected file missing after extraction: {marker}")
    print(f"[deps] Extracted to {dest_dir} (verified {required_rel_path})")


def main() -> int:
    install_pip_package("flip-evaluator")
    import flip_evaluator  # noqa: F401
    print("[deps] Verified import flip_evaluator")

    download_and_extract_nupkg(
        DXC_URL,
        THIRD_PARTY / "dxc",
        "build/native/bin/x64/dxc.exe",
    )
    download_and_extract_nupkg(
        AGILITY_URL,
        THIRD_PARTY / "agility_sdk",
        "build/native/bin/x64/D3D12Core.dll",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
