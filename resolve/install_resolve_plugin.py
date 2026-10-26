#!/usr/bin/env python3
"""Install the CorridorKey Fusion node into DaVinci Resolve.

Run this with the Python environment you set up for CorridorKey (the one with
PyTorch installed), from anywhere:

    venv\\Scripts\\python resolve\\install_resolve_plugin.py      (Windows)
    venv/bin/python resolve/install_resolve_plugin.py           (Linux / macOS)

It copies CorridorKey.fuse into Resolve's Fuses folder with the path to this
CorridorKey folder and to this Python filled in, then checks the setup.
Restart Resolve afterwards. The node appears in the Fusion page under
Add Tool > Fuses > Keying > CorridorKey (or search for "CorridorKey").
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FUSE_NAME = "CorridorKey.fuse"


def fuses_dirs() -> list[Path]:
    home = Path.home()
    if sys.platform.startswith("win"):
        appdata = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
        return [appdata / "Blackmagic Design" / "DaVinci Resolve" / "Support" / "Fusion" / "Fuses"]
    if sys.platform == "darwin":
        dirs = [home / "Library" / "Application Support" / "Blackmagic Design"
                / "DaVinci Resolve" / "Fusion" / "Fuses"]
        app_store = (home / "Library" / "Containers" / "com.blackmagic-design.DaVinciResolveAppStore"
                     / "Data" / "Library" / "Application Support" / "Fusion" / "Fuses")
        if app_store.parent.parent.exists():
            dirs.append(app_store)
        return dirs
    return [home / ".local" / "share" / "DaVinciResolve" / "Fusion" / "Fuses"]


def default_python() -> Path:
    exe = Path(sys.executable)
    if sys.platform.startswith("win"):
        # pythonw.exe runs the background job without opening a console window.
        pythonw = exe.with_name("pythonw.exe")
        if pythonw.exists():
            return pythonw
    return exe


def check_setup(python: Path) -> list[str]:
    problems = []
    weights = ROOT / "weights"
    if not any(weights.glob("corridorkey_v2.*")):
        problems.append("Model weights not found in weights/. Run: python scripts/download_weights.py")
    for sub in ("cradio", "moge", "rvm"):
        if not any(p.name != ".gitkeep" for p in (ROOT / "vfms" / sub).glob("*")):
            problems.append(f"Foundation model vfms/{sub} is missing. Run: python scripts/download_weights.py")
            break
    try:
        import torch
        if not torch.cuda.is_available():
            problems.append("PyTorch is installed but cannot see a CUDA GPU "
                            "(install the CUDA build of PyTorch, see README Setup)")
    except ImportError:
        problems.append(f"PyTorch is not installed for {python}. Follow the README Setup steps "
                        "and run this installer with that environment's Python.")
    for module in ("numpy", "cv2"):
        try:
            __import__(module)
        except ImportError:
            problems.append(f"Python package '{module}' is missing: pip install -r requirements.txt")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fuses-dir", type=Path, default=None,
                    help="Install into this Fuses folder instead of Resolve's default.")
    ap.add_argument("--python", type=Path, default=None,
                    help="Python the node should run (default: the one running this script).")
    args = ap.parse_args()

    # abspath, not resolve(): a venv's python is a symlink to the base interpreter,
    # and following it would lose the venv's packages.
    python = Path(os.path.abspath(args.python or default_python()))
    source = (HERE / FUSE_NAME).read_text(encoding="utf-8")
    fuse = (source
            .replace("@@CK_ROOT@@", ROOT.as_posix())
            .replace("@@CK_PYTHON@@", python.as_posix()))

    targets = [args.fuses_dir] if args.fuses_dir else fuses_dirs()
    for target in targets:
        target.mkdir(parents=True, exist_ok=True)
        (target / FUSE_NAME).write_text(fuse, encoding="utf-8")
        print(f"Installed {FUSE_NAME} -> {target}")
    print(f"  CorridorKey folder: {ROOT}")
    print(f"  Python:             {python}")

    problems = check_setup(python)
    if problems:
        print("\nThe node is installed, but keying will fail until these are fixed:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("\nSetup looks good.")
    print("\nRestart DaVinci Resolve, then in the Fusion page add the node from "
          "Add Tool > Fuses > Keying > CorridorKey.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
