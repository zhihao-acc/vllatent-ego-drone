#!/usr/bin/env python
"""Wrapper: inject NystromAttention shim, then run UniDepth's demo_mega-sam.py.

Usage (from MegaSaM repo root):
    python /path/to/run_unidepth.py [UniDepth demo_mega-sam args...]
"""
import importlib.util
import runpy
import sys
from pathlib import Path

shim_path = Path(__file__).resolve().parent / "nystrom_shim.py"
spec = importlib.util.spec_from_file_location("nystrom_shim", shim_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

sys.argv[0] = "UniDepth/scripts/demo_mega-sam.py"
runpy.run_path(sys.argv[0], run_name="__main__")
