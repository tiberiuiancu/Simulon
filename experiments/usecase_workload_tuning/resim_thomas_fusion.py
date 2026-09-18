#!/usr/bin/env python3
"""Analyze the THOMAS fused/unfused A/B (after trace_thomas_fusion_ab.sbatch lands).

For each of Tiberiu's top-5 configs it re-simulates the GH200 fused and unfused traces
and compares to his committed H100 MFU (results.csv). Reports:
  * fusion effect   = MFU(fused) - MFU(unfused)  on the SAME GH200 + container (CLEAN)
  * proxy gap       = MFU(unfused, GH200) - MFU(his, H100)  (how good GH200 is for H100)
  * est. H100 fused = his_H100 + fusion_effect   (the bias-corrected estimate vs 54.4%)

Run:  PYTHONPATH=. python experiments/usecase_workload_tuning/resim_thomas_fusion.py
"""
from __future__ import annotations

import contextlib
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

_BASE = Path(__file__).parent
_SCEN = _BASE / "scenarios_thomas"
_REPO = _BASE.parent.parent
_HIS_CSV = _BASE / "results.csv"
_TOPS = ["tp4_pp2_mbs2_vpp8", "tp4_pp2_mbs2_vpp16", "tp4_pp2_mbs2_vpp32",
         "tp8_pp1_mbs4", "tp4_pp2_mbs2_vpp1"]


def _his_mfu() -> dict[str, float]:
    out = {}
    with open(_HIS_CSV) as f:
        for row in csv.DictReader(f):
            if row.get("mfu_pct"):
                with contextlib.suppress(ValueError):
                    out[row["name"]] = float(row["mfu_pct"])
    return out


def _sim(scenario: Path) -> float | None:
    if not (scenario / "scenario.yaml").exists():
        return None
    env = dict(os.environ, PYTHONPATH=str(_REPO))
    cmd = [sys.executable, "-c", "from simulon.cli import app; app()",
           "simulate", str(scenario / "scenario.yaml")]
    try:
        out = subprocess.run(cmd, check=True, capture_output=True, text=True,
                             env=env, timeout=300).stdout
    except Exception:  # noqa: BLE001
        return None
    for line in out.splitlines():
        if "MFU:" in line:
            with contextlib.suppress(Exception):
                return float(line.split("MFU:")[-1].replace("%", "").strip())
    return None


def main() -> None:
    his = _his_mfu()
    print(f"{'config':22}{'his_H100':>9}{'GH200_unf':>10}{'GH200_fus':>10}"
          f"{'fusionΔ':>9}{'proxyΔ':>8}{'est_H100_fused':>15}")
    for cfg in _TOPS:
        h = his.get(cfg)
        unf = _sim(_SCEN / f"{cfg}_unfused")
        fus = _sim(_SCEN / f"{cfg}_fused")
        fusion_d = (fus - unf) if (fus is not None and unf is not None) else None
        proxy_d = (unf - h) if (unf is not None and h is not None) else None
        est = (h + fusion_d) if (h is not None and fusion_d is not None) else None
        def f(x, s=1): return f"{x:.2f}" if isinstance(x, (int, float)) else "-"
        print(f"{cfg:22}{f(h):>9}{f(unf):>10}{f(fus):>10}"
              f"{('%+.2f'%fusion_d) if fusion_d is not None else '-':>9}"
              f"{('%+.2f'%proxy_d) if proxy_d is not None else '-':>8}"
              f"{f(est):>15}")
    print("\nfusionΔ = pure fusion effect on Hopper (clean, same GH200+container).")
    print("proxyΔ  = GH200-unfused minus his H100 (small ⇒ GH200 is a good H100 proxy).")
    print("est_H100_fused = his_H100 + fusionΔ = bias-corrected MFU vs his 54.4%.")
    print("NOTE: SP=true bias is NOT corrected here — needs real H100 hardware.")


if __name__ == "__main__":
    main()
