#!/usr/bin/env python3
"""Quantify the SP=true optimism (after trace_thomas_sp_ab.sbatch lands).

Simulates each config with its SP=true and SP=false GH200 trace (fusions held OFF = his
state) and reports the SP=true - SP=false SIM delta. On real hardware SP is ~time-neutral,
so a negative delta (SP=true faster) is the optimism his Snellius traces carry. His H100
physical median is shown for reference (GH200!=H100 absolute, so compare the DELTA, not
the absolute).

Run:  PYTHONPATH=. python experiments/validate_e2e/resim_thomas_sp.py
"""
from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from pathlib import Path

_BASE = Path(__file__).parent
_SCEN = _BASE / "scenarios_thomas_sp"
_REPO = _BASE.parent.parent
# his H100 physical medians (from results.csv) for reference
_HIS_PHYS = {"qwen3-32b-tp4-pp2-mbs2-vpp8": 8329.0,
             "qwen3-32b-tp2-pp4-mbs1-vpp1": 8974.0}
_CONFIGS = ["qwen3-32b-tp4-pp2-mbs2-vpp8", "qwen3-32b-tp2-pp4-mbs1-vpp1"]


def _sim_ms(scenario: Path) -> float | None:
    sc = scenario / "scenario.yaml"
    if not sc.exists():
        return None
    env = dict(os.environ, PYTHONPATH=str(_REPO))
    cmd = [sys.executable, "-c", "from simulon.cli import app; app()", "simulate", str(sc)]
    try:
        out = subprocess.run(cmd, check=True, capture_output=True, text=True,
                             env=env, timeout=300).stdout
    except Exception:  # noqa: BLE001
        return None
    import re
    for line in out.splitlines():
        if "Iteration wall time" in line:
            with contextlib.suppress(Exception):
                return float(re.findall(r"([0-9.]+)\s*ms", line)[0])
    return None


def main() -> None:
    print(f"{'config':32}{'SP=true':>10}{'SP=false':>10}{'SPΔ':>9}{'his_H100_phys':>14}")
    for cfg in _CONFIGS:
        t = _sim_ms(_SCEN / f"{cfg}_sptrue")
        fa = _sim_ms(_SCEN / f"{cfg}_spfalse")
        d = (t / fa - 1) * 100 if (t and fa) else None
        phys = _HIS_PHYS.get(cfg)
        print(f"{cfg:32}{(f'{t:.0f}' if t else '-'):>10}{(f'{fa:.0f}' if fa else '-'):>10}"
              f"{(f'{d:+.1f}%' if d is not None else '-'):>9}{(f'{phys:.0f}' if phys else '-'):>14}")
    print("\nSPΔ = SP=true sim / SP=false sim - 1, same GH200 (powercap cancels).")
    print("Real hardware: SP is ~time-neutral, so a negative SPΔ = the SP=true optimism")
    print("his Snellius traces carry (which his fusions-off bug was partly cancelling).")


if __name__ == "__main__":
    main()
