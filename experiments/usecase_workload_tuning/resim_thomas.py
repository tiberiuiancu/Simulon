#!/usr/bin/env python3
"""Re-simulate Tiberiu's workload-tuning sweep on the CURRENT (thomas) code.

His original sweep (results.csv) ran on main, before the validate-qwen3-33b changes
(kernel-timing GEMM correction, memory model, NCCL by_topology, _TRACE_DEFAULTS fusion
fix). This driver re-runs `simulon simulate` on his EXISTING H100 traces — NO re-tracing —
so it isolates the effect of our CODE-level changes on his simulated MFU. It CANNOT undo
trace-level biases (his traces are SP=true and fusions-off); those need a re-trace.

Output: results_thomas.csv (his MFU vs ours, side by side) + a ranking-shift summary.

Run:  PYTHONPATH=. python experiments/usecase_workload_tuning/resim_thomas.py
"""
from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from pathlib import Path

_BASE = Path(__file__).parent
_SCEN = _BASE / "scenarios"
_REPO = _BASE.parent.parent
_HIS_CSV = _BASE / "results.csv"
_OUT_CSV = _BASE / "results_thomas.csv"


def _parse_name(name: str):
    tp = pp = mbs = None
    vpp = ""
    for p in name.split("_"):
        if p.startswith("tp"): tp = p[2:]
        elif p.startswith("pp"): pp = p[2:]
        elif p.startswith("mbs"): mbs = p[3:]
        elif p.startswith("vpp"): vpp = p[3:]
    return tp, pp, mbs, vpp


def _trace_dir_for(scenario: Path) -> Path | None:
    """Read traces_dir from the scenario and check the traces actually exist."""
    import yaml
    with open(scenario) as f:
        sc = yaml.safe_load(f)
    td = (sc.get("workload") or {}).get("traces_dir")
    if not td:
        return None
    p = _REPO / td
    return p if (p / "trace_rank_0.json").exists() else None


def _resim(scenario: Path) -> dict | None:
    env = dict(os.environ, PYTHONPATH=str(_REPO))
    cmd = [sys.executable, "-c", "from simulon.cli import app; app()",
           "simulate", str(scenario)]
    try:
        out = subprocess.run(cmd, check=True, capture_output=True, text=True,
                             env=env, timeout=300).stdout
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)[:80]}
    m = {}
    for line in out.splitlines():
        if "MFU:" in line:
            with __import__("contextlib").suppress(Exception):
                m["mfu"] = float(line.split("MFU:")[-1].replace("%", "").strip())
        elif "Iteration wall time" in line:
            with __import__("contextlib").suppress(Exception):
                m["iter_ms"] = float(re.findall(r"([0-9.]+)\s*ms", line)[0])
    return m or {"error": "no MFU parsed"}


def main() -> None:
    # his MFU by config name
    his = {}
    with open(_HIS_CSV) as f:
        for row in csv.DictReader(f):
            if row.get("mfu_pct"):
                try:
                    his[row["name"]] = float(row["mfu_pct"])
                except ValueError:
                    pass

    rows = []
    scen_dirs = sorted(d for d in _SCEN.iterdir() if d.is_dir())
    for d in scen_dirs:
        sc = d / "scenario.yaml"
        if not sc.exists():
            continue
        if _trace_dir_for(sc) is None:
            continue  # no traces -> can't re-sim (was OOM/error in his run)
        res = _resim(sc)
        tp, pp, mbs, vpp = _parse_name(d.name)
        r = {"name": d.name, "tp": tp, "pp": pp, "mbs": mbs, "vpp": vpp,
             "mfu_his": his.get(d.name, ""),
             "mfu_thomas": res.get("mfu", ""),
             "iter_ms_thomas": res.get("iter_ms", ""),
             "error": res.get("error", "")}
        if r["mfu_his"] != "" and r["mfu_thomas"] != "":
            r["delta_pp"] = round(r["mfu_thomas"] - r["mfu_his"], 2)
        else:
            r["delta_pp"] = ""
        rows.append(r)
        print(f"  {d.name:26} his={r['mfu_his'] or '-':>7} thomas={r['mfu_thomas'] or '-':>7} "
              f"d={r['delta_pp'] if r['delta_pp']!='' else '-':>6}  {r['error']}")

    with open(_OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "tp", "pp", "mbs", "vpp",
                                          "mfu_his", "mfu_thomas", "delta_pp",
                                          "iter_ms_thomas", "error"])
        w.writeheader()
        w.writerows(rows)

    # ranking comparison on configs both runs simulated
    both = [r for r in rows if r["mfu_his"] != "" and r["mfu_thomas"] != ""]
    print("\n" + "=" * 70)
    print(f"re-simulated {len(rows)} configs with traces; {len(both)} comparable to his run")
    if both:
        deltas = [abs(r["delta_pp"]) for r in both]
        print(f"MFU delta (thomas - his): max|Δ|={max(deltas):.2f}pp  "
              f"mean|Δ|={sum(deltas)/len(deltas):.2f}pp")
        his_rank = sorted(both, key=lambda r: -r["mfu_his"])
        our_rank = sorted(both, key=lambda r: -r["mfu_thomas"])
        print(f"his winner:    {his_rank[0]['name']} @ {his_rank[0]['mfu_his']:.2f}%")
        print(f"thomas winner: {our_rank[0]['name']} @ {our_rank[0]['mfu_thomas']:.2f}%")
        print(f"winner unchanged: {his_rank[0]['name'] == our_rank[0]['name']}")
        # top-5 ranking stability
        top_his = [r["name"] for r in his_rank[:5]]
        top_our = [r["name"] for r in our_rank[:5]]
        print(f"top-5 identical set: {set(top_his) == set(top_our)}")
        print(f"  his  top5: {top_his}")
        print(f"  ours top5: {top_our}")
    print(f"\nwrote {_OUT_CSV}")


if __name__ == "__main__":
    main()
