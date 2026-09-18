#!/usr/bin/env python3
"""Attribute a simulated iteration's communication cost to TP / DP / PP, by ablation.

The grounding scorecard showed the simulator over-predicting every DP>1 configuration by
more than any host cost can explain (experiments/host_cost_transfer.py: no solution even at
host_cost_us=0), with ~2 s of exposed communication on a config whose data-parallel traffic
cannot physically exceed ~0.8 s. `SimulationResult.exposed_comm_by_type` is keyed by
collective TYPE, which cannot separate them: under sequence parallelism the tensor-parallel
collectives are ReduceScatter/AllGather too, exactly like the distributed optimizer's.

So: populate the DAG once, then zero one family's durations at a time and re-replay. The
drop in total time is that family's contribution to the critical path, and comparing it with
the family's raw duration says how much is exposed versus hidden. Families:

  TP   intra-node tensor-parallel collectives (all-gather / reduce-scatter under SP)
  DP   the distributed optimizer's grad reduce-scatter and param all-gather
       (recorded by param_and_grad_buffer._record_dp_collective, or synthesized at DP=1)
  PP   pipeline point-to-point

No cluster time: this replays traces already on disk.

    PYTHONPATH=. python experiments/comm_exposure.py [--cells ...] [--host-cost-us 0]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
logging.disable(logging.INFO)

import yaml  # noqa: E402

from experiments.ground import DEFAULT_REGISTRY, anchor_measurement, scan_registry  # noqa: E402

DP_NAMES = ("dist_opt_", "ddp_grad_")


def family(node) -> str:
    name = str(getattr(node, "name", "") or "")
    if name.startswith(DP_NAMES):
        return "DP"
    if node.collective_type in ("PP_Send", "PP_Recv"):
        return "PP"
    return "TP"


def build(cell_dir: Path, nodes: int, gbs: int | None, host: float | None):
    from simulon.backend.analytical import run_trace
    from simulon.backend.dag.collective_populate import populate_collective_network
    from simulon.config.dc import DatacenterConfig
    from simulon.config.scenario import ScenarioConfig
    from simulon.config.workload import MegatronWorkload

    cfg = yaml.safe_load((cell_dir / "workload.yaml").read_text())["config"]
    if gbs:
        cfg["global-batch-size"] = gbs
    cfg["num-gpus"] = nodes * 4
    node = "jupiter-gh200-4g" if host is None else {"from": "jupiter-gh200-4g", "host_cost_us": host}
    dc = DatacenterConfig.model_validate({"num_nodes": nodes, "node": node})
    wl = MegatronWorkload.model_validate(
        {"framework": "megatron", "config": cfg, "traces_dir": str(cell_dir)})
    sc = ScenarioConfig(datacenter=dc, workload=wl)
    dag = run_trace(sc, overlap_async_collectives=True)
    populate_collective_network(dag, dc)
    return dag, dc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", nargs="*", default=["tp4pp2-dp4-ogr", "tp4pp2-dp4-noogr",
                                                   "tp4pp2-bf16-noubo", "tp4pp4-bf16-noubo"])
    ap.add_argument("--host-cost-us", type=float, default=0.0,
                    help="0 isolates the comm question from the host model (default)")
    a = ap.parse_args()

    from simulon.backend.dag.replayer import replay
    reg = {t.path.name: t for t in scan_registry(DEFAULT_REGISTRY)}

    for cell in a.cells:
        t = reg.get(cell)
        if t is None:
            print(f"{cell}: not in the registry"); continue
        m = anchor_measurement(t)
        nodes = int(yaml.safe_load((t.path / "workload.yaml").read_text())["config"]["num-gpus"]) // 4
        dag, _dc = build(t.path, nodes, None, a.host_cost_us)

        nodes_by_fam: dict[str, list] = {"TP": [], "DP": [], "PP": []}
        for n in dag.collective_nodes.values():
            nodes_by_fam[family(n)].append(n)
        saved = {id(n): n.duration_ms for n in dag.collective_nodes.values()}

        base = replay(dag, network_simulation="collective", host_cost_us=a.host_cost_us)
        print(f"\n=== {cell}  (pp={t.pp} dp={t.dp}, {nodes} nodes)   measured "
              f"{m.med_ms if m else float('nan'):.0f} ms   sim {base.total_time_ms:.0f} ms"
              f"   compute {base.compute_ms:.0f}   exposed comm {base.exposed_comm_ms:.0f}")
        print(f"{'family':<5}{'nodes':>7}{'raw dur/rank':>14}{'on crit path':>14}{'hidden':>9}"
              f"   {'bytes/rank':>13}")
        world = max(1, dag_world(dag))
        for fam in ("TP", "DP", "PP"):
            fam_nodes = nodes_by_fam[fam]
            if not fam_nodes:
                continue
            raw = sum(n.duration_ms or 0.0 for n in fam_nodes) / world
            byts = sum(getattr(n, "data_size", 0) or 0 for n in fam_nodes) / world
            for n in fam_nodes:
                n.duration_ms = 0.0
            r = replay(dag, network_simulation="collective", host_cost_us=a.host_cost_us)
            for n in fam_nodes:
                n.duration_ms = saved[id(n)]
            crit = base.total_time_ms - r.total_time_ms
            hidden = (1 - crit / raw) * 100 if raw else 0.0
            print(f"{fam:<5}{len(fam_nodes):>7}{raw:>13.0f}m{crit:>13.0f}m{hidden:>8.0f}%"
                  f"   {byts/1e9:>11.2f} GB")
        # all comm off -> the pure compute+bubble floor
        for n in dag.collective_nodes.values():
            n.duration_ms = 0.0
        r0 = replay(dag, network_simulation="collective", host_cost_us=a.host_cost_us)
        for n in dag.collective_nodes.values():
            n.duration_ms = saved[id(n)]
        print(f"{'none':<5}{'':>7}{'':>14}{base.total_time_ms - r0.total_time_ms:>13.0f}m"
              f"{'':>9}   (all comm removed -> {r0.total_time_ms:.0f} ms floor)")


def dag_world(dag) -> int:
    ranks = {r for n in dag.collective_nodes.values() for r in (n.group_ranks or [])}
    return len(ranks) or 1


if __name__ == "__main__":
    main()
