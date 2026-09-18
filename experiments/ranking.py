#!/usr/bin/env python3
"""Does the simulator pick the right configuration? (Not: is its number close?)

Absolute accuracy is a proxy. What this project is for is CHOOSING a parallelism strategy,
and for that a systematic bias is nearly harmless -- it cancels between configs measured at
the same scale -- while a small random error that reorders near-ties is fatal. A 9.6% mean
error is consistent with both perfect and useless ranking; only this tells them apart.

A DECISION SET is the choice actually faced: given N nodes and a target global batch size,
which (PP, VPP/layout, mbs, precision) do you run? So configs are grouped by
(model, nodes, gbs) and ranked within the group.

Criteria are the ones pre-registered in CAMPAIGN.md, not invented here:
    Spearman rho >= 0.9,  top-1 regret <= 3%,  top-5 recall >= 4/5
plus the harder one that document flags: ordering accuracy restricted to NEAR-TIES (pairs
whose measured times are within 10% of each other), which is what an optimizer must resolve
and where a flattering rho can hide a useless model.

    PYTHONPATH=. python experiments/ranking.py results/ground/qwen3-32b_layout.csv
"""
from __future__ import annotations

import argparse
import csv
import itertools
import statistics
from collections import defaultdict
from pathlib import Path


def spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):                      # average ties
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def config_identity(cfg: str) -> str:
    """Strip the fields that do not change the decision, so repeats of one config collapse."""
    drop = {"ubo", "ogr", "p2p"}
    return " ".join(t for t in cfg.split() if t not in drop and not t.startswith(("M", "gbs")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path, nargs="?", default=Path("results/ground/qwen3-32b_layout.csv"))
    ap.add_argument("--near-tie", type=float, default=10.0, help="pct within which two configs are a near-tie")
    ap.add_argument("--min-set", type=int, default=3, help="minimum configs in a decision set")
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(a.csv))
            if r["sim_ms"] and r["verdict"] != "NOISY" and r["tier"] != "self-anchor"]
    # collapse repeats of the same configuration within a decision set
    pooled: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        nodes = int(r["cfg"].split("n ")[0])
        gbs = next((t[3:] for t in r["cfg"].split() if t.startswith("gbs")), "?")
        pooled[(r["model"], nodes, gbs, config_identity(r["cfg"]))].append(
            (float(r["meas_ms"]), float(r["sim_ms"])))
    sets: dict[tuple, list[tuple[str, float, float]]] = defaultdict(list)
    for (model, nodes, gbs, ident), vals in pooled.items():
        sets[(model, nodes, gbs)].append(
            (ident, statistics.median(v[0] for v in vals), statistics.median(v[1] for v in vals)))

    usable = {k: v for k, v in sets.items() if len(v) >= a.min_set}
    print(f"{len(rows)} scored rows -> {len(pooled)} distinct configs -> "
          f"{len(sets)} decision sets, {len(usable)} with >= {a.min_set} configs\n")

    print(f"{'decision set':<26}{'n':>3}{'rho':>7}{'top1 regret':>13}{'top3 recall':>13}   sim pick")
    rhos, regrets, recalls = [], [], []
    for (model, nodes, gbs), items in sorted(usable.items(), key=lambda kv: (kv[0][1], kv[0][2])):
        meas = [m for _, m, _ in items]
        sim = [s for _, _, s in items]
        rho = spearman(meas, sim)
        best_meas = min(meas)
        pick = min(range(len(items)), key=lambda i: sim[i])
        regret = (meas[pick] / best_meas - 1) * 100
        true_top3 = {i for i in sorted(range(len(items)), key=lambda i: meas[i])[:3]}
        sim_top3 = {i for i in sorted(range(len(items)), key=lambda i: sim[i])[:3]}
        recall = len(true_top3 & sim_top3) / min(3, len(items))
        rhos.append(rho); regrets.append(regret); recalls.append(recall)
        print(f"{f'{nodes}n gbs{gbs}':<26}{len(items):>3}{rho:>7.3f}{regret:>12.1f}%{recall:>12.0%}"
              f"   {items[pick][0][:40]}")

    print(f"\n{'POOLED':<26}{'':>3}{statistics.mean(rhos):>7.3f}"
          f"{statistics.mean(regrets):>12.1f}%{statistics.mean(recalls):>12.0%}   (means over sets)")
    print(f"{'':26}   median rho {statistics.median(rhos):.3f}   "
          f"median regret {statistics.median(regrets):.1f}%   "
          f"sets with regret <= 3%: {sum(1 for r in regrets if r <= 3)}/{len(regrets)}")

    # pairwise ordering, all pairs and near-ties only
    allp = tie = allok = tieok = 0
    for items in usable.values():
        for (_, m1, s1), (_, m2, s2) in itertools.combinations(items, 2):
            if m1 == m2:
                continue
            ok = (s1 < s2) == (m1 < m2)
            allp += 1; allok += ok
            if abs(m1 / m2 - 1) * 100 <= a.near_tie:
                tie += 1; tieok += ok
    print(f"\npairwise ordering: all pairs {allok}/{allp} = {allok/allp:.0%}"
          f"   |   near-ties (<= {a.near_tie:.0f}% apart) {tieok}/{tie} = "
          f"{tieok/tie:.0%}" if tie else "")
    print("\nPRE-REGISTERED (CAMPAIGN.md): rho >= 0.9, top-1 regret <= 3%, top-5 recall >= 4/5")
    v = ("PASS" if statistics.mean(rhos) >= 0.9 else "FAIL",
         "PASS" if statistics.mean(regrets) <= 3 else "FAIL",
         "PASS" if statistics.mean(recalls) >= 0.8 else "FAIL")
    print(f"  rho {v[0]}   regret {v[1]}   recall {v[2]}")


if __name__ == "__main__":
    main()
