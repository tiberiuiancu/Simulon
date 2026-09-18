#!/usr/bin/env python3
"""KNOWN BIAS THIS SCRIPT CANNOT DETECT (measured 2026-08-03)
--------------------------------------------------------
The LAST pipeline stage's fake-PG trace under-measures real multi-GPU compute by 23.2%
(tp2pp2-mbs2 stage1: trace 9841 ms vs nsys 12812 ms), while every other stage is only
4-10% low. That stage owns the 262144-vocab lm_head and the vocab-parallel cross-entropy,
which the fake process group does not execute faithfully. PP predictions are therefore
systematically OPTIMISTIC, and more so the deeper the pipeline; it accounts for the whole
of tp2pp2-mbs2's -17.8%.

No ratio heuristic catches it -- the last stage still looks heavier than the first (1.63x
to 2.51x across the 1.7B grid), because it IS heavier, just not heavy enough. Detecting it
needs an nsys profile of the same config. Do NOT paper over it with a correction factor;
the real fixes are a real multi-GPU trace for the last stage, or an analytic lm_head term.

Gate traces before ANY of them is used for a verdict.

Written after a silent corruption: the kernel-timing traces booked every kernel onto the
FORWARD slots (fwd=6832ms, bwd=EXACTLY 0ms), which zeroes backward compute in the replay
and invalidates every schedule-sensitive result. Nothing caught it because the traces
looked fine — right slot counts, plausible totals, no errors. These checks would have.

    PYTHONPATH=. python experiments/check_trace_health.py                 # all registries
    PYTHONPATH=. python experiments/check_trace_health.py <traces_dir>    # one directory

Exit code 1 if any FAIL, so it can gate a pipeline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXPECT_PADDED_VOCAB = 262272  # NullTokenizer needs padded > vocab; see the TP=1 crash
# Fallbacks ONLY for traces whose workload.yaml is missing (e.g. the cancelled-job rc
# trace). The real values are read from each trace's own workload.yaml so this checker
# cannot drift out of sync with the configs it is checking.
FALLBACK_GBS, FALLBACK_GPUS = 256, 4

DEFAULT_DIRS = [
    REPO / "templates/gpu/gh200_jupiter-1p7b/traces",
    REPO / "templates/gpu/gh200_jupiter-1p7b-kerneltime/traces",
]

FAIL, WARN, OK = "FAIL", "WARN", "ok"


def parse_cell(name: str) -> tuple[int, int, int] | None:
    """'tp2pp1-mbs8' -> (tp, pp, mbs); None if the name is not a standard cell."""
    try:
        head, mbs_part = name.split("-")[0], [p for p in name.split("-") if p.startswith("mbs")][0]
        tp = int(head.split("pp")[0].replace("tp", ""))
        pp = int(head.split("pp")[1])
        return tp, pp, int(mbs_part.replace("mbs", ""))
    except (ValueError, IndexError):
        return None


def cell_config(d: Path) -> tuple[int, int, int, int, int] | None:
    """(tp, pp, mbs, gbs, gpus) from the trace's OWN workload.yaml; else parse the dir name.

    Reading the shipped config rather than assuming it keeps this checker correct when the
    grid changes (different gbs, node count, or model) instead of silently mis-flagging.
    """
    wl = d / "workload.yaml"
    if wl.exists():
        try:
            import yaml
            cfg = yaml.safe_load(wl.read_text())["config"]
            pp = int(cfg.get("pipeline-model-parallel-size", 1))
            nl = int(cfg.get("num-layers", 0))
            # Virtual pipeline chunks PER STAGE. An interleaved schedule runs one slot per
            # (microbatch, chunk, direction), so the slot count below must account for it or
            # every VPP/layout trace FAILs a check it passes. Two ways to express it:
            # an explicit layout (count its "|"-separated groups) or num-layers-per-
            # virtual-pipeline-stage.
            layout = cfg.get("pipeline-model-parallel-layout")
            nlvps = cfg.get("num-layers-per-virtual-pipeline-stage")
            if layout:
                chunks = max(1, len(str(layout).replace("\\|", "|").split("|")) // pp)
            elif nlvps:
                chunks = max(1, nl // (pp * int(nlvps))) if nl else 1
            else:
                chunks = 1
            return (int(cfg.get("tensor-model-parallel-size", 1)),
                    pp,
                    int(cfg.get("micro-batch-size", 1)),
                    int(cfg.get("global-batch-size", FALLBACK_GBS)),
                    int(cfg.get("num-gpus", FALLBACK_GPUS)),
                    nl, chunks)
        except Exception:  # malformed yaml -> fall through to the name
            pass
    p = parse_cell(d.name)
    return (p[0], p[1], p[2], FALLBACK_GBS, FALLBACK_GPUS, 0, 1) if p else None


def check_cell(d: Path) -> list[tuple[str, str]]:
    """Return [(level, message)] for one trace directory."""
    out: list[tuple[str, str]] = []
    ranks = sorted(d.glob("trace_rank_*.json"))
    if not ranks:
        return [(FAIL, "no trace_rank_*.json")]

    parsed = cell_config(d)
    tr = json.loads(ranks[0].read_text())
    evs = tr.get("events", [])
    slots = [e for e in evs if e.get("type") == "slot_begin"]
    kslots = [e for e in slots if "kernel_device_ms" in (e.get("metadata") or {})]

    # 1) rank coverage must match PP (one trace per pipeline stage)
    if parsed:
        tp, pp, mbs, gbs, gpus, _nl, chunks = parsed
        if len(ranks) != pp:
            out.append((WARN, f"{len(ranks)} rank file(s), PP={pp} (expect one per stage)"))
        # 2) slot count must match the microbatch schedule
        dp = max(1, gpus // (tp * pp))
        n_mb = gbs // (mbs * dp)
        # fwd+bwd per (microbatch, virtual chunk), plus the optimizer step
        expect = n_mb * chunks * 2 + 1
        if len(slots) != expect:
            out.append((FAIL, f"{len(slots)} slots, expected {expect} "
                              f"(n_mb={n_mb}, chunks/stage={chunks})"))
        if chunks > 1:
            missing = [e for e in slots
                       if (e.get("metadata") or {}).get("direction") != "step"
                       and "model_chunk_id" not in (e.get("metadata") or {})]
            if missing:
                out.append((WARN, f"interleaved trace ({chunks} chunks/stage) but "
                                  f"{len(missing)}/{len(slots)} slots carry no model_chunk_id "
                                  f"-- captured before the tracer recorded it; slot->layer "
                                  f"mapping is not recoverable"))

    # TP collective coverage: with SP=false a transformer layer needs 2 AllReduce per
    # direction (row-parallel FORWARD, column-parallel BACKWARD). Only the forward half was
    # instrumented until 2026-07-30, so traces carried exactly half the real TP traffic and
    # the simulator under-priced every TP config. Catch a regression here.
    if parsed and parsed[0] > 1:  # tp > 1
        tp, pp, mbs, gbs, gpus, n_layers, _chunks = parsed
        ar = [e for e in evs if e.get("type") == "collective"
              and (e.get("metadata") or {}).get("collective_type") == "AllReduce"]
        if ar:
            dp = max(1, gpus // (tp * pp))
            n_mb = gbs // (mbs * dp)
            per_mb = len(ar) / n_mb if n_mb else 0
            dirs = {(e.get("metadata") or {}).get("direction") for e in ar}
            # 2 AllReduce per layer per direction => ~4 x layers-ON-THIS-RANK per microbatch.
            # With PP the rank holds only num_layers/pp of them, so divide by pp or a
            # correct PP trace is flagged as missing half its collectives.
            expect_mb = 4 * (n_layers // pp) if n_layers else 0
            if expect_mb and per_mb < 0.75 * expect_mb:
                out.append((FAIL, f"{per_mb:.0f} AllReduce/microbatch vs ~{expect_mb} expected "
                                  f"({expect_mb/per_mb:.1f}x short) — backward TP collectives missing"))
            if dirs == {None}:
                out.append((WARN, "TP AllReduces carry no direction tag (pre-fix trace)"))
            elif not {"bwd"} & dirs:
                out.append((FAIL, "no BACKWARD TP AllReduce recorded"))

    if not kslots:
        return out or [(OK, f"span trace, {len(slots)} slots")]

    # ---- kernel-timing specific ----------------------------------------------------
    by_dir: dict[str, float] = {}
    for e in kslots:
        m = e["metadata"]
        by_dir[m.get("direction")] = by_dir.get(m.get("direction"), 0.0) + m["kernel_device_ms"]
    fwd, bwd = by_dir.get("fwd", 0.0), by_dir.get("bwd", 0.0)

    # 3) THE bug: backward must carry real kernel time. Backward is normally ~1-2x forward.
    if bwd <= 0:
        out.append((FAIL, f"bwd kernel time is ZERO (fwd={fwd:.0f}ms) — attribution broken"))
    elif fwd > 0:
        r = bwd / fwd
        if not 0.5 <= r <= 3.0:
            out.append((WARN, f"bwd/fwd = {r:.2f} (expect ~1-2)"))

    # 4) kernel time cannot exceed the wall span it ran in
    tot_k = tr.get("total_kernel_device_ms")
    tot_s = tr.get("total_span_ms")
    if tot_k and tot_s and tot_k > tot_s:
        out.append((FAIL, f"kernel {tot_k:.0f}ms > span {tot_s:.0f}ms (impossible)"))

    # 5) kernels landing outside every slot window mean the annotations miss real compute
    un = tr.get("unattributed_kernel_device_ms")
    if un is not None and tot_k:
        frac = un / (tot_k or 1) * 100
        if frac > 5:
            out.append((WARN, f"{frac:.1f}% of kernel time unattributed to any slot"))
    elif tot_k:
        out.append((WARN, "no unattributed_kernel_device_ms field — trace predates the "
                          "timestamp-attribution fix; re-capture before trusting per-slot data"))

    # 6) per-slot kernel time must scale with mbs (a mbs=N slot does N x the work)
    if parsed:
        mbs = parsed[2]
        fwd_slots = [e["metadata"]["kernel_device_ms"] for e in kslots
                     if e["metadata"].get("direction") == "fwd"]
        if fwd_slots and mbs > 1:
            out.append((OK, f"median fwd slot {sorted(fwd_slots)[len(fwd_slots)//2]:.1f}ms "
                            f"@mbs{mbs} (compare to mbs1 x{mbs}, minus efficiency gain)"))

    if not out:
        out.append((OK, f"{len(slots)} slots, fwd={fwd:.0f}ms bwd={bwd:.0f}ms"))
    return out


def check_workload(d: Path) -> list[tuple[str, str]]:
    """Config-side checks from the trace's saved workload.yaml."""
    wl = d / "workload.yaml"
    if not wl.exists():
        return [(WARN, "no workload.yaml (cannot verify config provenance)")]
    txt = wl.read_text()
    out = []
    import re
    m = re.search(r"padded[-_]vocab[-_]size[^0-9]*([0-9]+)", txt)
    if m and int(m.group(1)) != EXPECT_PADDED_VOCAB:
        out.append((WARN, f"padded_vocab={m.group(1)}, expected {EXPECT_PADDED_VOCAB}"))
    out += check_capture_matches_workload(d, txt)
    return out


# Settings that must agree between the workload.yaml the simulator reads and the
# capture_config the tracer recorded. A trace captured under different values is not
# comparable to a measurement taken under the workload's values.
#
# Added after 2026-08-04, when every real-PG trace turned out to be captured with
# data_parallel_sharding_strategy='optim_grads_params' (this Megatron's DEFAULT) while the
# workload.yaml -- and therefore the simulation -- said 'no_shard'. The captured runs were
# 1.4-5x slower than the archived campaign, scaling with microbatch count, and their
# host_ops were inflated by parameter all-gather bookkeeping. Nothing flagged it: the
# scorecard simply absorbed the mismatch into the calibrated host_cost_us.
_MUST_MATCH = {
    "data_parallel_sharding_strategy": "data-parallel-sharding-strategy",
    "micro_batch_size": "micro-batch-size",
    "global_batch_size": "global-batch-size",
    "tensor_model_parallel_size": "tensor-model-parallel-size",
    "pipeline_model_parallel_size": "pipeline-model-parallel-size",
    "sequence_parallel": "sequence-parallel",
    "recompute_granularity": "recompute-granularity",
}


def check_capture_matches_workload(d: Path, wl_txt: str) -> list[tuple[str, str]]:
    """Compare the tracer's recorded capture_config against the workload.yaml."""
    import json
    import re

    f = d / "trace_rank_0.json"
    if not f.exists():
        return []
    try:
        cap = json.loads(f.read_text()).get("capture_config") or {}
    except Exception:  # noqa: BLE001
        return []
    if not cap:
        # Pre-dates the self-describing trace format. Say so rather than pass silently:
        # these are exactly the traces whose provenance cannot be verified.
        return [(WARN, "trace has no capture_config (captured before the tracer recorded "
                       "it) — config provenance UNVERIFIABLE, not merely unchecked")]
    out = []
    for key, yaml_key in _MUST_MATCH.items():
        if key not in cap:
            continue
        m = re.search(rf"^\s*{re.escape(yaml_key)}:\s*(\S+)\s*$", wl_txt, re.MULTILINE)
        if not m:
            continue
        want, got = m.group(1).strip().strip("'\""), str(cap[key])
        if want.lower() != got.lower():
            out.append((FAIL, f"{yaml_key}: workload says {want!r} but the trace was "
                              f"CAPTURED with {got!r} — the trace does not describe the "
                              f"config being simulated"))
    return out



def main() -> None:
    dirs = [Path(a) for a in sys.argv[1:]] or DEFAULT_DIRS
    worst_fail = False
    for base in dirs:
        if not base.exists():
            print(f"\n{base}: MISSING")
            continue
        print(f"\n=== {base.relative_to(REPO) if base.is_relative_to(REPO) else base} ===")
        for cell in sorted(p for p in base.iterdir() if p.is_dir()):
            msgs = check_cell(cell) + check_workload(cell)
            lvl = FAIL if any(l == FAIL for l, _ in msgs) else (
                WARN if any(l == WARN for l, _ in msgs) else OK)
            worst_fail |= lvl == FAIL
            print(f"  [{lvl:4}] {cell.name}")
            for l, msg in msgs:
                if l != OK or lvl == OK:
                    print(f"           {l}: {msg}" if l != OK else f"           {msg}")
    print()
    if worst_fail:
        print("RESULT: FAIL — do not use these traces for verdicts until fixed.")
        sys.exit(1)
    print("RESULT: no FAILs.")


if __name__ == "__main__":
    main()
