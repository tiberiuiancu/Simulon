#!/usr/bin/env python3
"""Ground Simulon against every measured Megatron run on disk.

One command, three stages, one scorecard:

  1. INVENTORY   every oellm-autoexp / production run dir under --roots: the RESOLVED
                 Megatron argument dump (what actually ran, not the directory name) and the
                 steady-state iteration time (median over iterations >= --skip). Cached by
                 (path, size, mtime) in --cache so re-runs are seconds, not minutes.
  2. CLASSIFY    each unique measured config against the trace registry:
                   native         exact (model, tp, pp, sp, fp8, rc) trace, mbs exact
                   mbs-extrap     same, mbs scaled from the traced mbs (validated to 4x)
                   pp-derived     no trace at this PP; synthesized from a deeper-PP trace
                                  (pp_synth, layer-linear -- measured exact to ~2% on 32B)
                   BLOCKED:<why>  what would unlock it (layout/VPP, CP, MoE, no trace, ...)
                 plus a comm flag: whether the DP-group topology has a MEASURED NCCL curve
                 in the node template or falls back to the analytical model.
  3. SIMULATE    the simulable set, report sim vs measured per tier, so extrapolation
                 accuracy is scored SEPARATELY from native accuracy. That number is what
                 tells you how few calibration traces a new model or cluster needs.

  SELF-ANCHORS   (--anchors) every trace is also scored against its OWN capture job's wall
                 time (<trace>/.megatron.log): same kernels, same nodes, same day, zero extra
                 cluster time. --calibrate-host bisects the one calibrated scalar in the
                 compute path, host_cost_us, on the clean anchors -- so a new model or cluster
                 needs exactly its capture jobs and nothing else to be calibrated.

The one calibrated scalar is host_cost_us (per-op host cost); everything else -- kernel
times, collective sizes (incl. the synthesized dist-optimizer grad sync at DP>1), NCCL
curves, schedule -- is measured or derived. Verdicts: |delta| <= 5% GOOD, <= 10% ACCEPTABLE,
else FAIL; NOISY when the repeats of a config disagree by > 15% among themselves (then the
measurement, not the simulator, is the open question). Iteration time per log = median over
iterations > --skip of the SLOWEST reporting rank; a log holding several runs is split at
each Megatron argument dump and the last complete run is used.

    PYTHONPATH=. python experiments/ground.py --model qwen3-32b --list-blocked   # what unlocks the most
    PYTHONPATH=. python experiments/ground.py --model qwen3-32b --anchors --calibrate-host --derive-pp
    PYTHONPATH=. python experiments/ground.py --model qwen3-32b --anchors --derive-pp --host-cost-us 26 \
        --md results/ground/qwen3-32b.md --csv results/ground/qwen3-32b.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
logging.disable(logging.INFO)

import yaml  # noqa: E402

# ----------------------------------------------------------------------------------------
# 1. INVENTORY
# ----------------------------------------------------------------------------------------

DEFAULT_ROOTS = [
    "/e/project1/e-sta-openeurollm/vanosch1/oellm-autoexp/output",
    "/e/project1/e-sta-openeurollm/pre_production_training",
    "/e/project1/e-sta-openeurollm/pre_production_training_jj",
    "/e/project1/e-sta-openeurollm/production_training",
]
DEFAULT_CACHE = REPO / "results" / "ground" / "inventory_cache.json"
# Trusted registries: real process group, kernel-time + host-op fields, workload.yaml written
# by the capture job. Everything else on disk is fake-PG and/or span-only (see SWEEP_RUNBOOK).
DEFAULT_REGISTRY = [
    REPO / "templates" / "gpu" / "gh200_jupiter-32b-target" / "traces",
    REPO / "templates" / "gpu" / "gh200_jupiter-1p7b-realpg-v2" / "traces",
]

_ITER_RE = re.compile(
    r"iteration\s+(\d+)/\s*\d+.*?elapsed time per iteration \(ms\):\s*([\d.]+)"
    r".*?throughput per GPU \(TFLOP/s/GPU\):\s*([\d.]+)"
)
_ARG_RE = re.compile(r"^(?:\[[^\]]*\]:)?\s*([a-zA-Z0-9_]+)\s\.{3,}\s*(.+?)\s*$")

# Every resolved argument that decides whether -- and how -- a run can be simulated.
WANTED = [
    "world_size", "tensor_model_parallel_size", "pipeline_model_parallel_size",
    "num_layers_per_virtual_pipeline_stage", "pipeline_model_parallel_layout",
    "context_parallel_size", "expert_model_parallel_size", "num_experts",
    "micro_batch_size", "global_batch_size", "sequence_parallel",
    "recompute_activations", "recompute_granularity", "fp8", "tp_comm_overlap",
    "overlap_grad_reduce", "overlap_param_gather", "overlap_p2p_comm",
    "use_distributed_optimizer", "grad_reduce_in_bf16", "use_torch_fsdp2",
    "num_layers", "hidden_size", "ffn_hidden_size", "seq_length", "padded_vocab_size",
    "num_attention_heads", "num_query_groups", "transformer_impl",
]

# (num_layers, hidden, ffn) -> name. Anything else is reported as its dims.
MODELS = {
    (64, 5120, 25600): "qwen3-32b",
    (36, 4096, 12288): "qwen3-8b",
    (28, 2048, 6144): "qwen3-1.7b",
    (48, 2048, 6144): "qwen3-4b",
    (32, 4096, 14336): "llama3-8b",
}


def _coerce(v: str):
    v = v.strip()
    if v in ("True", "False"):
        return v == "True"
    if v == "None":
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


@dataclass
class Measurement:
    """One log file with >= 3 iterations = one measured sample of one config."""
    run_dir: str
    log: str
    job: str
    args: dict
    n_iters: int
    med_ms: float
    min_ms: float
    tflops: float

    # ---- derived identity --------------------------------------------------------
    @property
    def model(self) -> str:
        k = (self.args.get("num_layers"), self.args.get("hidden_size"), self.args.get("ffn_hidden_size"))
        return MODELS.get(k, f"L{k[0]}-h{k[1]}-f{k[2]}")

    @property
    def nodes(self) -> int:
        ws = self.args.get("world_size") or 0
        return max(1, ws // 4)

    def g(self, k, default=None):
        v = self.args.get(k)
        return default if v is None else v

    @property
    def tp(self): return int(self.g("tensor_model_parallel_size", 1))
    @property
    def pp(self): return int(self.g("pipeline_model_parallel_size", 1))
    @property
    def cp(self): return int(self.g("context_parallel_size", 1))
    @property
    def ep(self): return int(self.g("expert_model_parallel_size", 1))
    @property
    def mbs(self): return int(self.g("micro_batch_size", 1))
    @property
    def gbs(self): return int(self.g("global_batch_size", 0))
    @property
    def sp(self): return bool(self.g("sequence_parallel", False))
    @property
    def fp8(self): return bool(self.g("fp8", None))
    @property
    def rc(self):
        return bool(self.g("recompute_activations", False)) or bool(self.g("recompute_granularity", None))
    @property
    def vpp(self): return self.g("num_layers_per_virtual_pipeline_stage", None)
    @property
    def layout(self):
        v = self.g("pipeline_model_parallel_layout", None)
        return v if v not in (None, "None", "") else None
    @property
    def dp(self):
        return max(1, (self.args.get("world_size") or 0) // (self.tp * self.pp * self.cp * max(1, self.ep if self.g("num_experts") else 1)))
    @property
    def m(self):
        """Microbatches per DP rank -- the schedule's actual shape."""
        return self.gbs // (self.mbs * self.dp) if self.dp and self.mbs else 0

    @property
    def key(self) -> tuple:
        """Unique measured configuration (everything that changes the iteration time)."""
        return (self.model, self.nodes, self.tp, self.pp, self.cp, self.ep, self.mbs, self.gbs,
                self.g("seq_length"), self.sp, self.fp8, self.rc, self.vpp, self.layout,
                bool(self.g("tp_comm_overlap", False)), bool(self.g("overlap_grad_reduce", False)),
                bool(self.g("overlap_p2p_comm", False)), bool(self.g("grad_reduce_in_bf16", False)))


def _parse_log(path: Path, skip: int) -> Measurement | None:
    # A log may hold SEVERAL runs back to back (a requeue or a resubmit into the same
    # file). Each Megatron argument dump opens a new segment; iterations belong to the
    # segment whose dump preceded them. Taking the first dump's args for the whole file
    # mislabelled a seq_length=2560 rerun as the seq_length=4096 config it shared a log
    # with (16n PP1: 6474 ms filed under a 12 s config, a 90% "spread" that was not noise).
    segments: list[tuple[dict, list]] = [({}, [])]
    try:
        with open(path, errors="replace") as f:
            for line in f:
                m = _ARG_RE.match(line)
                if m and m.group(1) in WANTED:
                    args, its = segments[-1]
                    if m.group(1) in args or its:      # repeated key or args after iterations -> new run
                        segments.append(({}, []))
                        args, its = segments[-1]
                    args[m.group(1)] = _coerce(m.group(2))
                    continue
                m = _ITER_RE.search(line)
                if m:
                    segments[-1][1].append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    except OSError:
        return None
    usable = [(a, i) for a, i in segments if len(i) >= 3 and "world_size" in a]
    if not usable:
        return None
    args, its = usable[-1]        # the last complete run in the file
    # Every node's local rank 0 may print the iteration line, and the per-rank "elapsed
    # time" differs by rank skew (16n PP1 job 581697: 5.5-6.6 s across nodes for one
    # iteration). The iteration is over when the SLOWEST rank is done, so collapse
    # duplicates of one iteration to their max before taking the median over iterations.
    by_it: dict[int, tuple[float, float]] = {}
    for it, ms, tf in its:
        cur = by_it.get(it)
        by_it[it] = (ms, tf) if cur is None or ms > cur[0] else cur
    its = [(it, ms, tf) for it, (ms, tf) in sorted(by_it.items())]
    if len(its) < 3:
        return None
    steady = [x for x in its if x[0] > skip] or its
    job = re.search(r"(\d{5,})", path.name)
    return Measurement(
        run_dir=str(path.parent if path.parent.name != "logs" else path.parent.parent),
        log=str(path), job=job.group(1) if job else "?", args=args, n_iters=len(its),
        med_ms=statistics.median(x[1] for x in steady), min_ms=min(x[1] for x in steady),
        tflops=statistics.median(x[2] for x in steady),
    )


def inventory(roots: list[str], cache_path: Path, skip: int, refresh: bool) -> list[Measurement]:
    cache: dict = {}
    if cache_path.exists() and not refresh:
        cache = json.loads(cache_path.read_text())
    out: list[Measurement] = []
    seen: set[str] = set()
    n_new = 0
    for root in roots:
        root_p = Path(root)
        if not root_p.is_dir():
            continue
        # run dirs are either root/<run>/ or root/<group>/<run>/; logs sit in the run dir
        # or in its logs/ subdir. Every log with >= 3 iterations is one measurement.
        for log in list(root_p.glob("*/slurm-*.log")) + list(root_p.glob("*/logs/slurm-*.log")) \
                + list(root_p.glob("*/*/slurm-*.log")) + list(root_p.glob("*/*/logs/slurm-*.log")):
            key = str(log)
            if key in seen:
                continue
            seen.add(key)
            try:
                st = log.stat()
            except OSError:
                continue
            sig = f"v3:{st.st_size}:{int(st.st_mtime)}:{skip}"
            ent = cache.get(key)
            if ent is not None and ent.get("sig") == sig:
                if ent.get("m"):
                    out.append(Measurement(**ent["m"]))
                continue
            m = _parse_log(log, skip)
            cache[key] = {"sig": sig, "m": asdict(m) if m else None}
            n_new += 1
            if m:
                out.append(m)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache))
    if n_new:
        print(f"inventory: parsed {n_new} new log(s); {len(out)} measurements total", file=sys.stderr)
    return out


# ----------------------------------------------------------------------------------------
# 2. TRACE REGISTRY + CLASSIFY
# ----------------------------------------------------------------------------------------

@dataclass
class TraceEntry:
    path: Path
    model: str
    tp: int
    pp: int
    sp: bool
    fp8: bool
    rc: bool
    ubo: bool            # tp_comm_overlap ON -> userbuffer kernels booked as compute (contaminated)
    real_pg: bool
    mbs: int
    m: int               # traced microbatches per DP rank (trim ceiling; replication is fenced for pp>1)
    kernel_time: bool    # carries kernel_device_ms (needed for the host model)
    host_ops: bool
    interleaved: bool    # captured with an explicit layout / VPP -> slots are per (mb, chunk)
    dp: int              # DP degree of the CAPTURE (1 => the DP collectives are synthesized)

    @property
    def quality(self) -> tuple:
        # higher is better: clean compute first, real PG, kernel-time, host_ops
        return (not self.ubo, self.real_pg, self.kernel_time, self.host_ops)


def scan_registry(dirs: list[Path]) -> list[TraceEntry]:
    out: list[TraceEntry] = []
    for base in dirs:
        for wl in base.glob("**/workload.yaml"):
            d = wl.parent
            r0 = d / "trace_rank_0.json"
            if not r0.exists():
                continue
            try:
                cfg = yaml.safe_load(wl.read_text())["config"]
                # only the top-level keys are needed; json.dump writes them before AND after
                # the (large) events array, so read both ends instead of parsing the whole file
                with open(r0, "rb") as f:
                    head = f.read(20000).decode(errors="replace")
                    f.seek(max(0, r0.stat().st_size - 20000))
                    head += f.read().decode(errors="replace")
            except Exception:
                continue
            g = lambda k, dft=None: cfg.get(k, cfg.get(k.replace("-", "_"), dft))  # noqa: E731
            tp, pp = int(g("tensor-model-parallel-size", 1)), int(g("pipeline-model-parallel-size", 1))
            ngpu = int(g("num-gpus", g("num_gpus", tp * pp)) or tp * pp)
            dp = max(1, ngpu // (tp * pp))
            mbs, gbs = int(g("micro-batch-size", 1)), int(g("global-batch-size", 0) or 0)
            dims = (g("num-layers"), g("hidden-size"), g("ffn-hidden-size"))
            out.append(TraceEntry(
                path=d, model=MODELS.get(tuple(dims), f"L{dims[0]}-h{dims[1]}-f{dims[2]}"),
                tp=tp, pp=pp, sp=bool(g("sequence-parallel", False)), fp8=bool(g("fp8-format", None)),
                rc=bool(g("recompute-activations", False)) or bool(g("recompute-granularity", None)),
                ubo=bool(g("tp-comm-overlap", False)),
                real_pg='"trace_real_process_group": true' in head,
                mbs=mbs, m=(gbs // (mbs * dp)) if gbs else 0,
                kernel_time='"total_kernel_device_ms"' in head or "kernel_device_ms" in head,
                host_ops='"total_host_ops"' in head,
                interleaved=bool(g("pipeline-model-parallel-layout")
                                 or g("num-layers-per-virtual-pipeline-stage")),
                dp=dp,
            ))
    return out


@dataclass
class Plan:
    """How one measured config will be simulated, or why it cannot be."""
    tier: str                      # native | mbs-extrap | pp-derived | BLOCKED
    reason: str = ""
    trace: TraceEntry | None = None
    src_pp: int | None = None      # for pp-derived
    comm: str = ""                 # measured | fallback
    notes: list[str] = field(default_factory=list)


def _target_chunks(m: Measurement, n_layers: int) -> int:
    """Virtual pipeline chunks per stage of the MEASURED config."""
    if m.layout:
        return max(1, len(str(m.layout).replace("\\", "").split("|")) // max(1, m.pp))
    if m.vpp and n_layers:
        return max(1, n_layers // (m.pp * int(m.vpp)))
    return 1


def dp_topology_key(m: Measurement) -> str:
    """Shape of the DP communicator as the node template names it: <nodes>n<gpus-per-node>g."""
    per_node = 4
    ranks_per_node_in_group = max(1, per_node // (m.tp * m.pp * m.cp)) if m.tp * m.pp * m.cp < per_node else 1
    nodes_in_group = max(1, m.dp // ranks_per_node_in_group)
    return f"{nodes_in_group}n{ranks_per_node_in_group}g"


def measured_topologies(node_template: str) -> set[str]:
    p = REPO / "templates" / "node" / f"{node_template}.yaml"
    try:
        d = yaml.safe_load(p.read_text())
        return set((d.get("nccl") or {}).get("by_topology", {}).keys())
    except Exception:
        return set()


def classify(m: Measurement, reg: list[TraceEntry], topos: set[str], derive_pp: bool,
             max_mbs_ratio: float, allow_span: bool = False) -> Plan:
    if m.cp > 1:
        return Plan("BLOCKED", "context parallel not modelled")
    if m.g("num_experts"):
        return Plan("BLOCKED", "MoE: no expert traces")
    if m.g("use_torch_fsdp2"):
        return Plan("BLOCKED", "FSDP2 not modelled")

    cands = [t for t in reg if (t.model, t.tp, t.sp, t.fp8, t.rc) == (m.model, m.tp, m.sp, m.fp8, m.rc)]
    if not cands:
        return Plan("BLOCKED", f"no trace for {m.model} tp{m.tp} sp={m.sp} fp8={m.fp8} rc={m.rc}")
    # Always synthesize from a PLAIN (non-interleaved) capture. An interleaved capture has
    # one slot per (microbatch, CHUNK) under its own layout, so replaying or PP-deriving from
    # it models whatever schedule it happened to be captured with (measured: deriving PP1
    # from a 4-chunk PP4 layout trace gave -70%). interleave_synth builds the target schedule
    # from per-layer blocks instead, which is layout-agnostic.
    cands = [t for t in cands if not t.interleaved]
    if not cands:
        return Plan("BLOCKED", f"no plain (non-interleaved) trace for {m.model} tp{m.tp} sp={m.sp}")
    if not allow_span:
        good = [t for t in cands if t.kernel_time and t.host_ops]
        if not good:
            return Plan("BLOCKED", f"no trace with kernel_device_ms+host_ops for {m.model} tp{m.tp} sp={m.sp} "
                                   f"fp8={m.fp8} rc={m.rc} (span-only traces: --allow-span-traces --no-host)")
        cands = good

    notes: list[str] = []
    comm_key = dp_topology_key(m)
    comm = "measured" if (m.dp == 1 or comm_key in topos) else f"fallback({comm_key})"

    def pick(pool: list[TraceEntry]) -> TraceEntry:
        # exact mbs beats extrapolation; then clean compute, real PG, kernel time, host_ops
        return max(pool, key=lambda t: ((t.mbs == m.mbs), t.quality, t.m))

    if m.layout or m.vpp:
        # Interleaved target: synthesize the schedule. Unlike the plain paths this is not
        # limited by the source's microbatch count -- the slots are built from per-layer
        # blocks, so any M can be generated.
        src = [t for t in cands if t.pp == m.pp] or cands
        t = pick(src)
        n_layers = int(m.g("num_layers", 0) or 0)
        chunks = _target_chunks(m, n_layers)
        if chunks < 1:
            return Plan("BLOCKED", "cannot determine virtual chunks per stage")
        if t.mbs != m.mbs and (m.mbs / t.mbs > max_mbs_ratio or t.mbs / m.mbs > max_mbs_ratio):
            return Plan("BLOCKED", f"mbs {t.mbs}->{m.mbs} beyond validated {max_mbs_ratio:g}x")
        return Plan("layout-synth", trace=t, comm=comm,
                    notes=notes + [f"{chunks} chunks/stage from pp{t.pp} plain"])

    same_pp = [t for t in cands if t.pp == m.pp]
    if same_pp:
        t = pick(same_pp)
        tier = "native" if t.mbs == m.mbs else "mbs-extrap"
        ratio = m.mbs / t.mbs
        if ratio > max_mbs_ratio or ratio < 1 / max_mbs_ratio:
            return Plan("BLOCKED", f"mbs {t.mbs}->{m.mbs} beyond validated {max_mbs_ratio:g}x")
        # trimming needs traced M >= needed M (replication is fenced off for pp>1)
        if t.m and m.m > t.m and m.pp > 1:
            return Plan("BLOCKED", f"needs M={m.m} microbatches, trace has {t.m} (pp>1 replication fenced)")
        if t.ubo:
            notes.append("trace has tp_comm_overlap ON: kernel time contaminated")
        if not t.real_pg:
            notes.append("fake-PG trace")
        return Plan(tier, trace=t, comm=comm, notes=notes)

    if derive_pp:
        deeper = [t for t in cands if t.pp > m.pp and t.pp % m.pp == 0]
        if deeper:
            # the SHALLOWEST valid source: fewest stages to regroup, so the least synthesis
            best_pp = min(t.pp for t in deeper)
            t = pick([t for t in deeper if t.pp == best_pp])
            if t.m and m.m > t.m and m.pp > 1:
                return Plan("BLOCKED", f"needs M={m.m}, source trace has {t.m}")
            return Plan("pp-derived", trace=t, src_pp=t.pp, comm=comm,
                        notes=notes + [f"synthesized from pp{t.pp}"])
    have = sorted({t.pp for t in cands})
    return Plan("BLOCKED", f"no trace at pp{m.pp} (have pp{have}); --derive-pp can synthesize from a deeper PP")


# ----------------------------------------------------------------------------------------
# 3. SIMULATE
# ----------------------------------------------------------------------------------------

def _interleaved_dir(t: TraceEntry, m: Measurement, chunks: int, n_layers: int) -> Path:
    from simulon.backend.dag.interleave_synth import derive_interleaved_from_dir
    # The prefix is a SCHEDULE VERSION, not decoration: a derived trace is only valid for the
    # interleave_synth that produced it, and this cache is keyed on the CONFIG, not the code.
    # Bump it whenever the emitted schedule changes -- a run otherwise silently reuses traces
    # built by the previous version and reproduces its results exactly (il2 -> il3: the
    # chunks=1 warmup fix, which was invisible until the stale cache was noticed).
    tag = f"pp{m.pp}x{chunks}_M{m.m}_{'layout' if m.layout else f'vpp{m.vpp}'}"
    dest = REPO / "results" / "ground" / "derived_traces" / f"il3__{t.path.name}__{tag}"
    if not (dest / "trace_rank_0.json").exists():
        derive_interleaved_from_dir(t.path, dest, t.pp, m.pp, chunks, m.m, n_layers,
                                    layout=str(m.layout) if m.layout else None)
    return dest


def _derived_dir(t: TraceEntry, target_pp: int) -> Path:
    from simulon.backend.dag.pp_synth import derive_pp_from_dir
    # versioned: bump when pp_synth's output changes so stale derived dirs are never reused
    dest = REPO / "results" / "ground" / "derived_traces" / f"v3__{t.path.parent.parent.name}__{t.path.name}__pp{target_pp}"
    if not (dest / "trace_rank_0.json").exists():
        derive_pp_from_dir(t.path, dest, t.pp, target_pp, num_microbatches=t.m or 64)
    return dest


def simulate(m: Measurement, plan: Plan, node_template: str, host_cost_us: float | None):
    from simulon.backend.analytical import simulate as run_simulation
    from simulon.config.dc import DatacenterConfig
    from simulon.config.scenario import ScenarioConfig
    from simulon.config.workload import MegatronWorkload

    t = plan.trace
    assert t is not None
    if plan.tier == "layout-synth":
        tdir = _interleaved_dir(t, m, _target_chunks(m, int(m.g("num_layers", 0) or 0)),
                                int(m.g("num_layers", 0) or 0))
    elif plan.tier == "pp-derived":
        tdir = _derived_dir(t, m.pp)
    else:
        tdir = t.path
    cfg = yaml.safe_load((tdir / "workload.yaml").read_text())["config"]
    cfg["num-gpus"] = m.nodes * 4
    cfg["global-batch-size"] = m.gbs
    cfg["micro-batch-size"] = m.mbs
    cfg["pipeline-model-parallel-size"] = m.pp
    if plan.tier == "layout-synth" and m.layout:
        cfg["pipeline-model-parallel-layout"] = str(m.layout)
    # the measured run's P2P mode decides whether the per-exchange device sync is modelled
    cfg["overlap-p2p-comm"] = bool(m.g("overlap_p2p_comm", False))
    # the run's DP grad-sync settings drive the synthesized dist-optimizer collectives
    cfg["overlap-grad-reduce"] = bool(m.g("overlap_grad_reduce", False))
    cfg["overlap-param-gather"] = bool(m.g("overlap_param_gather", False))
    cfg["grad-reduce-in-bf16"] = bool(m.g("grad_reduce_in_bf16", False))
    cfg.pop("fp8-param-gather", None)
    node = node_template if host_cost_us is None else {"from": node_template, "host_cost_us": host_cost_us}
    dc = DatacenterConfig.model_validate({"num_nodes": m.nodes, "node": node})
    wl = MegatronWorkload.model_validate({"framework": "megatron", "config": cfg, "traces_dir": str(tdir)})
    _dag, r = run_simulation(ScenarioConfig(datacenter=dc, workload=wl), overlap_async_collectives=True)
    return r


_CAP_ITER_RE = re.compile(r"iteration\s+(\d+)/\s*(\d+).*?elapsed time per iteration \(ms\):\s*([\d.]+)")


def anchor_measurement(t: TraceEntry, skip: int = 2) -> Measurement | None:
    """The trace's OWN capture run as a measurement: same kernels, same nodes, same day.

    Every `simulon trace` / real-PG capture logs Megatron's iteration times to
    <trace>/.megatron.log. The traced iterations (the last two, under the profiler) are
    excluded; the median of the untraced steady iterations is the wall time the trace's
    kernels were captured at. Simulating that exact config is the purest grounding there
    is -- and it costs no cluster time, so it is the calibration source of choice when a
    model or cluster is new. Returns None when the log is missing.
    """
    log = t.path / ".megatron.log"
    if not log.exists():
        return None
    its = [(int(m.group(1)), int(m.group(2)), float(m.group(3)))
           for m in (_CAP_ITER_RE.search(l) for l in log.read_text(errors="replace").splitlines()) if m]
    if not its:
        return None
    total = its[-1][1]
    steady = [ms for it, _, ms in its if it > skip and it <= total - 2] or [ms for _, _, ms in its]
    cfg = yaml.safe_load((t.path / "workload.yaml").read_text())["config"]
    g = lambda k, dft=None: cfg.get(k, dft)  # noqa: E731
    args = {
        "world_size": int(g("num-gpus", t.tp * t.pp)), "tensor_model_parallel_size": t.tp,
        "pipeline_model_parallel_size": t.pp, "micro_batch_size": t.mbs,
        "global_batch_size": int(g("global-batch-size", 0)), "sequence_parallel": t.sp,
        "fp8": "hybrid" if t.fp8 else None, "recompute_activations": t.rc,
        "tp_comm_overlap": t.ubo, "overlap_p2p_comm": bool(g("overlap-p2p-comm", False)),
        "overlap_grad_reduce": bool(g("overlap-grad-reduce", False)),
        "overlap_param_gather": bool(g("overlap-param-gather", False)),
        "num_layers": g("num-layers"), "hidden_size": g("hidden-size"), "ffn_hidden_size": g("ffn-hidden-size"),
        "pipeline_model_parallel_layout": g("pipeline-model-parallel-layout"),
    }
    return Measurement(run_dir=str(t.path), log=str(log), job="capture", args=args, n_iters=len(its),
                       med_ms=statistics.median(steady), min_ms=min(steady), tflops=0.0)


def calibrate_host_cost(anchors: list[tuple[Measurement, Plan]], node_template: str,
                        lo: float = 0.0, hi: float = 60.0, iters: int = 7) -> float:
    """Solve host_cost_us so the mean signed error over the self-anchors is zero.

    Bisection on the mean signed delta (monotone in host_cost_us: more host time per op can
    only slow the replay). This is the ONE calibrated scalar in the compute path, and it is
    derived from the capture jobs' own wall time -- never from the runs being scored.
    """
    def mean_delta(h: float) -> float:
        ds = []
        for m, plan in anchors:
            r = simulate(m, plan, node_template, h)
            ds.append(r.total_time_ms / m.med_ms - 1)
        return statistics.mean(ds)
    d_lo, d_hi = mean_delta(lo), mean_delta(hi)
    print(f"calibrate: host_cost_us={lo:.1f} -> {d_lo:+.1%}   {hi:.1f} -> {d_hi:+.1%}", file=sys.stderr)
    if d_lo > 0:
        return lo
    if d_hi < 0:
        return hi
    for _ in range(iters):
        mid = (lo + hi) / 2
        d = mean_delta(mid)
        print(f"calibrate: host_cost_us={mid:.2f} -> {d:+.1%}", file=sys.stderr)
        if d < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def verdict(delta_pct: float, spread_pct: float | None = None) -> str:
    """|delta| <= 5 GOOD, <= 10 ACCEPTABLE, else FAIL -- unless the repeats of that config
    disagree by more than 15% among themselves, in which case no model verdict is possible
    and the row is marked NOISY (the measurement, not the simulator, is the open question)."""
    if spread_pct is not None and spread_pct > 15:
        return "NOISY"
    a = abs(delta_pct)
    return "GOOD" if a <= 5 else ("ACCEPTABLE" if a <= 10 else "FAIL")


# ----------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------

def _fmt_cfg(m: Measurement) -> str:
    bits = [f"{m.nodes}n", f"tp{m.tp}", f"pp{m.pp}", f"mbs{m.mbs}", f"gbs{m.gbs}", f"M{m.m}",
            "sp" if m.sp else "nosp"]
    if m.g("seq_length") and int(m.g("seq_length")) != 4096: bits.append(f"seq{m.g('seq_length')}")
    if m.fp8: bits.append("fp8")
    if m.rc: bits.append("rc")
    if m.vpp: bits.append(f"vpp{m.vpp}")
    if m.layout: bits.append("layout")
    if m.cp > 1: bits.append(f"cp{m.cp}")
    if m.g("num_experts"): bits.append(f"ep{m.ep}")
    if m.g("tp_comm_overlap"): bits.append("ubo")
    if m.g("overlap_grad_reduce"): bits.append("ogr")
    if m.g("overlap_p2p_comm"): bits.append("p2p")
    return " ".join(bits)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="*", default=DEFAULT_ROOTS)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--refresh", action="store_true", help="re-parse every log")
    ap.add_argument("--skip", type=int, default=2, help="iterations to skip before the median (default 2)")
    ap.add_argument("--model", default=None, help="only this model (e.g. qwen3-32b)")
    ap.add_argument("--max-nodes", type=int, default=None)
    ap.add_argument("--registry", nargs="*", type=Path, default=DEFAULT_REGISTRY,
                    help="trace dirs to scan (**/workload.yaml). Default: the real-PG registries only -- "
                         "the older hash-named fake-PG dirs carry build-inconsistent workload.yaml files "
                         "and span-only timings that need the deprecated launch_latency model")
    ap.add_argument("--allow-span-traces", action="store_true",
                    help="also use traces without kernel_device_ms/host_ops (span model; pair with --no-host)")
    ap.add_argument("--node", default="jupiter-gh200-4g")
    ap.add_argument("--host-cost-us", type=float, default=19.0,
                    help="two-resource host model; disables the fitted launch_latency_ms (None = off)")
    ap.add_argument("--no-host", action="store_true", help="replay without the host model")
    ap.add_argument("--derive-pp", action="store_true", help="synthesize missing PP depths from deeper traces")
    ap.add_argument("--max-mbs-ratio", type=float, default=4.0)
    ap.add_argument("--anchors", action="store_true",
                    help="also score each trace against its own capture run's wall time (self-anchor)")
    ap.add_argument("--calibrate-host", action="store_true",
                    help="derive host_cost_us from the clean self-anchors (bisection), then score with it")
    ap.add_argument("--list-blocked", action="store_true", help="only print what blocks each config")
    ap.add_argument("--no-sim", action="store_true")
    ap.add_argument("--md", type=Path, default=None)
    ap.add_argument("--csv", type=Path, default=None)
    a = ap.parse_args()
    host = None if a.no_host else a.host_cost_us

    meas = inventory(a.roots, a.cache, a.skip, a.refresh)
    if a.model:
        meas = [m for m in meas if m.model == a.model]
    if a.max_nodes:
        meas = [m for m in meas if m.nodes <= a.max_nodes]

    # group repeated measurements of the same config
    groups: dict[tuple, list[Measurement]] = defaultdict(list)
    for m in meas:
        groups[m.key].append(m)
    print(f"{len(meas)} measurements, {len(groups)} unique configs"
          + (f" for {a.model}" if a.model else "") + f"; skip={a.skip}, node={a.node}, host_cost_us={host}"
          + (" (to be calibrated)" if a.calibrate_host else ""))

    reg = scan_registry(a.registry)
    topos = measured_topologies(a.node)
    print(f"registry: {len(reg)} trace dirs  ({sum(t.real_pg for t in reg)} real-PG, "
          f"{sum(not t.ubo for t in reg)} clean-compute); measured DP topologies: {sorted(topos)}")

    rows = []
    blocked = Counter()

    # ---- self-anchors -------------------------------------------------------------
    anchors: list[tuple[Measurement, Plan]] = []
    if a.anchors or a.calibrate_host:
        for t in reg:
            if a.model and t.model != a.model:
                continue
            am = anchor_measurement(t, a.skip)
            if am is None:
                continue
            anchors.append((am, Plan("self-anchor", trace=t, comm="measured",
                                     notes=[] if not t.ubo else ["tp_comm_overlap ON: kernel time contaminated"])))
        print(f"self-anchors: {len(anchors)} capture runs with iteration logs")
    if a.calibrate_host:
        clean = [(m, p) for m, p in anchors if not p.trace.ubo and not m.layout]
        if not clean:
            sys.exit("no clean (tp_comm_overlap off, no layout) anchors to calibrate on")
        host = calibrate_host_cost(clean, a.node)
        print(f"calibrated host_cost_us = {host:.2f} from {len(clean)} clean anchors "
              f"(scored runs below use it; the previous default was 19.0 from the 1.7B grid)")
    for am, plan in anchors:
        if am.layout and not a.allow_span_traces:
            plan = Plan("BLOCKED", "layout: interleaved-1F1B DAG not built (step 2)", trace=plan.trace)
        row = dict(model=am.model, cfg=_fmt_cfg(am) + " [capture]", n=1, meas_ms=round(am.med_ms), spread_pct=None,
                   tflops=0.0, tier=plan.tier, reason=plan.reason, comm=plan.comm,
                   trace=str(plan.trace.path.relative_to(REPO)), notes="; ".join(plan.notes),
                   sim_ms=None, delta_pct=None, verdict="", compute_ms=None, exposed_comm_ms=None,
                   bubble_ms=None, host_stall_ms=None, run=am.run_dir, job="capture")
        if plan.tier != "BLOCKED" and not a.no_sim:
            try:
                r = simulate(am, plan, a.node, host)
                row.update(sim_ms=round(r.total_time_ms), delta_pct=(r.total_time_ms / am.med_ms - 1) * 100,
                           compute_ms=round(r.compute_ms), exposed_comm_ms=round(r.exposed_comm_ms),
                           bubble_ms=round(r.bubble_ms), host_stall_ms=round(r.host_stall_ms))
                row["verdict"] = verdict(row["delta_pct"])
            except Exception as exc:  # noqa: BLE001
                row.update(tier="ERROR", reason=f"{type(exc).__name__}: {str(exc)[:160]}")
        rows.append(row)
    for key, ms in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3], kv[0][6], kv[0][7])):
        rep = ms[0]
        meds = [x.med_ms for x in ms]
        med = statistics.median(meds)
        spread = (max(meds) / min(meds) - 1) * 100 if len(meds) > 1 else None
        plan = classify(rep, reg, topos, a.derive_pp, a.max_mbs_ratio, a.allow_span_traces)
        row = dict(model=rep.model, cfg=_fmt_cfg(rep), n=len(ms), meas_ms=round(med), spread_pct=spread,
                   tflops=round(statistics.median(x.tflops for x in ms), 1), tier=plan.tier, reason=plan.reason,
                   comm=plan.comm, trace=str(plan.trace.path.relative_to(REPO)) if plan.trace else "",
                   notes="; ".join(plan.notes), sim_ms=None, delta_pct=None, verdict="",
                   compute_ms=None, exposed_comm_ms=None, bubble_ms=None, host_stall_ms=None,
                   run=rep.run_dir, job=rep.job)
        if plan.tier == "BLOCKED":
            blocked[plan.reason.split(":")[0].split(" for ")[0]] += 1
        elif not a.no_sim and not a.list_blocked:
            try:
                r = simulate(rep, plan, a.node, host)
                row.update(sim_ms=round(r.total_time_ms), delta_pct=(r.total_time_ms / med - 1) * 100,
                           compute_ms=round(r.compute_ms), exposed_comm_ms=round(r.exposed_comm_ms),
                           bubble_ms=round(r.bubble_ms), host_stall_ms=round(r.host_stall_ms))
                row["verdict"] = verdict(row["delta_pct"], spread)
            except Exception as exc:  # noqa: BLE001
                row.update(tier="ERROR", reason=f"{type(exc).__name__}: {str(exc)[:160]}")
        rows.append(row)

    # ---- report --------------------------------------------------------------------
    sim_rows = [r for r in rows if r["sim_ms"] is not None]
    print()
    print(f"{'model':<11}{'config':<52}{'n':>2}{'meas':>8}{'±%':>5}{'sim':>8}{'Δ%':>7}  {'verdict':<10}{'tier':<11}{'comm':<16}notes")
    for r in rows:
        if r["tier"] == "BLOCKED" and not a.list_blocked:
            continue
        sp = f"{r['spread_pct']:.1f}" if r["spread_pct"] is not None else ""
        sim = f"{r['sim_ms']}" if r["sim_ms"] is not None else ""
        d = f"{r['delta_pct']:+.1f}" if r["delta_pct"] is not None else ""
        note = r["reason"] if r["tier"] in ("BLOCKED", "ERROR") else r["notes"]
        print(f"{r['model']:<11}{r['cfg']:<52}{r['n']:>2}{r['meas_ms']:>8}{sp:>5}{sim:>8}{d:>7}  "
              f"{r['verdict']:<10}{r['tier']:<11}{r['comm']:<16}{note}")

    print()
    for tier in ("self-anchor", "native", "mbs-extrap", "pp-derived", "layout-synth"):
        sub = [r for r in sim_rows if r["tier"] == tier]
        if sub:
            v = Counter(r["verdict"] for r in sub)
            # contaminated = the trace's kernel time includes userbuffer comm waits (tp_comm_overlap
            # ON capture); it is shown, flagged, and kept out of the accuracy statistic
            scored = [abs(r["delta_pct"]) for r in sub
                      if r["verdict"] != "NOISY" and "contaminated" not in r["notes"]]
            stat = (f"mean|Δ| {statistics.mean(scored):5.1f}%  max {max(scored):5.1f}%" if scored else "no scorable rows")
            print(f"{tier:<11} n={len(sub):<3} {stat}   GOOD {v['GOOD']}  ACCEPTABLE {v['ACCEPTABLE']}  "
                  f"FAIL {v['FAIL']}  NOISY {v['NOISY']}")
    for comm in ("measured",):
        sub = [r for r in sim_rows if r["comm"] == comm]
        fb = [r for r in sim_rows if r["comm"].startswith("fallback")]
        if sub or fb:
            f = lambda s: f"{statistics.mean(abs(r['delta_pct']) for r in s):.1f}%" if s else "-"  # noqa: E731
            print(f"comm       measured-topology n={len(sub)} mean|Δ| {f(sub)}   fallback n={len(fb)} mean|Δ| {f(fb)}")
    errs = [r for r in rows if r["tier"] == "ERROR"]
    if errs:
        print(f"errors     {len(errs)}: " + "; ".join(f"{r['cfg']}: {r['reason'][:80]}" for r in errs[:5]))
    nb = sum(blocked.values())
    if nb:
        print(f"\nBLOCKED {nb} of {len(rows)} configs -- what unlocks the most:")
        for why, n in blocked.most_common():
            print(f"  {n:>4}  {why}")

    if a.csv:
        a.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"csv -> {a.csv}")
    if a.md:
        a.md.parent.mkdir(parents=True, exist_ok=True)
        with open(a.md, "w") as f:
            f.write(f"# Grounding scorecard — {date.today()}\n\n")
            f.write(f"node `{a.node}`, host_cost_us={host}, skip={a.skip}; {len(meas)} measurements, {len(groups)} configs\n\n")
            f.write("| model | config | n | meas ms | ±% | sim ms | Δ% | verdict | tier | comm | notes |\n|---|---|--:|--:|--:|--:|--:|---|---|---|---|\n")
            for r in rows:
                sp = f"{r['spread_pct']:.1f}" if r["spread_pct"] is not None else ""
                sim = r["sim_ms"] if r["sim_ms"] is not None else ""
                d = f"{r['delta_pct']:+.1f}" if r["delta_pct"] is not None else ""
                note = r["reason"] if r["tier"] in ("BLOCKED", "ERROR") else r["notes"]
                f.write(f"| {r['model']} | {r['cfg']} | {r['n']} | {r['meas_ms']} | {sp} | {sim} | {d} | {r['verdict']} | {r['tier']} | {r['comm']} | {note} |\n")
        print(f"markdown -> {a.md}")


if __name__ == "__main__":
    main()
