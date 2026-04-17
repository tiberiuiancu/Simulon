"""Tests for DP gradient sync step phase."""

import pytest

from simulon.backend.dag import DAGTracerConfig
from simulon.backend.dag.megatron_tracer import MegatronDAGTracer
from simulon.backend.dag.megatron_tracer import _params_per_tp_rank
from simulon.backend.dag.populate import populate_dag
from simulon.config.common import DType
from simulon.config.dc import ClusterSpec, DatacenterConfig, DatacenterMeta, GPUSpec, KernelRun, NodeSpec
from typing import Optional

from simulon.config.workload import LLMSpec, MegatronParallelism, MegatronTraining, MegatronWorkload


def make_workload(
    tp: int = 1,
    pp: int = 1,
    dp: int = 1,
    ep: int = 1,
    num_layers: int = 2,
    hidden_size: int = 512,
    vocab_size: int = 32000,
    num_experts: Optional[int] = None,
    distributed_optimizer: bool = False,
    num_microbatches: int = 2,
) -> MegatronWorkload:
    num_gpus = tp * pp * dp * ep
    return MegatronWorkload(
        framework="megatron",
        model=LLMSpec(
            name="test-model",
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=8,
            vocab_size=vocab_size,
            num_experts=num_experts,
        ),
        parallelism=MegatronParallelism(
            tp=tp,
            pp=pp,
            ep=ep,
            num_microbatches=num_microbatches,
            distributed_optimizer=distributed_optimizer,
        ),
        training=MegatronTraining(
            num_gpus=num_gpus,
            global_batch_size=num_microbatches * dp,
            micro_batch_size=1,
            sequence_length=128,
            dtype=DType.bf16,
        ),
    )


def make_dc(gpus: int = 8) -> DatacenterConfig:
    return DatacenterConfig(
        datacenter=DatacenterMeta(name="test"),
        cluster=ClusterSpec(num_nodes=1),
        node=NodeSpec(gpus_per_node=gpus, gpu=GPUSpec(name="H100")),
    )


def trace(wl: MegatronWorkload, dc: DatacenterConfig | None = None):
    if dc is None:
        dc = make_dc()
    return MegatronDAGTracer(DAGTracerConfig()).trace(wl, dc)


# ---------------------------------------------------------------------------
# dp=1: no step comm nodes
# ---------------------------------------------------------------------------


def test_step_dp1_no_allreduce():
    wl = make_workload(tp=1, pp=1, dp=1)
    dag = trace(wl)
    step_nodes = [n for n in dag.comm_nodes if n.phase == "step"]
    assert step_nodes == []


# ---------------------------------------------------------------------------
# dp=2: AllReduce nodes with phase="step"
# ---------------------------------------------------------------------------


def test_step_dp2_has_allreduce():
    wl = make_workload(tp=1, pp=1, dp=2)
    dag = trace(wl)
    step_ar = [n for n in dag.comm_nodes if n.phase == "step" and n.collective_type == "AllReduce"]
    assert len(step_ar) > 0


# ---------------------------------------------------------------------------
# Step comm nodes are on the DP group (not TP peers)
# ---------------------------------------------------------------------------


def test_step_allreduce_on_dp_group():
    # tp=2, dp=2 → 4 GPUs: tp_group={0,1} and {2,3}; dp_group={0,2} and {1,3}
    wl = make_workload(tp=2, pp=1, dp=2)
    dag = trace(wl)
    step_nodes = [n for n in dag.comm_nodes if n.phase == "step"]
    assert len(step_nodes) > 0

    cross = [n for n in step_nodes if n.src_gpu != n.dst_gpu]
    assert len(cross) > 0

    # All cross-GPU flows should be between DP peers (differ by tp stride=2),
    # not TP peers (differ by 1).
    for n in cross:
        diff = abs(n.src_gpu - n.dst_gpu)
        # TP peers differ by 1; DP peers differ by tp=2
        assert diff != 1, f"Step comm between TP peers {n.src_gpu}→{n.dst_gpu}"


# ---------------------------------------------------------------------------
# distributed_optimizer=True → ReduceScatter + AllGather, no AllReduce
# ---------------------------------------------------------------------------


def test_step_distributed_optimizer_rs_ag():
    wl = make_workload(tp=1, pp=1, dp=2, distributed_optimizer=True)
    dag = trace(wl)
    step_nodes = [n for n in dag.comm_nodes if n.phase == "step"]
    types = {n.collective_type for n in step_nodes}
    assert "ReduceScatter" in types
    assert "AllGather" in types
    assert "AllReduce" not in types


# ---------------------------------------------------------------------------
# AllReduce byte size matches formula: 4 * total_params // pp
# ---------------------------------------------------------------------------


def test_step_bytes_formula_dense():
    tp, pp, dp = 1, 1, 2
    wl = make_workload(tp=tp, pp=pp, dp=dp, num_layers=2, hidden_size=512, vocab_size=32000)
    dag = trace(wl)

    model = wl.model
    expected_params = _params_per_tp_rank(model, tp=tp, ep=1)
    expected_bytes = 4 * expected_params // pp

    step_ar = [n for n in dag.comm_nodes if n.phase == "step" and n.collective_type == "AllReduce"]
    assert len(step_ar) > 0

    # Ring AllReduce with dp=2 decomposes into ReduceScatter (1 step, 2 flows) +
    # AllGather (1 step, 2 flows) = 4 CommNodes total.  Each flow carries
    # chunk_size = expected_bytes // dp.  Assert every CommNode holds exactly that.
    chunk_size = expected_bytes // dp
    for node in step_ar:
        assert node.bytes == chunk_size, (
            f"Expected each AllReduce CommNode to carry {chunk_size} bytes, "
            f"got {node.bytes} on flow {node.src_gpu}→{node.dst_gpu}"
        )


def test_step_bytes_formula_moe():
    tp, pp, dp, ep = 1, 1, 2, 2
    num_experts = 4
    wl = make_workload(tp=tp, pp=pp, dp=dp, ep=ep, num_experts=num_experts,
                       num_layers=2, hidden_size=512, vocab_size=32000,
                       num_microbatches=2)
    dag = trace(wl, make_dc(gpus=8))

    model = wl.model
    expected_params = _params_per_tp_rank(model, tp=tp, ep=ep)
    expected_bytes = 4 * expected_params // pp

    step_ar = [n for n in dag.comm_nodes if n.phase == "step" and n.collective_type == "AllReduce"]
    assert len(step_ar) > 0

    # Ring AllReduce with dp=2: each CommNode carries chunk_size = expected_bytes // dp.
    chunk_size = expected_bytes // dp
    for node in step_ar:
        assert node.bytes == chunk_size, (
            f"MoE: expected each AllReduce CommNode to carry {chunk_size} bytes, "
            f"got {node.bytes} on flow {node.src_gpu}→{node.dst_gpu}"
        )


# ---------------------------------------------------------------------------
# Step comm node has an edge from the last bwd_wg compute node
# ---------------------------------------------------------------------------


def test_step_ordered_after_bwd():
    wl = make_workload(tp=1, pp=1, dp=2, num_layers=1)
    dag = trace(wl)

    step_nodes = [n for n in dag.comm_nodes if n.phase == "step"]
    assert len(step_nodes) > 0
    step_ids = {n.node_id for n in step_nodes}

    # Build sets of bwd-phase node ids (both compute and comm sides).
    bwd_phases = {"bwd_ig", "bwd_wg"}
    bwd_compute_ids = {n.node_id for n in dag.compute_nodes if n.phase in bwd_phases}
    bwd_comm_ids = {n.node_id for n in dag.comm_nodes if n.phase in bwd_phases}
    bwd_node_ids = bwd_compute_ids | bwd_comm_ids

    # Find edge(s) that point INTO a step comm node (not step→step dependency edges)
    incoming = [e for e in dag.edges if e.dst_node_id in step_ids and e.src_node_id not in step_ids]
    assert len(incoming) > 0, "Expected at least one non-step edge leading into the step phase"

    # Every such predecessor must be a bwd-phase node, not a fwd node or unrelated node.
    for edge in incoming:
        assert edge.src_node_id in bwd_node_ids, (
            f"Step comm node predecessor {edge.src_node_id} is not a bwd-phase node"
        )


# ---------------------------------------------------------------------------
# Helpers for optimizer step tests (no DAG cache to avoid stale hits)
# ---------------------------------------------------------------------------


def no_cache_trace(wl: MegatronWorkload, dc: DatacenterConfig | None = None):
    """Trace without the on-disk DAG cache so tests always use the current implementation."""
    if dc is None:
        dc = make_dc()
    return MegatronDAGTracer(DAGTracerConfig(cache_dir=None)).trace(wl, dc)


# ---------------------------------------------------------------------------
# Optimizer step ComputeNode: dp=1
# ---------------------------------------------------------------------------


def test_adamw_node_present_dp1():
    """dp=1: exactly one AdamW ComputeNode per GPU in phase='step'."""
    wl = make_workload(tp=1, pp=1, dp=1)
    dag = no_cache_trace(wl)
    adamw_nodes = [n for n in dag.compute_nodes if n.kernel == "adamw"]
    num_gpus = 1
    assert len(adamw_nodes) == num_gpus


def test_adamw_node_fields_dp1():
    """dp=1: AdamW node has kernel='adamw', phase='step', layer_id=-1, microbatch_id=-1."""
    wl = make_workload(tp=1, pp=1, dp=1)
    dag = no_cache_trace(wl)
    adamw_nodes = [n for n in dag.compute_nodes if n.kernel == "adamw"]
    assert len(adamw_nodes) == 1
    node = adamw_nodes[0]
    assert node.kernel == "adamw"
    assert node.phase == "step"
    assert node.layer_id == -1
    assert node.microbatch_id == -1


def test_adamw_no_step_comm_nodes_dp1():
    """dp=1: no CommNodes in phase='step' (no gradient sync needed)."""
    wl = make_workload(tp=1, pp=1, dp=1)
    dag = no_cache_trace(wl)
    step_comm = [n for n in dag.comm_nodes if n.phase == "step"]
    assert step_comm == []


def test_adamw_wired_after_last_bwd_node_dp1():
    """dp=1: the AdamW node's only predecessor is a non-step node (last bwd node)."""
    wl = make_workload(tp=1, pp=1, dp=1, num_layers=2)
    dag = no_cache_trace(wl)
    adamw_nodes = [n for n in dag.compute_nodes if n.kernel == "adamw"]
    assert len(adamw_nodes) == 1
    opt_id = adamw_nodes[0].node_id

    step_node_ids = {n.node_id for n in dag.compute_nodes if n.phase == "step"}
    incoming = [e for e in dag.edges if e.dst_node_id == opt_id]
    # The optimizer must have at least one incoming edge
    assert len(incoming) > 0
    # All predecessors are non-step nodes
    for edge in incoming:
        assert edge.src_node_id not in step_node_ids


def test_adamw_extra_params_num_params_dp1():
    """dp=1: AdamW node has extra_params['num_params'] > 0."""
    wl = make_workload(tp=1, pp=1, dp=1)
    dag = no_cache_trace(wl)
    adamw_nodes = [n for n in dag.compute_nodes if n.kernel == "adamw"]
    assert len(adamw_nodes) == 1
    assert adamw_nodes[0].extra_params.get("num_params", 0) > 0


# ---------------------------------------------------------------------------
# Optimizer step ComputeNode: dp>1, dist_opt=False
# ---------------------------------------------------------------------------


def test_adamw_node_present_dp2_no_dist_opt():
    """dp=2, dist_opt=False: each GPU has exactly one AdamW step node."""
    wl = make_workload(tp=1, pp=1, dp=2, distributed_optimizer=False)
    dag = no_cache_trace(wl)
    adamw_nodes = [n for n in dag.compute_nodes if n.kernel == "adamw"]
    assert len(adamw_nodes) == 2  # 2 GPUs


def test_adamw_after_allreduce_dp2_no_dist_opt():
    """dp=2, dist_opt=False: all incoming edges to AdamW come from step AllReduce flows."""
    wl = make_workload(tp=1, pp=1, dp=2, num_layers=1, distributed_optimizer=False)
    dag = no_cache_trace(wl)

    adamw_nodes = [n for n in dag.compute_nodes if n.kernel == "adamw"]
    step_ar_ids = {n.node_id for n in dag.comm_nodes if n.phase == "step" and n.collective_type == "AllReduce"}
    step_ar_by_gpu: dict[int, set[int]] = {}
    for n in dag.comm_nodes:
        if n.phase == "step" and n.collective_type == "AllReduce":
            step_ar_by_gpu.setdefault(n.src_gpu, set()).add(n.node_id)

    # For each AdamW node, at least one predecessor is a step AllReduce flow
    for opt_node in adamw_nodes:
        incoming_src = {e.src_node_id for e in dag.edges if e.dst_node_id == opt_node.node_id}
        assert incoming_src & step_ar_ids, (
            f"GPU {opt_node.gpu_rank} adamw node has no edge from step AllReduce"
        )


def test_no_allgather_in_step_dp2_no_dist_opt():
    """dp=2, dist_opt=False: no AllGather in step phase (only AllReduce)."""
    wl = make_workload(tp=1, pp=1, dp=2, distributed_optimizer=False)
    dag = no_cache_trace(wl)
    step_ag = [n for n in dag.comm_nodes if n.phase == "step" and n.collective_type == "AllGather"]
    assert step_ag == []


# ---------------------------------------------------------------------------
# Optimizer step ComputeNode: dp>1, dist_opt=True
# ---------------------------------------------------------------------------


def test_adamw_between_rs_and_ag_dist_opt():
    """dp=2, dist_opt=True: ReduceScatter → AdamW → AllGather ordering in edges."""
    wl = make_workload(tp=1, pp=1, dp=2, num_layers=1, distributed_optimizer=True)
    dag = no_cache_trace(wl)

    adamw_nodes = [n for n in dag.compute_nodes if n.kernel == "adamw"]
    assert len(adamw_nodes) == 2

    step_rs_ids = {n.node_id for n in dag.comm_nodes if n.phase == "step" and n.collective_type == "ReduceScatter"}
    step_ag_ids = {n.node_id for n in dag.comm_nodes if n.phase == "step" and n.collective_type == "AllGather"}

    assert len(step_rs_ids) > 0, "ReduceScatter flows must be present in step phase"
    assert len(step_ag_ids) > 0, "AllGather flows must be present in step phase"

    for opt_node in adamw_nodes:
        opt_id = opt_node.node_id
        # Check: at least one RS flows into this AdamW node
        incoming_src = {e.src_node_id for e in dag.edges if e.dst_node_id == opt_id}
        assert incoming_src & step_rs_ids, (
            f"GPU {opt_node.gpu_rank} adamw node has no edge from ReduceScatter"
        )
        # Check: at least one AG flow comes OUT of this AdamW node
        outgoing_dst = {e.dst_node_id for e in dag.edges if e.src_node_id == opt_id}
        assert outgoing_dst & step_ag_ids, (
            f"GPU {opt_node.gpu_rank} adamw node has no edge to AllGather"
        )


def test_no_direct_rs_to_ag_edge_dist_opt():
    """dp=2, dist_opt=True: no direct ReduceScatter→AllGather edge; AdamW sits between them."""
    wl = make_workload(tp=1, pp=1, dp=2, num_layers=1, distributed_optimizer=True)
    dag = no_cache_trace(wl)

    step_rs_ids = {n.node_id for n in dag.comm_nodes if n.phase == "step" and n.collective_type == "ReduceScatter"}
    step_ag_ids = {n.node_id for n in dag.comm_nodes if n.phase == "step" and n.collective_type == "AllGather"}

    for edge in dag.edges:
        assert not (edge.src_node_id in step_rs_ids and edge.dst_node_id in step_ag_ids), (
            "Direct ReduceScatter→AllGather edge found; AdamW should be in between"
        )


def test_no_allreduce_in_step_dist_opt():
    """dp=2, dist_opt=True: no AllReduce in step phase (uses RS+AG instead)."""
    wl = make_workload(tp=1, pp=1, dp=2, distributed_optimizer=True)
    dag = no_cache_trace(wl)
    step_ar = [n for n in dag.comm_nodes if n.phase == "step" and n.collective_type == "AllReduce"]
    assert step_ar == []


# ---------------------------------------------------------------------------
# extra_params: num_params correctness
# ---------------------------------------------------------------------------


def test_adamw_num_params_dp1_formula():
    """dp=1: extra_params['num_params'] matches _params_per_tp_rank // pp."""
    tp, pp, dp = 1, 2, 1
    wl = make_workload(tp=tp, pp=pp, dp=dp, num_layers=2, hidden_size=512, vocab_size=32000)
    dag = no_cache_trace(wl, make_dc(gpus=tp * pp * dp))
    adamw_nodes = [n for n in dag.compute_nodes if n.kernel == "adamw"]
    expected = _params_per_tp_rank(wl.model, tp=tp, ep=1) // pp
    for node in adamw_nodes:
        assert node.extra_params["num_params"] == expected


def test_adamw_num_params_dist_opt_formula():
    """dp=2, dist_opt=True: extra_params['num_params'] matches _params_per_tp_rank // pp // dp."""
    tp, pp, dp = 1, 1, 2
    wl = make_workload(tp=tp, pp=pp, dp=dp, distributed_optimizer=True, num_layers=2, hidden_size=512)
    dag = no_cache_trace(wl)
    adamw_nodes = [n for n in dag.compute_nodes if n.kernel == "adamw"]
    expected = _params_per_tp_rank(wl.model, tp=tp, ep=1) // pp // dp
    for node in adamw_nodes:
        assert node.extra_params["num_params"] == expected


def test_adamw_num_params_no_dist_opt_formula():
    """dp=2, dist_opt=False: extra_params['num_params'] matches _params_per_tp_rank // pp (not // dp)."""
    tp, pp, dp = 1, 1, 2
    wl = make_workload(tp=tp, pp=pp, dp=dp, distributed_optimizer=False, num_layers=2, hidden_size=512)
    dag = no_cache_trace(wl)
    adamw_nodes = [n for n in dag.compute_nodes if n.kernel == "adamw"]
    expected = _params_per_tp_rank(wl.model, tp=tp, ep=1) // pp
    for node in adamw_nodes:
        assert node.extra_params["num_params"] == expected


# ---------------------------------------------------------------------------
# populate_dag: AdamW duration assignment
# ---------------------------------------------------------------------------


def test_populate_dag_assigns_adamw_duration():
    """populate_dag() assigns duration_ms to AdamW nodes using num_params lookup."""
    tp, pp, dp = 1, 1, 1
    wl = make_workload(tp=tp, pp=pp, dp=dp, num_layers=2, hidden_size=512, vocab_size=32000)
    dag = no_cache_trace(wl, make_dc(gpus=1))

    adamw_nodes = [n for n in dag.compute_nodes if n.kernel == "adamw"]
    assert len(adamw_nodes) == 1
    num_params = adamw_nodes[0].extra_params["num_params"]

    gpu_spec = GPUSpec(
        name="test-gpu",
        kernel_runs=[
            KernelRun(
                kernel="adamw",
                params={"num_params": num_params, "dtype": "bf16"},
                times_ms=[5.0],
            )
        ],
    )

    # Populate; cache key uses id(gpu_spec) so this fresh object is clean
    populate_dag(dag, wl, gpu_spec)

    for node in adamw_nodes:
        assert node.duration_ms == pytest.approx(5.0), (
            f"AdamW duration not assigned: got {node.duration_ms}"
        )


def test_populate_dag_adamw_uses_num_params_not_hidden_size():
    """populate_dag() does NOT use hidden_size/seq_len params for AdamW lookup."""
    tp, pp, dp = 1, 1, 1
    wl = make_workload(tp=tp, pp=pp, dp=dp, num_layers=2, hidden_size=512, vocab_size=32000)
    dag = no_cache_trace(wl, make_dc(gpus=1))

    adamw_nodes = [n for n in dag.compute_nodes if n.kernel == "adamw"]
    num_params = adamw_nodes[0].extra_params["num_params"]

    # Provide only a run with num_params (no hidden_size), ensure it still matches
    gpu_spec = GPUSpec(
        name="test-gpu",
        kernel_runs=[
            KernelRun(
                kernel="adamw",
                params={"num_params": num_params, "dtype": "bf16"},
                times_ms=[7.5],
            )
        ],
    )
    populate_dag(dag, wl, gpu_spec)

    for node in adamw_nodes:
        assert node.duration_ms == pytest.approx(7.5)
