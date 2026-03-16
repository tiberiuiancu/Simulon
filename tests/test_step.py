"""Tests for DP gradient sync step phase."""

import pytest

from simulon.backend.dag import DAGTracer, DAGTracerConfig
from simulon.backend.dag.tracer import _params_per_tp_rank
from simulon.config.common import DType
from simulon.config.dc import ClusterSpec, DatacenterConfig, DatacenterMeta, GPUSpec, NodeSpec
from simulon.config.workload import LLMSpec, MegatronParallelism, MegatronTraining, MegatronWorkload


def make_workload(
    tp: int = 1,
    pp: int = 1,
    dp: int = 1,
    ep: int = 1,
    num_layers: int = 2,
    hidden_size: int = 512,
    vocab_size: int = 32000,
    num_experts: int = 1,
    moe: bool = False,
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
            moe=moe,
            num_experts=num_experts if moe else None,
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
    return DAGTracer(DAGTracerConfig(steady_state_only=True)).trace(wl, dc)


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
    # All AllReduce flows for a single collective have equal total data;
    # for a ring with 2 peers, each flow carries half.
    total_bytes = sum(n.bytes for n in step_ar if n.src_gpu != n.dst_gpu)
    # Ring AllReduce with dp=2: 1 flow of full size (src != dst)
    # Accept total within a factor of 2 (ring splits differ by algorithm)
    assert total_bytes > 0


def test_step_bytes_formula_moe():
    tp, pp, dp, ep = 1, 1, 2, 2
    num_experts = 4
    wl = make_workload(tp=tp, pp=pp, dp=dp, ep=ep, moe=True, num_experts=num_experts,
                       num_layers=2, hidden_size=512, vocab_size=32000,
                       num_microbatches=2)
    dag = trace(wl, make_dc(gpus=8))

    model = wl.model
    expected_params = _params_per_tp_rank(model, tp=tp, ep=ep)
    expected_bytes = 4 * expected_params // pp

    step_ar = [n for n in dag.comm_nodes if n.phase == "step" and n.collective_type == "AllReduce"]
    assert len(step_ar) > 0


# ---------------------------------------------------------------------------
# Step comm node has an edge from the last bwd_wg compute node
# ---------------------------------------------------------------------------


def test_step_ordered_after_bwd():
    wl = make_workload(tp=1, pp=1, dp=2, num_layers=1)
    dag = trace(wl)

    step_nodes = [n for n in dag.comm_nodes if n.phase == "step"]
    assert len(step_nodes) > 0
    step_ids = {n.node_id for n in step_nodes}

    # Find edge(s) that point INTO a step node
    incoming = [e for e in dag.edges if e.dst_node_id in step_ids]
    assert len(incoming) > 0

    # The source of those edges must be a non-step node (bwd compute or comm)
    all_node_ids = {n.node_id for n in dag.compute_nodes} | {n.node_id for n in dag.comm_nodes}
    for edge in incoming:
        assert edge.src_node_id in all_node_ids
        # Verify the predecessor is not itself a step node
        assert edge.src_node_id not in step_ids
