"""Unit tests for MoE/EP support: LayerExpander and DAGTracer."""

import pytest

from simulon.backend.dag.layer_expander import LayerExpander
from simulon.backend.dag import DAGTracer, DAGTracerConfig
from simulon.config.common import DType
from simulon.config.dc import ClusterSpec, DatacenterConfig, DatacenterMeta, GPUSpec, NodeSpec
from simulon.config.workload import LLMSpec, MegatronParallelism, MegatronTraining, MegatronWorkload


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _moe_expander(**kwargs):
    defaults = dict(
        sublayer_type="moe",
        phase="fwd",
        gpu_rank=0,
        pipeline_stage=0,
        microbatch_id=0,
        layer_idx=0,
        tp_group_ranks=[0],
        activation_bytes=8192,
        node_id_start=0,
        ep_group_ranks=[0],
        moe_data_bytes=4096,
    )
    defaults.update(kwargs)
    return LayerExpander().expand_sublayer(**defaults)


def make_moe_workload(
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
    num_gpus: int = 4,
    num_layers: int = 1,
    hidden_size: int = 512,
    num_experts: int = 4,
    top_k: int = 2,
    global_batch_size: int = 4,
    num_microbatches: int = 4,
) -> MegatronWorkload:
    return MegatronWorkload(
        framework="megatron",
        model=LLMSpec(
            name="test-moe",
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=8,
            vocab_size=32000,
            moe=True,
            num_experts=num_experts,
            top_k=top_k,
        ),
        parallelism=MegatronParallelism(tp=tp, pp=pp, ep=ep, num_microbatches=num_microbatches),
        training=MegatronTraining(
            num_gpus=num_gpus,
            global_batch_size=global_batch_size,
            micro_batch_size=1,
            sequence_length=128,
            dtype=DType.bf16,
        ),
    )


def make_dc() -> DatacenterConfig:
    return DatacenterConfig(
        datacenter=DatacenterMeta(name="test"),
        cluster=ClusterSpec(num_nodes=1),
        node=NodeSpec(gpus_per_node=8, gpu=GPUSpec(name="H100")),
    )


# ---------------------------------------------------------------------------
# LayerExpander: MoE fwd
# ---------------------------------------------------------------------------


def test_moe_fwd_ep1_tp1_no_comm():
    """ep=1, tp=1: no comm stubs, three compute kernels."""
    c, comm, edges, nid = _moe_expander(ep_group_ranks=[0], tp_group_ranks=[0])
    assert comm == []
    assert [n.kernel for n in c] == ["moe_norm", "moe_route", "moe_expert"]
    assert nid == 3


def test_moe_fwd_ep2_tp1_two_alltoall():
    """ep=2, tp=1: dispatch + combine AllToAll, no AG/RS."""
    c, comm, edges, nid = _moe_expander(
        ep_group_ranks=[0, 1], tp_group_ranks=[0],
    )
    types = [n.collective_type for n in comm]
    assert types == ["AllToAll", "AllToAll"]
    assert len(c) == 3  # moe_norm, moe_route, moe_expert
    # nid = 3 compute + 2 AllToAll = 5
    assert nid == 5


def test_moe_fwd_ep1_tp2_ag_rs_no_alltoall():
    """ep=1, tp=2: AllGather + ReduceScatter, no AllToAll."""
    c, comm, edges, nid = _moe_expander(
        ep_group_ranks=[0], tp_group_ranks=[0, 1],
    )
    types = [n.collective_type for n in comm]
    assert "AllGather" in types
    assert "ReduceScatter" in types
    assert "AllToAll" not in types
    assert len(c) == 3


def test_moe_fwd_ep2_tp2_all_comm_types():
    """ep=2, tp=2: A2A(dispatch) → AG → compute → RS → A2A(combine).
    Order matches AICB MOEMLP: permutation (A2A→AG) then unpermutation (RS→A2A).
    """
    c, comm, edges, nid = _moe_expander(
        ep_group_ranks=[0, 1], tp_group_ranks=[0, 1],
    )
    types = [n.collective_type for n in comm]
    assert types[0] == "AllToAll"   # dispatch
    assert types[1] == "AllGather"
    assert types[-2] == "ReduceScatter"
    assert types[-1] == "AllToAll"  # combine
    assert types.count("AllToAll") == 2
    # nid = 1 A2A + 1 AG + 3 compute + 1 RS + 1 A2A = 7
    assert nid == 7


def test_moe_fwd_kernel_order():
    """fwd kernels are moe_norm → moe_route → moe_expert."""
    c, _, _, _ = _moe_expander(ep_group_ranks=[0, 1], tp_group_ranks=[0, 1])
    assert [n.kernel for n in c] == ["moe_norm", "moe_route", "moe_expert"]


def test_moe_fwd_alltoall_data_bytes():
    """AllToAll stubs carry moe_data_bytes, not activation_bytes."""
    _, comm, _, _ = _moe_expander(
        ep_group_ranks=[0, 1], tp_group_ranks=[0],
        activation_bytes=8192, moe_data_bytes=1111,
    )
    a2a_stubs = [n for n in comm if n.collective_type == "AllToAll"]
    for stub in a2a_stubs:
        assert stub.bytes == 1111


def test_moe_fwd_edges_form_linear_chain():
    """Edges connect nodes in the order they were added (linear chain)."""
    c, comm, edges, nid = _moe_expander(
        ep_group_ranks=[0, 1], tp_group_ranks=[0, 1], node_id_start=10,
    )
    # All node ids in order: AG=10, norm=11, route=12, A2A=13, expert=14, A2A=15, RS=16
    src_ids = [e.src_node_id for e in edges]
    dst_ids = [e.dst_node_id for e in edges]
    assert dst_ids == list(range(11, 17))
    assert src_ids == list(range(10, 16))


# ---------------------------------------------------------------------------
# LayerExpander: MoE bwd_ig
# ---------------------------------------------------------------------------


def test_moe_bwd_ig_ep2_tp2_kernel_order():
    """bwd_ig kernels are moe_expert → moe_route → moe_norm (reversed)."""
    c, _, _, _ = _moe_expander(
        phase="bwd_ig", ep_group_ranks=[0, 1], tp_group_ranks=[0, 1],
    )
    assert [n.kernel for n in c] == ["moe_expert", "moe_route", "moe_norm"]


def test_moe_bwd_ig_ep2_tp2_comm_types():
    """bwd_ig: AG → expert → A2A → route → norm → RS."""
    _, comm, _, _ = _moe_expander(
        phase="bwd_ig", ep_group_ranks=[0, 1], tp_group_ranks=[0, 1],
    )
    types = [n.collective_type for n in comm]
    assert types[0] == "AllGather"
    assert types[-1] == "ReduceScatter"
    assert types.count("AllToAll") == 1  # one combine_bwd AllToAll


def test_moe_bwd_ig_ep1_tp1_no_comm():
    """bwd_ig ep=1 tp=1: no comm."""
    _, comm, _, _ = _moe_expander(phase="bwd_ig", ep_group_ranks=[0], tp_group_ranks=[0])
    assert comm == []


# ---------------------------------------------------------------------------
# LayerExpander: MoE bwd_wg
# ---------------------------------------------------------------------------


def test_moe_bwd_wg_no_comm():
    """bwd_wg: no comm regardless of ep/tp."""
    _, comm, _, _ = _moe_expander(
        phase="bwd_wg", ep_group_ranks=[0, 1], tp_group_ranks=[0, 1],
    )
    assert comm == []


def test_moe_bwd_wg_kernel_order():
    """bwd_wg: moe_route → moe_expert."""
    c, _, _, _ = _moe_expander(phase="bwd_wg")
    assert [n.kernel for n in c] == ["moe_route", "moe_expert"]


def test_moe_bwd_wg_node_count():
    """bwd_wg: only 2 compute nodes, nid advances by 2."""
    _, _, _, nid = _moe_expander(phase="bwd_wg", node_id_start=5)
    assert nid == 7


# ---------------------------------------------------------------------------
# LayerExpander: node id continuity
# ---------------------------------------------------------------------------


def test_moe_node_id_start_respected():
    """node_id_start offset is applied to all generated nodes."""
    c, comm, _, nid = _moe_expander(
        ep_group_ranks=[0, 1], tp_group_ranks=[0], node_id_start=100,
    )
    # tp=1, ep=2: A2A(dispatch)=100, moe_norm=101, moe_route=102, moe_expert=103, A2A(combine)=104
    assert comm[0].node_id == 100  # dispatch A2A is first
    assert c[0].node_id == 101    # moe_norm follows
    # 3 compute + 2 A2A = 5 nodes → nid = 105
    assert nid == 105


# ---------------------------------------------------------------------------
# DAGTracer: global_rank formula
# ---------------------------------------------------------------------------


def test_global_rank_ep1_backward_compatible():
    """With ep=1, global_rank(dp, pp, ep=0, tp) == old formula dp*(pp*tp)+pp*tp+tp."""
    # Old formula: dp*(pp*tp) + pp_stage*tp + tp_rank
    # New formula: dp*(pp*ep*tp) + pp_stage*(ep*tp) + ep_rank*tp + tp_rank  with ep=1
    tp, pp, ep = 4, 3, 1

    def old_rank(dp_rank, pp_stage, tp_rank):
        return dp_rank * (pp * tp) + pp_stage * tp + tp_rank

    def new_rank(dp_rank, pp_stage, ep_rank, tp_rank):
        return dp_rank * (pp * ep * tp) + pp_stage * (ep * tp) + ep_rank * tp + tp_rank

    for dp in range(2):
        for p in range(pp):
            for t in range(tp):
                assert old_rank(dp, p, t) == new_rank(dp, p, 0, t)


def test_global_rank_ep2_formula():
    """With ep=2 the formula assigns unique ranks and covers all GPUs."""
    tp, pp, ep, dp = 2, 2, 2, 2
    total = dp * pp * ep * tp  # 16

    def rank(dp_rank, pp_stage, ep_rank, tp_rank):
        return dp_rank * (pp * ep * tp) + pp_stage * (ep * tp) + ep_rank * tp + tp_rank

    ranks = set()
    for d in range(dp):
        for p in range(pp):
            for e in range(ep):
                for t in range(tp):
                    ranks.add(rank(d, p, e, t))
    assert ranks == set(range(total))


# ---------------------------------------------------------------------------
# DAGTracer: MoE integration
# ---------------------------------------------------------------------------


def test_tracer_moe_ep1_no_alltoall():
    """moe=True, ep=1: no AllToAll nodes (single-rank EP is a no-op)."""
    wl = make_moe_workload(tp=1, pp=1, ep=1, num_gpus=2, num_layers=1, global_batch_size=4, num_microbatches=4)
    dc = make_dc()
    dag = DAGTracer(DAGTracerConfig()).trace(wl, dc)
    a2a = [n for n in dag.comm_nodes if n.collective_type == "AllToAll"]
    assert a2a == []


def test_tracer_moe_ep2_generates_alltoall():
    """moe=True, ep=2: AllToAll comm nodes are present."""
    # tp=1, pp=1, ep=2, dp=1 → 2 GPUs
    wl = make_moe_workload(tp=1, pp=1, ep=2, num_gpus=2, num_layers=1, global_batch_size=4, num_microbatches=4)
    dc = make_dc()
    dag = DAGTracer(DAGTracerConfig()).trace(wl, dc)
    a2a = [n for n in dag.comm_nodes if n.collective_type == "AllToAll"]
    assert len(a2a) > 0


def test_tracer_moe_kernels_in_dag():
    """DAG contains moe_route and moe_expert compute nodes when moe=True."""
    wl = make_moe_workload(tp=1, pp=1, ep=1, num_gpus=1, num_layers=1, global_batch_size=1, num_microbatches=1)
    dc = make_dc()
    dag = DAGTracer(DAGTracerConfig()).trace(wl, dc)
    kernels = {n.kernel for n in dag.compute_nodes}
    assert "moe_route" in kernels
    assert "moe_expert" in kernels


def test_tracer_moe_no_mlp_kernels():
    """When moe=True, mlp_linear1/mlp_linear2 are absent."""
    wl = make_moe_workload(tp=1, pp=1, ep=1, num_gpus=1, num_layers=1, global_batch_size=1, num_microbatches=1)
    dc = make_dc()
    dag = DAGTracer(DAGTracerConfig()).trace(wl, dc)
    kernels = {n.kernel for n in dag.compute_nodes}
    assert "mlp_linear1" not in kernels
    assert "mlp_linear2" not in kernels


def test_tracer_moe_ep2_alltoall_on_ep_group():
    """AllToAll flows are between EP group members, not TP peers."""
    # tp=2, ep=2, pp=1, dp=1 → 4 GPUs
    # EP group for ep_rank∈{0,1} with tp_rank=0: GPUs 0 and 2
    # EP group for ep_rank∈{0,1} with tp_rank=1: GPUs 1 and 3
    wl = make_moe_workload(tp=2, pp=1, ep=2, num_gpus=4, num_layers=1, global_batch_size=4, num_microbatches=4)
    dc = make_dc()
    dag = DAGTracer(DAGTracerConfig()).trace(wl, dc)

    a2a = [n for n in dag.comm_nodes if n.collective_type == "AllToAll"]
    assert len(a2a) > 0

    # AllToAll src and dst should differ (cross-rank communication)
    cross = [n for n in a2a if n.src_gpu != n.dst_gpu]
    assert len(cross) > 0


def test_tracer_moe_tp2_still_generates_ag_rs():
    """With moe=True and tp=2, AllGather/ReduceScatter are still present."""
    wl = make_moe_workload(tp=2, pp=1, ep=1, num_gpus=2, num_layers=1, global_batch_size=2, num_microbatches=2)
    dc = make_dc()
    dag = DAGTracer(DAGTracerConfig()).trace(wl, dc)
    ag = [n for n in dag.comm_nodes if n.collective_type == "AllGather"]
    rs = [n for n in dag.comm_nodes if n.collective_type == "ReduceScatter"]
    assert len(ag) > 0
    assert len(rs) > 0


def test_tracer_dense_ep1_unchanged():
    """Dense model (moe=False) with ep=1 produces the same GPU count as before."""
    from simulon.config.workload import MegatronParallelism, MegatronTraining

    wl = MegatronWorkload(
        framework="megatron",
        model=LLMSpec(name="dense", hidden_size=512, num_layers=1, num_heads=8, vocab_size=32000),
        parallelism=MegatronParallelism(tp=2, pp=1, ep=1, num_microbatches=2),
        training=MegatronTraining(num_gpus=4, global_batch_size=4, micro_batch_size=1, sequence_length=64),
    )
    dc = make_dc()
    dag = DAGTracer(DAGTracerConfig()).trace(wl, dc)
    gpu_ids = {n.gpu_rank for n in dag.compute_nodes}
    assert gpu_ids == {0, 1, 2, 3}  # dp=2, tp=2 → 4 GPUs
