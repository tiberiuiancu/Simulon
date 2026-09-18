from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class ComputeNode:
    node_id: int
    gpu_rank: int
    kernel: str  # layernorm|attn_qkv|attn_flash|attn_proj|mlp_linear1|mlp_act|mlp_linear2|embedding|logit|loss_ce|adamw
    layer_id: int
    microbatch_id: int
    pipeline_stage: int
    phase: str  # fwd|bwd_ig|bwd_wg
    duration_ms: float | None = None
    start_ms: float | None = None
    finish_ms: float | None = None
    fused_kernels: list[str] = field(
        default_factory=list
    )  # non-empty when this node fuses multiple kernels
    extra_params: dict = field(
        default_factory=dict
    )  # kernel-specific lookup params (e.g. num_params for adamw)
    is_extrapolated: bool = (
        False  # True when duration_ms was obtained via extrapolation, not exact/partial match
    )
    host_ops: float = 0.0  # framework ops issued here; cost = ops x NodeSpec.host_cost_us
    # CUDA launches attributed here. Carried for diagnostics only -- the replayer's
    # bounded-launch-queue model that once consumed it was falsified and removed
    # (see replayer.replay). Kept because it is measured and cheap, and it is the
    # denominator to check if a launch-rate effect is ever suspected again.
    kernel_ct: float = 0.0


@dataclass(slots=True)
class CommNode:
    node_id: int
    src_gpu: int
    dst_gpu: int
    bytes: int
    collective_type: str  # AllGather|ReduceScatter|AllReduce|PP_Send
    layer_id: int
    phase: str
    flow_id: int
    parent_flow_ids: list[int] = field(default_factory=list)
    duration_ms: float | None = None
    start_ms: float | None = None
    finish_ms: float | None = None


@dataclass(slots=True)
class CollectiveNode:
    node_id: int
    collective_type: str
    group_ranks: list[int]
    data_size: int
    name: str
    timestamp_ms: float
    layer_id: int
    phase: str
    algorithm: str
    num_channels: int
    duration_ms: float | None = None
    start_ms: float | None = None
    finish_ms: float | None = None
    host_ops: float = 0.0  # framework ops to issue this collective


@dataclass(slots=True)
class DAGEdge:
    src_node_id: int
    dst_node_id: int


@dataclass(slots=True)
class ExecutionDAG:
    compute_nodes: list[ComputeNode] = field(default_factory=list)
    comm_nodes: list[CommNode] = field(default_factory=list)
    collective_nodes: dict[int, CollectiveNode] = field(default_factory=dict)
    edges: list[DAGEdge] = field(default_factory=list)
    total_flops: float | None = None
    profiled_ranks: set[int] = field(default_factory=set)
    energy_kwh: float | None = None
    co2eq_kg: float | None = None
    # Per-rank issue order: the sequence in which that rank's host thread submits work.
    # The replayer walks it as a serial host resource, so a rank's GPU can only start a
    # node once the host has finished issuing everything ahead of it on that rank.
    rank_program: dict[int, list[int]] = field(default_factory=dict)
    # Nodes whose completion BLOCKS the issuing host thread, so it cannot run ahead past
    # them. Populated for pipeline P2P when overlap-p2p-comm is off (Megatron issues a
    # blocking recv there). Without this the host runs ahead across the whole iteration and
    # the replay reports a perfectly-pipelined schedule that hardware does not achieve.
    host_sync_nodes: set[int] = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "compute_nodes": [asdict(n) for n in self.compute_nodes],
            "comm_nodes": [asdict(n) for n in self.comm_nodes],
            "collective_nodes": [asdict(n) for n in self.collective_nodes.values()],
            "edges": [asdict(e) for e in self.edges],
            "total_flops": self.total_flops,
            "profiled_ranks": sorted(self.profiled_ranks),
            "energy_kwh": self.energy_kwh,
            "co2eq_kg": self.co2eq_kg,
            "rank_program": {str(r): ids for r, ids in self.rank_program.items()},
            "host_sync_nodes": sorted(self.host_sync_nodes),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def add_compute_node(self, node: ComputeNode):
        self.compute_nodes.append(node)

    def add_comm_node(self, node: CommNode):
        self.comm_nodes.append(node)

    def add_collective_node(self, node: CollectiveNode):
        self.collective_nodes[node.node_id] = node

    def add_edge(self, edge: DAGEdge):
        self.edges.append(edge)
