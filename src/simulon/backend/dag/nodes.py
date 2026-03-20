from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional


@dataclass
class ComputeNode:
    node_id: int
    gpu_rank: int
    kernel: str  # layernorm|attn_qkv|attn_flash|attn_proj|mlp_linear1|mlp_act|mlp_linear2|embedding|logit
    layer_id: int
    microbatch_id: int
    pipeline_stage: int
    phase: str  # fwd|bwd_ig|bwd_wg
    duration_ms: Optional[float] = None
    start_ms: Optional[float] = None
    finish_ms: Optional[float] = None
    fused_kernels: list[str] = field(default_factory=list)  # non-empty when this node fuses multiple kernels


@dataclass
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
    duration_ms: Optional[float] = None
    start_ms: Optional[float] = None
    finish_ms: Optional[float] = None


@dataclass
class DAGEdge:
    src_node_id: int
    dst_node_id: int


@dataclass
class ExecutionDAG:
    compute_nodes: list[ComputeNode] = field(default_factory=list)
    comm_nodes: list[CommNode] = field(default_factory=list)
    edges: list[DAGEdge] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "compute_nodes": [asdict(n) for n in self.compute_nodes],
            "comm_nodes": [asdict(n) for n in self.comm_nodes],
            "edges": [asdict(e) for e in self.edges],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
