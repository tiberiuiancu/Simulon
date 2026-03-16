from dataclasses import dataclass, field


@dataclass
class P2PFlow:
    flow_id: int
    src: int
    dst: int
    flow_size: int
    parent_flow_ids: list[int]
    child_flow_ids: list[int]
    channel_id: int
    chunk_id: int
    chunk_count: int
    conn_type: str  # "RING" | "NVLS"
