"""NVLS (NVLink SHARP) collective algorithms.

NVLS offloads reduction to the NVSwitch fabric, enabling in-network AllReduce
within a single NVLink domain (one node). Two variants exist in NCCL:

  NVLS       — single-node: each GPU gathers to NVSwitch, which reduces and
                scatters back. Latency ~23µs base (NCCL simple protocol).
  NVLS_TREE  — multi-node: intra-node NVLS combined with an inter-node tree
                for scale-out. Requires NVLink Switch System (GB200 NVL72+).

Key design constraint (learned the hard way):
  NVLS must only be selected for INTRA-NODE groups. The algorithm must be
  chosen per-collective based on whether all ranks share the same NVSwitch
  domain — NOT as a global flag applied to all collectives. Applying it to
  inter-node DP AllReduce or to AllGather/ReduceScatter is wrong.

References:
  - NCCL MockNccl.h: NCCL_ALGO_NVLS = 4, NCCL_ALGO_NVLS_TREE = 5
  - astra-sim MockNcclGroup.cc: genAllreduceNVLSFlowModels (single-node),
    generate_flow_model_nvls_tree_allreduce_up/down (multi-node tree)
  - astra-sim: NVSwitch IDs come from gp_info.NVSwitchs (topology config),
    NOT derived from max(group_ranks)+1 at collective decomposition time.

Not yet implemented.
"""

from simulon.collective.common import P2PFlow


def nvls_all_reduce(
    group_ranks: list[int],
    data_size: int,
    num_channels: int = 1,
    flow_id_start: int = 0,
) -> tuple[list[P2PFlow], int]:
    raise NotImplementedError(
        "NVLS AllReduce is not yet implemented. "
        "Use algorithm='ring' instead."
    )


def nvls_tree_all_reduce(
    group_ranks: list[int],
    data_size: int,
    num_channels: int = 1,
    flow_id_start: int = 0,
) -> tuple[list[P2PFlow], int]:
    raise NotImplementedError(
        "NVLS Tree AllReduce is not yet implemented. "
        "Use algorithm='ring' instead."
    )
