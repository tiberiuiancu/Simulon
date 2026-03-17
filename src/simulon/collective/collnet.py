"""CollNet collective algorithms (Direct and Chain).

CollNet (Collective Network) offloads reduction to in-network compute elements
such as Mellanox SHARP switches. Two variants exist in NCCL:

  COLLNET_DIRECT  — each rank communicates directly with the in-network
                    aggregation node; no multi-hop chaining.
  COLLNET_CHAIN   — ranks are arranged in a chain; partial reductions flow
                    hop-by-hop through the network fabric before the final
                    result is broadcast back.

References:
  - NCCL MockNcclGroup.cc: genAllreduceCollnetFlowModels (direct/chain)
  - NCCL MockNccl.h: NCCL_ALGO_COLLNET_DIRECT = 2, NCCL_ALGO_COLLNET_CHAIN = 3
  - astra-sim: CollNet topology uses virtual aggregation-switch nodes analogous
    to NVLS NVSwitchReduceNode, but for Infiniband SHARP fabric.

Not yet implemented.
"""

from simulon.collective.common import P2PFlow


def collnet_direct_all_reduce(
    group_ranks: list[int],
    data_size: int,
    num_channels: int = 1,
    flow_id_start: int = 0,
) -> tuple[list[P2PFlow], int]:
    raise NotImplementedError(
        "CollNet Direct AllReduce is not yet implemented. "
        "Use algorithm='ring' instead."
    )


def collnet_chain_all_reduce(
    group_ranks: list[int],
    data_size: int,
    num_channels: int = 1,
    flow_id_start: int = 0,
) -> tuple[list[P2PFlow], int]:
    raise NotImplementedError(
        "CollNet Chain AllReduce is not yet implemented. "
        "Use algorithm='ring' instead."
    )
