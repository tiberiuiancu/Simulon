"""Tree collective algorithm.

Implements the binary tree reduce-broadcast pattern used by NCCL's ALGO_TREE.
Each rank sends to a parent and receives from children in the reduce phase,
then the reverse in the broadcast phase.

References:
  - NCCL MockNcclGroup.cc: generate_flow_model_tree_allreduce_up / _down
  - astra-sim: ncclChannelNode tree topology, gen_tree_intra/inter_channels

Not yet implemented.
"""

from simulon.collective.common import P2PFlow


def tree_all_reduce(
    group_ranks: list[int],
    data_size: int,
    num_channels: int = 1,
    flow_id_start: int = 0,
) -> tuple[list[P2PFlow], int]:
    raise NotImplementedError(
        "Tree AllReduce is not yet implemented. "
        "Use algorithm='ring' instead."
    )
