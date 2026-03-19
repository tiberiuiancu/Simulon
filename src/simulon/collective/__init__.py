from typing import Protocol, runtime_checkable

from simulon.collective.common import P2PFlow
from simulon.collective.decompose import CollectiveResult, decompose_collective


@runtime_checkable
class CCLDecomposer(Protocol):
    def decompose(
        self,
        collective_type: str,
        group_ranks: list[int],
        data_size: int,
        num_channels: int,
        algorithm: str,
        flow_id_start: int,
    ) -> tuple[CollectiveResult, int]: ...


class DefaultCCLDecomposer:
    def decompose(
        self,
        collective_type: str,
        group_ranks: list[int],
        data_size: int,
        num_channels: int,
        algorithm: str,
        flow_id_start: int,
    ) -> tuple[CollectiveResult, int]:
        return decompose_collective(
            collective_type=collective_type,
            group_ranks=group_ranks,
            data_size=data_size,
            num_channels=num_channels,
            algorithm=algorithm,
            flow_id_start=flow_id_start,
        )


__all__ = [
    "P2PFlow",
    "CollectiveResult",
    "decompose_collective",
    "CCLDecomposer",
    "DefaultCCLDecomposer",
]
