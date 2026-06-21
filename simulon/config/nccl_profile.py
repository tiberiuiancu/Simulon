"""Pydantic models for nccl-tests measurement profiles.

A NcclProfile is loaded from a <gpu>.nccl.yaml file and feeds into
calbusbw.py for algorithm selection and effective bandwidth derivation.
"""

from __future__ import annotations

from pydantic import BaseModel


class NcclDataPoint(BaseModel):
    size_bytes: int
    bus_bw_GBps: float


class NcclAlgoMeasurements(BaseModel):
    ring: list[NcclDataPoint] = []
    nvls: list[NcclDataPoint] = []
    nvls_tree: list[NcclDataPoint] = []
    tree: list[NcclDataPoint] = []


class NcclProfile(BaseModel):
    gpus_per_node: int = 8
    name: str | None = None
    AllReduce: NcclAlgoMeasurements = NcclAlgoMeasurements()
    AllGather: NcclAlgoMeasurements = NcclAlgoMeasurements()
    ReduceScatter: NcclAlgoMeasurements = NcclAlgoMeasurements()
    AllToAll: NcclAlgoMeasurements = NcclAlgoMeasurements()
    # Optional per-call latency added on top of the bandwidth-limited collective time.
    # Default 0.0 — only set if you have a reliable direct measurement of NCCL launch
    # overhead for this cluster (not a calibration residual).
    launch_latency_ms: float = 0.0
    # Optional sub-profiles measured at a specific intra-node communicator size
    # (number of GPUs participating). On NVLink fabrics busbw is NOT rank-count
    # independent — a TP=2 group over 2 GPUs reaches only ~1/3 of the 4-GPU busbw
    # because fewer NVLink links are engaged. The top-level measurements describe
    # the full-node communicator (gpus_per_node); by_nranks[k] overrides them for a
    # k-rank intra-node collective. Keyed by communicator rank-count.
    by_nranks: dict[int, "NcclProfile"] = {}
    # Optional sub-profiles measured at a specific MULTI-node topology, keyed by
    # "<nodes>n<gpus_per_node>g" (e.g. "2n4g", "16n1g"). The busbw measured by
    # nccl-tests at a given topology already bakes in NIC bandwidth, rail count and
    # the inter-node fabric, so when a collective's topology matches one of these
    # the duration is taken directly from real measurement instead of the modelled
    # NIC-efficiency table (calbusbw). Falls back to the model for unmeasured topologies.
    by_topology: dict[str, "NcclProfile"] = {}

    def for_nranks(self, nranks: int) -> "NcclProfile":
        """Return the sub-profile measured at this communicator size, else self.

        Falls back to the top-level (full-node) measurements when no rank-specific
        profile was provided — preserving behaviour for the calibrated rank count.
        """
        return self.by_nranks.get(nranks, self)

    @staticmethod
    def topology_key(num_nodes: int, gpus_per_node: int) -> str:
        return f"{num_nodes}n{gpus_per_node}g"

    def for_topology(self, num_nodes: int, gpus_per_node: int) -> "NcclProfile | None":
        """Return the sub-profile measured at this exact multi-node topology, or None.

        None signals "no direct measurement" so the caller falls back to the modelled
        (calbusbw) inter-node bandwidth.
        """
        return self.by_topology.get(self.topology_key(num_nodes, gpus_per_node))


NcclProfile.model_rebuild()
