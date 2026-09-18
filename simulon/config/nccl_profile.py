"""Pydantic models for nccl-tests measurement profiles.

A NcclProfile is loaded from a <gpu>.nccl.yaml file and feeds into
calbusbw.py for algorithm selection and effective bandwidth derivation.
"""

from __future__ import annotations

from pydantic import BaseModel, model_validator


class NcclDataPoint(BaseModel):
    size_bytes: int
    bus_bw_GBps: float


class NcclAlgoMeasurements(BaseModel):
    ring: list[NcclDataPoint] = []
    nvls: list[NcclDataPoint] = []
    nvls_tree: list[NcclDataPoint] = []
    tree: list[NcclDataPoint] = []


class NcclProfile(BaseModel):
    """Measured NCCL bus-bandwidth curves for a cluster.

    REMOVED FIELD: ``launch_latency_ms``. It was a fixed per-collective cost fitted to a
    residual (0.911 ms at tp4/mbs1, then 22-26% too high at mbs2 -- it declines with
    microbatch size, so no constant works). ``NodeSpec.host_cost_us`` models the same
    physics from measured per-slot host time and supersedes it.

    It was deleted rather than defaulted to 0 because of how it failed: it was applied
    unless ``host_cost_us`` happened to be set, so any caller that passed a node template
    by name silently got 16,430 collectives/rank x 0.911 ms = ~15 s of phantom
    communication (hit while diagnosing the comm model, 2026-09-17). A term that is armed
    by default and disarmed by an unrelated setting is worse than no term. Setting it now
    raises instead of being ignored.
    """

    gpus_per_node: int = 8
    name: str | None = None
    AllReduce: NcclAlgoMeasurements = NcclAlgoMeasurements()
    AllGather: NcclAlgoMeasurements = NcclAlgoMeasurements()
    ReduceScatter: NcclAlgoMeasurements = NcclAlgoMeasurements()
    AllToAll: NcclAlgoMeasurements = NcclAlgoMeasurements()
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

    @model_validator(mode="before")
    @classmethod
    def _reject_launch_latency(cls, data):
        if isinstance(data, dict) and "launch_latency_ms" in data:
            raise ValueError(
                "launch_latency_ms was removed (see the class docstring): it is a fitted "
                "per-collective constant that does not hold across microbatch size, and it "
                "was applied silently whenever host_cost_us was unset. Set "
                "NodeSpec.host_cost_us instead -- calibrate it with "
                "experiments/host_cost_transfer.py."
            )
        return data

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
