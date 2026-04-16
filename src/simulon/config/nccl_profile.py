"""Pydantic models for nccl-tests measurement profiles.

A NcclProfile is loaded from a <gpu>.nccl.yaml file and feeds into
calbusbw.py for algorithm selection and effective bandwidth derivation.
"""
from __future__ import annotations

from typing import Optional

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
    name: Optional[str] = None
    AllReduce: NcclAlgoMeasurements = NcclAlgoMeasurements()
    AllGather: NcclAlgoMeasurements = NcclAlgoMeasurements()
    ReduceScatter: NcclAlgoMeasurements = NcclAlgoMeasurements()
    AllToAll: NcclAlgoMeasurements = NcclAlgoMeasurements()
