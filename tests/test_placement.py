from types import SimpleNamespace

from simulon.config.dc import ClusterSpec, DatacenterConfig, DatacenterMeta, NodeSpec
from simulon.config.placement import NodeSlice, place_workloads
from simulon.config.workload import (
    CollectiveWorkload,
    InferenceParallelism,
    InferenceRun,
    InferenceWorkload,
    MegatronParallelism,
    MegatronTraining,
    MegatronWorkload,
)


def make_datacenter(num_nodes: int = 8, gpus_per_node: int = 4) -> DatacenterConfig:
    return DatacenterConfig(
        datacenter=DatacenterMeta(name="test"),
        cluster=ClusterSpec(num_nodes=num_nodes),
        node=NodeSpec(gpus_per_node=gpus_per_node),
    )


def make_instance(name: str, workload):
    return SimpleNamespace(name=name, workload=workload)


def test_place_workloads_assigns_contiguous_slices():
    dc = make_datacenter()
    placements = place_workloads(
        [
            make_instance(
                "train",
                MegatronWorkload(
                    framework="megatron",
                    model="m",
                    parallelism=MegatronParallelism(),
                    training=MegatronTraining(
                        num_gpus=6,
                        global_batch_size=1,
                        micro_batch_size=1,
                        sequence_length=1,
                    ),
                ),
            ),
            make_instance(
                "infer",
                InferenceWorkload(
                    framework="inference",
                    model="m",
                    parallelism=InferenceParallelism(),
                    inference=InferenceRun(num_gpus=3, batch_size=1, seq_length=1),
                ),
            ),
        ],
        dc,
    )

    assert placements["train"] == NodeSlice(0, 1, 0, 7, 8)
    assert placements["infer"] == NodeSlice(2, 2, 8, 11, 4)


def test_collective_defaults_to_full_cluster():
    dc = make_datacenter(num_nodes=3, gpus_per_node=2)
    placements = place_workloads(
        [make_instance("collective", CollectiveWorkload(framework="collective", collective_type="AllReduce", message_size_bytes=1))],
        dc,
    )

    assert placements["collective"] == NodeSlice(0, 2, 0, 5, 6)


def test_collective_uses_explicit_num_gpus_when_present():
    dc = make_datacenter(num_nodes=5, gpus_per_node=4)
    collective = SimpleNamespace(framework="collective", num_gpus=5)
    placements = place_workloads([make_instance("collective", collective)], dc)

    assert placements["collective"] == NodeSlice(0, 1, 0, 7, 8)
