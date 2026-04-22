import pytest
from pydantic import ValidationError

from simulon.config.dc import ClusterSpec, DatacenterConfig, DatacenterMeta, GPUSpec, NodeSpec
from simulon.config.scenario import ScenarioConfig, WorkloadInstance
from simulon.config.workload import LLMSpec, MegatronParallelism, MegatronTraining, MegatronWorkload


def make_datacenter(num_nodes: int = 1, gpus_per_node: int = 4) -> DatacenterConfig:
    return DatacenterConfig(
        datacenter=DatacenterMeta(name="test-dc"),
        cluster=ClusterSpec(num_nodes=num_nodes),
        node=NodeSpec(
            gpus_per_node=gpus_per_node,
            gpu=GPUSpec(name="test-gpu"),
        ),
    )


def make_workload(name: str = "test", num_gpus: int = 1) -> MegatronWorkload:
    return MegatronWorkload(
        framework="megatron",
        model=LLMSpec(
            name=name,
            hidden_size=64,
            num_layers=2,
            num_heads=2,
            vocab_size=128,
        ),
        parallelism=MegatronParallelism(
            tp=1,
            pp=1,
            dp=num_gpus,
        ),
        training=MegatronTraining(
            num_gpus=num_gpus,
            global_batch_size=1,
            micro_batch_size=1,
            sequence_length=128,
        ),
    )


def test_duplicate_workload_names():
    dc = make_datacenter(num_nodes=2, gpus_per_node=4)
    with pytest.raises(ValidationError) as exc_info:
        ScenarioConfig(
            datacenter=dc,
            workloads=[
                WorkloadInstance(name="job_a", workload=make_workload("a", 4)),
                WorkloadInstance(name="job_a", workload=make_workload("b", 4)),
            ],
        )
    assert "duplicate workload names" in str(exc_info.value)


def test_missing_dependency():
    dc = make_datacenter(num_nodes=2, gpus_per_node=4)
    with pytest.raises(ValidationError) as exc_info:
        ScenarioConfig(
            datacenter=dc,
            workloads=[
                WorkloadInstance(
                    name="job_a",
                    workload=make_workload("a", 4),
                    start={"after_finish": ["nonexistent"]},
                ),
            ],
        )
    assert "unknown after_finish dependency names" in str(exc_info.value)


def test_dependency_cycle():
    dc = make_datacenter(num_nodes=2, gpus_per_node=4)
    with pytest.raises(ValidationError) as exc_info:
        ScenarioConfig(
            datacenter=dc,
            workloads=[
                WorkloadInstance(
                    name="job_a",
                    workload=make_workload("a", 4),
                    start={"after_finish": ["job_b"]},
                ),
                WorkloadInstance(
                    name="job_b",
                    workload=make_workload("b", 4),
                    start={"after_finish": ["job_a"]},
                ),
            ],
        )
    assert "cycle detected" in str(exc_info.value)


def test_gpu_over_allocation():
    dc = make_datacenter(num_nodes=1, gpus_per_node=4)
    with pytest.raises(ValidationError) as exc_info:
        ScenarioConfig(
            datacenter=dc,
            workloads=[
                WorkloadInstance(name="job_a", workload=make_workload("a", 8)),
                WorkloadInstance(name="job_b", workload=make_workload("b", 8)),
            ],
        )
    assert "workload GPU demand" in str(exc_info.value)
    assert "exceeds cluster capacity" in str(exc_info.value)


def test_backward_compat_singular_workload_alias():
    dc = make_datacenter(num_nodes=1, gpus_per_node=4)
    sc = ScenarioConfig(
        datacenter=dc,
        workload=make_workload("default", 4),
    )
    assert len(sc.workloads) == 1
    assert sc.workloads[0].name == "default"


def test_valid_multi_workload_parsing():
    dc = make_datacenter(num_nodes=2, gpus_per_node=4)
    sc = ScenarioConfig(
        datacenter=dc,
        workloads=[
            WorkloadInstance(name="job_a", workload=make_workload("a", 4)),
            WorkloadInstance(
                name="job_b",
                workload=make_workload("b", 4),
                start={"after_finish": ["job_a"]},
            ),
        ],
    )
    assert len(sc.workloads) == 2
    assert sc.workloads[0].name == "job_a"
    assert sc.workloads[1].name == "job_b"
    assert sc.workloads[1].start.after_finish == ["job_a"]
