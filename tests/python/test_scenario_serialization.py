import yaml

from simulon.config.common import Cost, DType
from simulon.config.dc import (
    ClusterSpec,
    CPUSpec,
    DatacenterConfig,
    DatacenterMeta,
    KernelRun,
    LinkSpec,
    NICSpec,
    NodeCoolingSpec,
    NodeSpec,
    GPUSpec,
    QueueDiscipline,
    RackCoolingSpec,
    RackSpec,
    ScaleOutSpec,
    ScaleOutTopologySpec,
    ScaleUpSpec,
    ScaleUpTopology,
    ScaleUpTechnology,
    SwitchSpec,
    TopologyType,
)
from simulon.config.scenario import ScenarioConfig
from simulon.config.workload import (
    LLMSpec,
    MegatronParallelism,
    MegatronTraining,
    MegatronWorkload,
)


def make_scenario() -> ScenarioConfig:
    return ScenarioConfig(
        datacenter=DatacenterConfig(
            datacenter=DatacenterMeta(
                name="Test Cluster",
                location="EU-West",
                profiles_dir="./templates",
                pue=1.3,
                electricity_cost_per_kwh=0.08,
                rack=RackSpec(
                    nodes_per_rack=8,
                    rack_units=42,
                    max_power_kw=120.0,
                    cost=Cost(value=12000, min=10000, max=15000),
                    cooling=RackCoolingSpec(
                        capacity_kw=120.0,
                        tdp_w=600.0,
                        cost=5000.0,  # scalar cost form
                    ),
                ),
            ),
            cluster=ClusterSpec(num_nodes=64),
            node=NodeSpec(
                gpus_per_node=8,
                num_switches_per_node=4,
                gpus_per_nic=1,
                gpu=GPUSpec(
                    name="H100-SXM5-80GB",
                    vendor="nvidia",
                    memory_capacity_gb=80.0,
                    tdp_w=700.0,
                    flops_multiplier=1.0,
                    cost=Cost(value=30000, min=25000, max=35000),
                    kernel_runs=[
                        KernelRun(
                            kernel="matmul",
                            params={"M": 4096, "N": 4096, "K": 4096, "dtype": "bf16"},
                            times_ms=[0.42, 0.41, 0.43, 0.42],
                        ),
                        KernelRun(
                            kernel="flash_attention",
                            params={"seq_len": 4096, "num_heads": 32, "head_dim": 128},
                            times_ms=[1.12, 1.11, 1.13],
                        ),
                    ],
                ),
                cpu=CPUSpec(
                    name="Intel Xeon Platinum 8480+",
                    vendor="intel",
                    sockets=2,
                    cores_per_socket=60,
                    memory_gb=512.0,
                    tdp_w=350.0,
                    cost=4000.0,  # scalar cost form
                    memory_cost_per_gb=5.0,
                ),
                cooling=NodeCoolingSpec(
                    tdp_w=200.0,
                    cost=Cost(value=1500),
                ),
            ),
            scale_up=ScaleUpSpec(
                topology=ScaleUpTopology.switched,
                technology=ScaleUpTechnology.nvlink,
                link_bandwidth="900Gbps",
                link_latency="0.000025ms",
                switch=SwitchSpec(
                    name="NVSwitch3",
                    tdp_w=110.0,
                    cost=Cost(value=3000),
                ),
            ),
            scale_out=ScaleOutSpec(
                nic=NICSpec(
                    name="ConnectX-7",
                    vendor="mellanox",
                    speed="400Gbps",
                    latency="0.0005ms",
                    tdp_w=25.0,
                    cost=2000.0,  # scalar cost form
                ),
                topology=ScaleOutTopologySpec(
                    type=TopologyType.fat_tree,
                    params={"k": 64, "num_tiers": 3, "oversubscription": 1.0},
                    switch=SwitchSpec(
                        name="Spectrum-4",
                        vendor="nvidia",
                        port_count=64,
                        port_speed="400Gbps",
                        buffer_per_port="32MB",
                        queue_discipline=QueueDiscipline.fq_codel,
                        queue_params={"flows": 1024, "target_delay": "5ms"},
                        tdp_w=300.0,
                        cost=Cost(value=50000),
                    ),
                    link=LinkSpec(
                        latency="0.0005ms",
                        error_rate=0.0,
                        cost=200.0,
                        cost_per_meter=2.5,
                    ),
                ),
            ),
        ),
        workload=MegatronWorkload(
            framework="megatron",
            model=LLMSpec(
                name="LLaMA-13B",
                hidden_size=5120,
                num_layers=40,
                num_heads=40,
                vocab_size=32000,
                ffn_hidden_size=13824,
                swiglu=True,
                moe=False,
            ),
            parallelism=MegatronParallelism(
                tp=4,
                pp=4,
                ep=1,
                dp=4,
                sp=True,
                vpp=2,
                distributed_optimizer=True,
            ),
            training=MegatronTraining(
                num_gpus=64,
                global_batch_size=1024,
                micro_batch_size=2,
                sequence_length=4096,
                dtype=DType.bf16,
                flash_attention=True,
                iterations=500,
            ),
        ),
    )


def test_scenario_yaml_round_trip():
    original = make_scenario()

    yaml_str = yaml.dump(
        original.model_dump(mode="json", by_alias=True, exclude_none=True),
        default_flow_style=False,
    )

    restored = ScenarioConfig.model_validate(yaml.safe_load(yaml_str))

    assert restored == original
