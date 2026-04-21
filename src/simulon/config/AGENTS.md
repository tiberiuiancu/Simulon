# config — Foundation Layer

7 files. Highest fan-out in the project (13 modules depend on this package).

## OVERVIEW

Pydantic v2 models for all configuration: datacenter hardware, workload specification, scenario composition, and template resolution. Everything downstream depends on these types.

## STRUCTURE

```
config/
├── common.py       # DType enum, Cost dataclass, PowerModel (TDP/idle/load curves)
├── dc.py           # DatacenterConfig, GPUSpec, KernelRun, NodeSpec, NICSpec, SwitchSpec, ClusterConfig
├── workload.py     # MegatronWorkload, CollectiveWorkload, InferenceWorkload, LLMSpec, Parallelism, Training
├── scenario.py     # ScenarioConfig — top-level container (datacenter + workload + collective)
├── nccl_profile.py # NCCLProfile, NCCLEntry — parsed nccl-tests JSON bandwidth data
├── resolve.py      # Factory functions: resolve_gpu_spec, resolve_node_spec, resolve_nccl_profile, resolve_scale_out
└── __init__.py     # Empty
```

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Add GPU hardware field | `dc.py` → `GPUSpec` | Also update `templates/gpu/*.yaml` |
| Add training parameter | `workload.py` → `Training` | Also update `docs/spec/config-workload.md` |
| Add parallelism dimension | `workload.py` → `Parallelism` | Also update `megatron_tracer.py` rank formula |
| New model architecture field | `workload.py` → `LLMSpec` | Used by layer_expander + profiling |
| New template resolution | `resolve.py` | Follow existing `resolve_gpu_spec` pattern |
| New workload type | `workload.py` + `scenario.py` | Add discriminated union variant |
| Network topology change | `dc.py` → `DatacenterConfig.network` | scale_up (intra-node) vs scale_out (inter-node) |

## CONVENTIONS

- **All models inherit `BaseModel`** — Pydantic v2 with `model_validate` / `model_dump`
- **Discriminated unions** — `ScenarioConfig.workload` uses `framework` field to dispatch: `megatron` → `MegatronWorkload`, `collective` → `CollectiveWorkload`
- **Template resolution is lazy** — `resolve.py` factory functions load YAML templates on demand, not at config parse time
- **Speed/latency as strings** — `"400Gbps"`, `"0.005ms"` parsed by `populate.py:_parse_speed/_parse_latency`
- **`kernel_runs`** — `GPUSpec.kernel_runs: list[KernelRun]` stores profiled timing data. Each run = (kernel, params_dict, duration_ms)
- **Optional fields with None** — many fields are Optional with None defaults; resolution fills them from templates

## ANTI-PATTERNS

- **dc.py is the coupling hotspot** — `DatacenterConfig` + `GPUSpec` imported by 13 modules. Changes here ripple everywhere
- **Speed strings not validated at parse time** — `"400Gbps"` is just a string in Pydantic; parsing happens in populate.py at simulation time
- **No schema versioning** — YAML configs have no version field. Breaking changes require manual migration
