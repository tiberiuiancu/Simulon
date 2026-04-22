- Added standalone `MegatronWorkload` YAML templates under `templates/workload/`.
- Validated both files with `yaml.safe_load()` + `MegatronWorkload.model_validate()`.
- Use inline `LLMSpec` objects for workload templates; do not reference `templates/model/`.
- YAML LSP diagnostics were unavailable because `yaml-language-server` is not installed in this environment.
Added optional `num_gpus` to `CollectiveWorkload` with a `None` default, keeping `extra="forbid"` intact and preserving backward-compatible validation for existing collective configs.
Placement can stay duck-typed: unwrap an optional `.workload` wrapper, read `framework`, and derive raw GPU demand from `training.num_gpus`, `inference.num_gpus`, or a collective `num_gpus` fallback.
Greedy node placement is simplest as a running `current_node` pointer; `NodeSlice` can be computed directly from aligned GPU count without any budget validation.
`ScenarioConfig` now accepts `workloads: list[WorkloadInstance]` and still expands legacy singular `workload:` input into a default-named instance with an empty `StartConfig`.
Cross-workload validation lives in a model validator: duplicate names, missing `after_finish` dependencies, dependency cycles, and GPU budget overflow all fail at parse time.
`ScenarioConfig.workload` remains as a compatibility property for single-workload callers; YAML LSP diagnostics were unavailable here because `basedpyright` is not installed.
Verified `_validate_workloads()` rejects duplicate names, missing dependencies, `after_finish` cycles, and GPU over-allocation with `ValueError`/`ValidationError`, while a valid workload still parses successfully.
GPU budget validation uses node-aligned demand: `max(gpus_per_node, ceil(raw_num_gpus / gpus_per_node) * gpus_per_node)`; 513 raw GPUs on 8-GPU nodes round to 520 and exceed a 512-GPU cluster.
13: `AnalyticalBackend` can resolve YAML-backed workload paths with `TypeAdapter(WorkloadConfig).validate_python()` after `yaml.safe_load()`.
14: `DatacenterConfig.model_dump()` + `DatacenterConfig.model_validate()` is the safest way to slice `cluster.num_nodes` without dropping nested network settings.
