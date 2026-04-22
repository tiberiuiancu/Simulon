- Added standalone `MegatronWorkload` YAML templates under `templates/workload/`.
- Validated both files with `yaml.safe_load()` + `MegatronWorkload.model_validate()`.
- Use inline `LLMSpec` objects for workload templates; do not reference `templates/model/`.
- YAML LSP diagnostics were unavailable because `yaml-language-server` is not installed in this environment.
Added optional `num_gpus` to `CollectiveWorkload` with a `None` default, keeping `extra="forbid"` intact and preserving backward-compatible validation for existing collective configs.
Placement can stay duck-typed: unwrap an optional `.workload` wrapper, read `framework`, and derive raw GPU demand from `training.num_gpus`, `inference.num_gpus`, or a collective `num_gpus` fallback.
Greedy node placement is simplest as a running `current_node` pointer; `NodeSlice` can be computed directly from aligned GPU count without any budget validation.
