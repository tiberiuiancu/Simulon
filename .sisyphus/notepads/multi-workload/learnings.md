# Learnings

- When testing multi-workload features, always verify cross-task integration (e.g., after_finish dependencies actually delay workload start times in the merged DAG replay).
- `--ignore-missing` CLI flag needs to be propagated through all export paths (GOAL, Chrome trace, etc.) to avoid failures when profiling data is incomplete.
- Pydantic v2 Union[Path, DatacenterConfig] parsing requires careful dict construction for programmatic test configs; Path is tried first.
- `per_gpu_finish` in the replayer is used for both tracking final GPU finish times AND start_offsets initialization. These two concerns must be separated (use `gpu_offsets` for offsets, `per_gpu_finish` for tracking) to avoid accidentally serializing independent operations on the same GPU.
