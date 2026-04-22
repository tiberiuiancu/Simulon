# Issues

1. **GOAL export ignores `--ignore-missing`** (src/simulon/cli/__init__.py:185)
   `write_goal_trace(dag, goal, start_offsets=output.start_offsets or None)` did not pass `ignore_missing` through. `dag_to_goal` then raised `ValueError` for any `ComputeNode.duration_ms` that was `None` (e.g., missing `embedding` kernel profile).
   **Fix**: Added `ignore_missing` parameter to `dag_to_goal` and `write_goal_trace`, and propagated it from CLI.

2. **start_offsets not applied in replayer** (src/simulon/backend/dag/replayer.py:342-347)
   `per_gpu_finish` was initialized with offset values but was never consulted when computing node `start_time`. The first node on an offset GPU started at `t=0` if it had no predecessors. This broke `after_finish` dependency semantics.
   **Fix**: Introduced a separate `gpu_offsets` dict (never mutated during replay) and used it in `start_time = max(pred_finish, gpu_offsets[gpu])` for both ComputeNode and CommNode.

3. **First replayer fix over-serialized independent CommNodes**
   Using `per_gpu_finish` (which is updated during replay) in `start_time` caused independent recvs on the same GPU to be serialized, breaking `test_overlapping_recvs_no_double_count`.
   **Fix**: Switched to immutable `gpu_offsets` so offsets are applied once without affecting subsequent independent operations.

4. **Datacenter config dict structure subtlety**
   Programmatic `ScenarioConfig.model_validate({...})` requires the datacenter dict to match `DatacenterConfig` exactly. Passing `{'datacenter': {'name': 'test'}, 'cluster': {'num_nodes': 1}, 'node': {'gpus_per_node': 4}}` triggered Path-parsing errors because Pydantic tried to coerce the dict to `Path` first due to `Union[Path, DatacenterConfig]`.
