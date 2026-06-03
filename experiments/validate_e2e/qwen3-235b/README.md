Added num-gpus additional param

```
apptainer run --nv --bind vendor:/opt/simulon/vendor --bind ./scripts:/opt/simulon/scripts --bind ./experiments:/opt/simulon/experiments --bind ./examples:/opt/simulon/examples simulon-nemo.sif python3 scripts/bridge_to_simulon.py --recipe vendor/Megatron-Bridge/scripts/performance/configs/qwen/qwen3_llm_pretrain.py --function qwen3_235b_a22b_pretrain_config_h100 --output experiments/validate_e2e/qwen3-235b/workload.yaml --megatron-bridge-path vendor/Megatron-Bridge/src --config-arg precision=fp8_cs
```

