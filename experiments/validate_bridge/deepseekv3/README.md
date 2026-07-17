Added num-gpus param, removed redundant vpp flag, also added:   disable-bias-linear: true
  no-bias-dropout-fusion: true
  swiglu: true

```
apptainer run --nv --bind vendor:/opt/simulon/vendor --bind ./scripts:/opt/simulon/scripts --bind ./experiments:/opt/simulon/experiments --bind ./examples:/opt/simulon/examples simulon-nemo.sif python3 scripts/bridge_to_simulon.py --recipe vendor/Megatron-Bridge/scripts/performance/configs/deepseek/deepseek_llm_pretrain.py --function deepseek_v3_pretrain_config_h100 --output experiments/validate_e2e/deepseekv3/workload.yaml --megatron-bridge-path vendor/Megatron-Bridge/src --config-arg precision=fp8_cs
```
