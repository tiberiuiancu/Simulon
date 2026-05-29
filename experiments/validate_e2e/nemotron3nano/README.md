I was unable to run this scenario due to OOM. 

To produce the configuration I ran:

```bash
apptainer run --nv --bind vendor:/opt/simulon/vendor --bind ./scripts:/opt/simulon/scripts --bind ./experiments:/opt/simulon/experiments --bind ./examples:/opt/simulon/examples simulon-nemo.sif python3 scripts/bridge_to_simulon.py --recipe vendor/Megatron-Bridge/scripts/performance/configs/nemotronh/nemotron_3_llm_pretrain.py --function nemotron_3_nano_pretrain_config_h100 --output experiments/validate_e2e/nemotron3nano/workload.yaml --megatron-bridge-path vendor/Megatron-Bridge/src --config-arg precision=fp8_cs
```

and additionally set `num-gpus: 16` and `disable-bias-linear: true`, both being required.

