# simulon

AI cluster simulator for LLM training workloads. Given a datacenter config and a
workload config, simulon generates a detailed communication/compute trace and feeds
it to a simulation backend to estimate training time, network utilization, and
performance bottlenecks.

---

## What it does

1. **Workload trace generation** — parses a `MegatronWorkload` config (model
   architecture + parallelism strategy) and produces a `WorkloadTrace`: an ordered
   sequence of `CommOp` and `ComputeOp` records representing one training iteration
   on a representative GPU.

2. **GPU profiling** — the `simulon profile gpu` CLI benchmarks transformer kernels
   on the local GPU and writes a hardware template YAML with per-kernel timing data.
   The trace generator uses this data to annotate each `ComputeOp` with a measured
   `compute_time_us`.

3. **Simulation backends** — a `Backend` ABC accepts a `ScenarioConfig` (datacenter
   + workload) and returns simulation results. Two backends are planned: an analytical
   model (fast, closed-form) and an NS-3 packet-level simulator.

---

## Project structure

```
simulon/
├── src/simulon/
│   ├── config/
│   │   ├── common.py        # DType, Cost
│   │   ├── dc.py            # DatacenterConfig, GPUSpec, KernelRun, ...
│   │   ├── workload.py      # MegatronWorkload, InferenceWorkload, LLMSpec
│   │   └── scenario.py      # ScenarioConfig (datacenter + workload)
│   ├── workload/
│   │   ├── trace.py         # WorkloadTrace, CommOp, ComputeOp (Pydantic)
│   │   └── megatron.py      # generate_megatron_trace()
│   ├── profiling/
│   │   └── kernels.py       # benchmark_kernels() — CUDA event timing
│   ├── backend/
│   │   ├── base.py          # Backend ABC
│   │   ├── analytical.py    # analytical backend (stub)
│   │   └── ns3.py           # NS-3 backend (stub)
│   └── cli/
│       └── __init__.py      # `simulon simulate` and `simulon profile gpu`
├── templates/
│   ├── gpu/                 # GPU hardware profiles (YAML)
│   ├── cpu/                 # CPU profiles
│   ├── nic/                 # NIC profiles
│   ├── switch/              # Switch profiles
│   └── model/               # LLM architecture profiles
├── docs/spec/               # Config format specifications
└── tests/
    ├── python/              # Config serialization tests
    └── workload/            # Trace generation tests
```

---

## Installation

Requires Python 3.11+. Uses [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

---

## Quick example

### 1. Generate a workload trace (no GPU needed)

```python
from simulon.config.workload import (
    LLMSpec, MegatronParallelism, MegatronTraining, MegatronWorkload
)
from simulon.workload import generate_megatron_trace

model = LLMSpec(
    name="LLaMA-7B",
    hidden_size=4096,
    num_layers=32,
    num_heads=32,
    vocab_size=32000,
    ffn_hidden_size=11008,
    swiglu=True,
)

workload = MegatronWorkload(
    framework="megatron",
    model=model,
    parallelism=MegatronParallelism(tp=4, pp=1, sp=True, distributed_optimizer=True),
    training=MegatronTraining(
        num_gpus=32,
        global_batch_size=256,
        micro_batch_size=2,
        sequence_length=4096,
    ),
)

trace = generate_megatron_trace(workload, model)
print(f"{len(trace.ops)} ops — {sum(1 for op in trace.ops if op.op == 'comm')} comm, "
      f"{sum(1 for op in trace.ops if op.op == 'compute')} compute")
```

### 2. Add GPU timing from a profile

```python
import yaml
from simulon.config.dc import GPUSpec

profile = GPUSpec.model_validate(yaml.safe_load(open("templates/gpu/mock-h100.yaml")))
trace = generate_megatron_trace(workload, model, gpu_profile=profile)

# ComputeOps now carry measured timing
for op in trace.ops:
    if op.op == "compute" and op.compute_time_us:
        print(f"  {op.kernel:15s}  {op.compute_time_us:.1f} µs")
        break
```

### 3. Profile a real GPU

```bash
simulon profile gpu \
  --name H100-SXM5-80GB \
  --vendor nvidia --memory-capacity-gb 80 --tdp-w 700 \
  --hidden-size 4096 --num-heads 32 --ffn-hidden-size 16384 \
  --seq-len 2048 --batch-size 1 --vocab-size 32000 \
  --dtype bf16 --tp 1 --epoch-num 20 \
  --output templates/gpu/h100.yaml
```

Run repeatedly with different `--tp`, `--seq-len`, or `--hidden-size` values to build
a richer profile. The command appends new `kernel_runs` entries to the existing file.

---

## Workload config format

A workload YAML uses exactly one framework. Two are currently supported:

**Megatron-LM training (`framework: megatron`)**

```yaml
framework: megatron

model:
  name: LLaMA-13B
  hidden_size: 5120
  num_layers: 40
  num_heads: 40
  vocab_size: 32000
  ffn_hidden_size: 13824
  swiglu: true

parallelism:
  tp: 4
  pp: 4
  sp: true
  distributed_optimizer: true

training:
  num_gpus: 64
  global_batch_size: 1024
  micro_batch_size: 2
  sequence_length: 4096
  dtype: bf16
  iterations: 500
```

See [`docs/spec/config-workload.md`](docs/spec/config-workload.md) for the full
specification including inference workloads, MoE models, and model profile references.

---

## Running tests

```bash
uv run pytest
```
