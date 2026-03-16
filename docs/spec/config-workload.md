# Workload Configuration Specification

This document specifies the format of the workload configuration file (`workload.yaml`).
The workload config describes the model, parallelism strategy, and workload parameters
to be simulated. Two frameworks are currently supported: **Megatron-LM** (training)
and **Inference**.

---

## Table of Contents

1. [Overview](#overview)
2. [Framework Selection](#framework-selection)
3. [Model Profiles and Inline Specs](#model-profiles-and-inline-specs)
4. [Megatron-LM Workload](#megatron-lm-workload)
   - [`model` Block](#model-block)
   - [`parallelism` Block](#parallelism-block)
   - [`training` Block](#training-block)
5. [Inference Workload](#inference-workload)
   - [`parallelism` Block (Inference)](#parallelism-block-inference)
   - [`inference` Block](#inference-block)
6. [Model Profile Reference](#model-profile-reference)
7. [Full Examples](#full-examples)

---

## Overview

Each workload configuration file declares exactly one `framework`. The remaining fields
are framework-specific. Supported frameworks: `megatron` (LLM training), `inference`
(LLM inference).

The workload config is independent of the datacenter config. The `num_gpus` field in
the workload must not exceed the total GPUs available in the datacenter config, but
the two files are validated together at simulation time rather than being coupled.

---

## Framework Selection

```yaml
framework: megatron      # megatron | inference  (more planned: deepspeed, vllm, ...)
```

> **Mutual exclusivity:** A workload file uses exactly **one** framework. The `megatron`
> framework requires `model`, `parallelism`, and `training` blocks. The `inference`
> framework requires `model`, `parallelism`, and `inference` blocks. Mixing blocks from
> different frameworks (e.g. a `framework: megatron` file that also contains an
> `inference:` key) is a validation error — both `MegatronWorkload` and
> `InferenceWorkload` use `extra="forbid"`.

---

## Model Profiles and Inline Specs

Model architecture is specified under the `model` key using the same
profile reference mechanism as hardware components (see `config-dc.md`).

The simulator searches for model profiles in:
1. The path specified by `profiles_dir` in the datacenter config (if set).
2. The built-in bundled model profile directory.

**Short-form reference** (string):

```yaml
model: llama-13b
```

**Long-form with overrides:**

```yaml
model:
  from: llama-13b       # inherit all fields from the named profile
  hidden_size: 6144     # override specific fields
  num_layers: 48
```

**Fully inline:**

```yaml
model:
  name: MyModel
  hidden_size: 4096
  num_layers: 32
  # ... all required fields
```

---

## Megatron-LM Workload

A Megatron workload config has three top-level blocks beneath `framework`:

```yaml
framework: megatron

model:       { ... }    # model architecture (profile or inline)
parallelism: { ... }    # parallelism strategy
training:    { ... }    # training hyperparameters and implementation options
```

---

### `model` Block

Defines the transformer architecture. All fields listed here may be present in a model
profile file, or declared inline, or used to override a profile via `from:`.

#### Dense Transformer Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | no | Human-readable model name |
| `hidden_size` | int | yes | Transformer hidden dimension $d_\text{model}$ |
| `num_layers` | int | yes | Number of transformer layers |
| `num_heads` | int | yes | Number of attention heads |
| `ffn_hidden_size` | int | no | FFN intermediate dimension. Defaults to `4 * hidden_size`, or `8/3 * hidden_size` rounded to a multiple of 64 when `swiglu: true` |
| `vocab_size` | int | yes | Vocabulary size. Used for embedding layer sizing. |
| `swiglu` | bool | no | Use SwiGLU activation in the FFN. Default: `false` |

#### MoE Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `moe` | bool | no | Enable Mixture-of-Experts. Default: `false` |
| `num_experts` | int | if `moe: true` | Total number of experts |
| `top_k` | int | if `moe: true` | Number of experts routed per token |

#### Notes

- `num_heads` must divide evenly into `hidden_size`.
- When `moe: true`, the FFN in each transformer layer is replaced by the expert routing + expert FFN structure.
- All fields can be overridden when using `from:`.

```yaml
model:
  from: llama-13b

# --- OR inline dense model ---
model:
  name: MyDenseModel
  hidden_size: 5120
  num_layers: 40
  num_heads: 40
  vocab_size: 32000
  ffn_hidden_size: 13824    # optional; derived from hidden_size + swiglu if omitted
  swiglu: true

# --- OR inline MoE model ---
model:
  name: MyMoEModel
  hidden_size: 7168
  num_layers: 61
  num_heads: 64
  vocab_size: 129280
  swiglu: true
  moe: true
  num_experts: 256
  top_k: 8
```

---

### `parallelism` Block

Defines the Megatron-style parallelism strategy.

| Field | Type | Default | Description |
|---|---|---|---|
| `tp` | int | `1` | Tensor Parallelism degree |
| `pp` | int | `1` | Pipeline Parallelism degree (number of pipeline stages) |
| `ep` | int | `1` | Expert Parallelism degree (MoE only; must divide `num_experts`) |
| `dp` | int | derived | Data Parallelism degree. If omitted, derived as `num_gpus / (tp × pp × ep)`. If provided, validated against this formula. |
| `sp` | bool | `false` | Enable Sequence Parallelism. Distributes LayerNorm and Dropout across the TP group. Requires `tp > 1`. |
| `vpp` | int | `1` | Virtual pipeline stages per pipeline rank (interleaved 1F1B schedule). `1` means standard non-interleaved schedule. Must satisfy `num_layers % (pp × vpp) == 0`. |
| `distributed_optimizer` | bool | `false` | Shard optimizer states across the DP group (Megatron distributed optimizer). Reduces per-GPU optimizer memory by `1/dp`. |
| `num_microbatches` | int | derived | Number of pipeline micro-batches per iteration. If omitted, derived as `global_batch_size / (micro_batch_size × dp)`. Override when the derived value is inconvenient (e.g. for DAG extraction with a fixed schedule). |

```yaml
parallelism:
  tp: 4
  pp: 4
  ep: 1
  dp: 4             # optional; derived if omitted, validated if provided
  sp: true
  vpp: 2
  distributed_optimizer: true
  num_microbatches: 8   # optional; derived if omitted
```

---

### `training` Block

Training hyperparameters and implementation options.

| Field | Type | Default | Description |
|---|---|---|---|
| `num_gpus` | int | required | Total GPUs to use for this workload. Must be ≤ cluster GPU count. Must equal `tp × pp × ep × dp`. |
| `global_batch_size` | int | required | Total batch size across all DP replicas |
| `micro_batch_size` | int | required | Batch size per forward/backward pass per pipeline stage. Number of pipeline micro-steps = `global_batch_size / (micro_batch_size × dp)`. |
| `sequence_length` | int | required | Input sequence length in tokens |
| `dtype` | string | `bf16` | Training precision: `fp32`, `fp16`, `bf16`, `fp8` |
| `flash_attention` | bool | `false` | Use FlashAttention for the attention kernel. Affects kernel execution time lookup. |
| `iterations` | int | `1` | Number of training iterations to simulate |

```yaml
training:
  num_gpus: 64
  global_batch_size: 1024
  micro_batch_size: 2
  sequence_length: 4096
  dtype: bf16
  flash_attention: true
  iterations: 100
```

---

## Inference Workload

An inference workload config has three top-level blocks beneath `framework`:

```yaml
framework: inference

model:       { ... }    # model architecture (profile or inline; same as Megatron)
parallelism: { ... }    # parallelism strategy
inference:   { ... }    # inference parameters
```

The `model` block is identical to the Megatron model block (see [`model` Block](#model-block)).

---

### `parallelism` Block (Inference)

Inference parallelism omits training-specific fields (`sp`, `vpp`, `distributed_optimizer`).
`dp` represents independent replica count (each replica handles separate request streams).

| Field | Type | Default | Description |
|---|---|---|---|
| `tp` | int | `1` | Tensor Parallelism degree |
| `pp` | int | `1` | Pipeline Parallelism degree (number of pipeline stages) |
| `ep` | int | `1` | Expert Parallelism degree (MoE only; must divide `num_experts`) |
| `dp` | int | derived | Number of independent inference replicas. If omitted, derived as `num_gpus / (tp × pp × ep)`. If provided, validated against this formula. |

```yaml
parallelism:
  tp: 8
  pp: 1
  ep: 8
  dp: 1             # optional; derived if omitted, validated if provided
```

---

### `inference` Block

Inference parameters and implementation options.

| Field | Type | Default | Description |
|---|---|---|---|
| `num_gpus` | int | required | Total GPUs for this workload. Must be ≤ cluster GPU count. Must equal `tp × pp × ep × dp`. |
| `phase` | string | `decode` | Inference phase to simulate: `prefill` or `decode` |
| `batch_size` | int | required | Number of concurrent requests (micro-batch size) |
| `seq_length` | int | required | Sequence length in tokens. For `prefill`: input prompt length. For `decode`: total context length (prefill + generated tokens). |
| `dtype` | string | `bf16` | Inference precision: `fp32`, `fp16`, `bf16`, `fp8` |
| `flash_attention` | bool | `false` | Use FlashAttention for the attention kernel. Affects kernel execution time lookup. |
| `routing_strategy` | string | `RoundRobin` | MoE expert routing strategy: `RoundRobin`, `Random`. Ignored for dense models. |

```yaml
inference:
  num_gpus: 64
  phase: decode
  batch_size: 32
  seq_length: 2048
  dtype: bf16
  flash_attention: true
  routing_strategy: RoundRobin
```

---

## Model Profile Reference

Model profiles are YAML files stored in the profile library. Each profile declares
the architecture fields from the `model` block above. Any field may be overridden
when referencing the profile via `from:`.

**Example profile — `llama-13b.yaml`:**

```yaml
name: LLaMA-13B
hidden_size: 5120
num_layers: 40
num_heads: 40
vocab_size: 32000
ffn_hidden_size: 13824
swiglu: true
moe: false
```

**Example profile — `deepseek-v3.yaml`:**

```yaml
name: DeepSeek-V3
hidden_size: 7168
num_layers: 61
num_heads: 128
vocab_size: 129280
ffn_hidden_size: 18432
swiglu: true
moe: true
num_experts: 256
top_k: 8
```

Bundled profiles will be provided for common models (LLaMA 7B/13B/70B, GPT variants,
DeepSeek-V2/V3, Qwen, Mixtral, etc.).

---

## Full Examples

### Megatron-LM Training Example

```yaml
framework: megatron

model:
  from: llama-13b
  ffn_hidden_size: 13824    # override for a custom variant

parallelism:
  tp: 4
  pp: 4
  ep: 1
  sp: true
  vpp: 2
  distributed_optimizer: true
  # dp is derived: 64 / (4 * 4 * 1) = 4

training:
  num_gpus: 64
  global_batch_size: 1024
  micro_batch_size: 2
  sequence_length: 4096
  dtype: bf16
  flash_attention: true
  iterations: 500
```

### Inference Example

```yaml
framework: inference

model:
  from: deepseek-v3

parallelism:
  tp: 8
  pp: 1
  ep: 8
  # dp is derived: 64 / (8 * 1 * 8) = 1

inference:
  num_gpus: 64
  phase: decode
  batch_size: 32
  seq_length: 2048
  dtype: fp8
  flash_attention: true
  routing_strategy: RoundRobin
```
