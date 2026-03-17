# Scenario Configuration Specification

This document specifies the format of the scenario configuration file (`scenario.yaml`).
A scenario is the top-level entry point for the simulator — it binds a datacenter config,
a workload config, and a collective communication config together into a single simulatable unit.

---

## Table of Contents

1. [Overview](#overview)
2. [Component References](#component-references)
3. [Collective Config](#collective-config)
4. [Full Examples](#full-examples)

---

## Overview

A scenario config has the following top-level keys:

| Field | Type | Required | Description |
|---|---|---|---|
| `datacenter` | path or inline | yes | The datacenter configuration. See `config-dc.md` for the full field reference. |
| `workload` | path or inline | yes | The workload configuration. See `config-workload.md` for the full field reference. |
| `collective` | inline | no | Collective communication library and algorithm settings. Defaults to `library: nccl, algorithm: ring, num_channels: 1`. |

Each of `datacenter` and `workload` accepts either a **file path** (string) pointing to a
standalone config file, or an **inline mapping** that embeds the config directly.

---

## Component References

**File path reference** (string):

```yaml
datacenter: ./dc.yaml
workload: ./workload.yaml
```

**Inline spec** (mapping):

```yaml
datacenter:
  # full datacenter config fields here (see config-dc.md)

workload:
  # full workload config fields here (see config-workload.md)
```

**Mixed** (one from file, one inline):

```yaml
datacenter: ./dc.yaml

workload:
  framework: megatron
  # ...
```

---

## Collective Config

The `collective` block selects the CCL library and algorithm used to decompose
collectives into P2P flows. It is optional — if omitted, NCCL with ring allreduce
and 1 channel is used.

### `library: nccl`

| Field | Type | Default | Description |
|---|---|---|---|
| `library` | string | `nccl` | CCL library identifier |
| `algorithm` | string | `ring` | Collective algorithm: `ring` \| `tree` \| `collnet_direct` \| `collnet_chain` \| `nvls` \| `nvls_tree` |
| `num_channels` | int | `1` | Number of parallel ring channels |

> **Note:** Only `ring` is fully implemented. `tree`, `collnet_direct`, `collnet_chain`,
> `nvls`, and `nvls_tree` raise `NotImplementedError`.

```yaml
collective:
  library: nccl
  algorithm: ring
  num_channels: 2
```

### `library: rccl`

AMD ROCm CCL library. Has the same fields as `nccl`. **Not yet implemented** — raises
`NotImplementedError` when used.

```yaml
collective:
  library: rccl
  algorithm: ring
  num_channels: 1
```

---

## Full Examples

### Both components from files

```yaml
datacenter: ./dc.yaml
workload: ./workload.yaml
```

### Datacenter from file, workload inline (Megatron)

```yaml
datacenter: ./h100-cluster.yaml

collective:
  library: nccl
  algorithm: ring
  num_channels: 1

workload:
  framework: megatron

  model:
    from: llama-13b

  parallelism:
    tp: 4
    pp: 4
    sp: true
    distributed_optimizer: true
    pipeline_schedule: 1f1b

  training:
    num_gpus: 64
    global_batch_size: 1024
    micro_batch_size: 2
    sequence_length: 4096
    dtype: bf16
    flash_attention: true
    iterations: 500
```

### Fully inline (Inference)

```yaml
datacenter:
  datacenter:
    name: Test Cluster
    pue: 1.3
    electricity_cost_per_kwh: 0.07

  cluster:
    num_nodes: 8

  node:
    gpu:
      from: H100-SXM5-80GB
      gpus_per_node: 8
    cpu:
      from: AMD-EPYC-9354

  scale_up:
    topology: switched
    technology: nvlink

  scale_out:
    nic:
      from: CX7-400G
    topology:
      type: fat_tree
      switch:
        from: QM9700

workload:
  framework: inference

  model:
    from: deepseek-v3

  parallelism:
    tp: 8
    pp: 1
    ep: 8

  inference:
    num_gpus: 64
    phase: decode
    batch_size: 32
    seq_length: 2048
    dtype: fp8
    flash_attention: true
    routing_strategy: RoundRobin
```
