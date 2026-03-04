# Scenario Configuration Specification

This document specifies the format of the scenario configuration file (`scenario.yaml`).
A scenario is the top-level entry point for the simulator — it binds a datacenter config
and a workload config together into a single simulatable unit.

---

## Table of Contents

1. [Overview](#overview)
2. [Component References](#component-references)
3. [Full Examples](#full-examples)

---

## Overview

A scenario config has two required top-level keys:

| Field | Type | Description |
|---|---|---|
| `datacenter` | path or inline | The datacenter configuration. See `config-dc.md` for the full field reference. |
| `workload` | path or inline | The workload configuration. See `config-workload.md` for the full field reference. |

Each key accepts either a **file path** (string) pointing to a standalone config file,
or an **inline mapping** that embeds the config directly.

---

## Component References

**File path reference** (string):

```yaml
datacenter: ./dc.yaml
workload: ./workload.yaml
```

Paths are resolved relative to the scenario file's directory. Absolute paths are also accepted.

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
  framework: inference
  # ...
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

workload:
  framework: megatron

  model:
    from: llama-13b

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
