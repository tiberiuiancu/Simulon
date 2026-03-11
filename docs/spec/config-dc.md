# Datacenter Configuration Specification

This document specifies the format of the datacenter configuration file (`datacenter.yaml`).
This file describes the hardware and network topology of the simulated datacenter cluster.

---

## Table of Contents

1. [Overview](#overview)
2. [Profile References](#profile-references)
3. [Cost Specification](#cost-specification)
4. [Top-Level Structure](#top-level-structure)
5. [`datacenter` Block](#datacenter-block)
6. [`cluster` Block](#cluster-block)
7. [`node` Block](#node-block)
   - [GPU](#gpu)
   - [CPU](#cpu)
   - [Node-Level Cooling](#node-level-cooling)
8. [`network` Block](#network-block)
   - [`network.scale_up` (Intra-Node)](#networkscale_up-intra-node)
   - [`network.scale_out` (Inter-Node)](#networkscale_out-inter-node)
   - [Topology Templates](#topology-templates)
9. [Full Example](#full-example)

---

## Overview

The datacenter configuration file is a YAML document that fully describes the simulated
hardware environment: compute nodes, intra-node (scale-up) networking, inter-node (scale-out)
networking, and datacenter-level infrastructure parameters (power, cooling, cost).

The configuration supports two ways to specify any hardware component:

- **Profile reference** – a string name or `from:` key pointing to a bundled or user-supplied
  hardware profile.
- **Inline specification** – all fields declared directly in the config file.

These can be combined: use `from:` to inherit a profile and override individual fields.

---

## Profile References

Hardware profiles (GPU, CPU, NIC, switch, etc.) can be stored as YAML files in a profile
library directory. The simulator searches for profiles in the following order:

1. The path specified by `datacenter.profiles_dir` (if set).
2. The built-in bundled profile directory (shipped with the simulator).

A profile is identified by its filename (without extension), e.g. `H100.yaml` → `H100`.

**Short-form reference** (string only):

```yaml
gpu: H100
```

**Long-form reference with overrides:**

```yaml
gpu:
  from: H100          # inherit all fields from the H100 profile
  name: B200          # override the display name
  flops_multiplier: 2.0
  tdp_w: 1000
```

**Fully inline** (no profile, all fields explicit):

```yaml
gpu:
  name: MyGPU
  vendor: nvidia
  # ... all required fields present
```

If a field is present both in the profile and in the inline override, the inline value wins.

---

## Cost Specification

Cost fields appear throughout the config wherever a hardware component is described.
They are always **optional**. When present, they are used for capex/opex modeling.
All monetary values are in **USD**.

```yaml
# Scalar shorthand
cost: 30000

# Object form (min/max are optional)
cost:
  value: 30000    # base/expected price
  min: 25000      # lower bound (e.g. volume discount)
  max: 35000      # upper bound (e.g. list price)
```

---

## Top-Level Structure

```yaml
datacenter:   { ... }   # datacenter-level infrastructure
cluster:      { ... }   # cluster scale and layout
node:         { ... }   # per-node hardware specification
network:      { ... }   # network fabric (scale-up + scale-out)
```

---

## `datacenter` Block

Datacenter-level infrastructure and operational parameters.

```yaml
datacenter:
  name: "My AI Cluster"         # optional, human-readable label
  location: "US-East-1"         # optional, geographic label

  # Path to hardware profile library.
  # Defaults to the built-in bundled profiles if omitted or empty.
  profiles_dir: "./templates"

  # Power Usage Effectiveness: multiplier applied to total IT power draw
  # to estimate total facility power (IT + cooling + distribution losses).
  # PUE = 1.0 is ideal (no overhead). Typical values: 1.1 – 1.5.
  pue: 1.2

  # Electricity cost in USD per kWh.
  electricity_cost_per_kwh: 0.07

  # Rack configuration
  rack:
    nodes_per_rack: 8             # number of compute nodes per rack
    rack_units: 42                # total rack height in U (optional, informational)
    max_power_kw: 100             # maximum power draw per rack in kW (optional)
    cost:                         # optional capex per rack (enclosure, cabling, PDU)
      value: 15000
      min: 10000
      max: 20000

    # Per-rack cooling unit (optional).
    # Use this when cooling is modeled at rack granularity rather than
    # only at the datacenter level via PUE.
    cooling:
      capacity_kw: 100            # cooling capacity in kW
      tdp_w: 500                  # power draw of the cooling unit itself
      cost:
        value: 8000
```

---

## `cluster` Block

Specifies the overall scale of the cluster.

```yaml
cluster:
  num_nodes: 64         # total number of compute nodes
```

> **Note:** The current implementation assumes a homogeneous cluster (all nodes identical).
> The spec permits per-node overrides for future extensibility, but simulator support is limited.

---

## `node` Block

Describes the hardware inside a single compute node.

### GPU

Every node must have at least one GPU.

**Node-level topology fields** (sibling to `gpu`, `cpu`, `cooling`):

| Field | Type | Description |
|---|---|---|
| `gpus_per_node` | int | Number of GPUs in this node |
| `gpus_per_nic` | int | Number of GPUs sharing each NIC. Default: `1`. Simulator support for values > 1 is limited. |

> **Note:** One NVSwitch per node is assumed for switched NVLink configurations.
> This is specified via `network.scale_up.switch`, not in the node block.

**GPU fields:**

| Field | Type | Description |
|---|---|---|
| `from` | string | Profile name to inherit from |
| `name` | string | Display name (overrides profile) |
| `vendor` | string | `nvidia` or `amd` |
| `flops_multiplier` | float | Uniform scalar applied to all profiled FLOP rates. Default: `1.0` |
| `memory_capacity_gb` | float | HBM capacity in GB |
| `tdp_w` | float | Thermal Design Power in watts |
| `cost` | cost | See [Cost Specification](#cost-specification) |
| `kernel_runs` | list | Measured kernel benchmarks. Populated by `simulon profile gpu`; empty when declared inline. See [Kernel Runs](#kernel-runs). |

Performance data (FLOP rates, memory bandwidth, kernel execution times) is loaded from the
referenced profile. The `flops_multiplier` scales all FLOP-based metrics uniformly;
it does not affect memory-bandwidth-bound kernels.

Additional profile fields (e.g. memory bandwidth, NVLink bandwidth, supported precisions)
can be extended in profile files without changes to this spec.

#### Kernel Runs

`kernel_runs` is a list of benchmark records produced by `simulon profile gpu`. Each entry:

| Field | Type | Description |
|---|---|---|
| `kernel` | string | Kernel name (e.g. `matmul`, `flash_attention`, `all_reduce`) |
| `params` | mapping | Kernel-specific parameters (e.g. `{M: 4096, N: 4096, K: 4096}`) |
| `times_ms` | list[float] | Measured wall-clock times in milliseconds, one per run |

```yaml
kernel_runs:
  - kernel: matmul
    params: {M: 4096, N: 4096, K: 4096, dtype: bf16}
    times_ms: [0.42, 0.41, 0.43, 0.42]
  - kernel: flash_attention
    params: {seq_len: 4096, num_heads: 32, head_dim: 128, dtype: bf16}
    times_ms: [1.12, 1.11, 1.13]
```

```yaml
node:
  gpus_per_node: 8                # number of GPUs in this node
  gpus_per_nic: 1                 # GPUs sharing each NIC

  gpu: H100                       # short-form profile reference

  # --- OR long-form ---
  gpu:
    from: H100
    name: H100-SXM-enhanced
    vendor: nvidia
    flops_multiplier: 1.2
    memory_capacity_gb: 80
    tdp_w: 750
    cost:
      value: 30000
      min: 25000
      max: 35000
```

### CPU

Optional. Used primarily for power and cost modeling.

**Fields:**

| Field | Type | Description |
|---|---|---|
| `from` | string | Profile name to inherit from |
| `name` | string | Display name or model string |
| `vendor` | string | e.g. `intel`, `amd` |
| `sockets` | int | Number of CPU sockets per node. Default: `2` |
| `cores_per_socket` | int | Physical cores per socket |
| `memory_gb` | float | Total system DRAM in GB |
| `tdp_w` | float | TDP per socket in watts |
| `cost` | cost | Cost per socket |
| `memory_cost_per_gb` | float | USD/GB; multiplied by `memory_gb` for total memory cost |

```yaml
  cpu:
    from: "Intel-Xeon-8480+"      # optional profile reference
    # --- OR inline ---
    name: "Intel Xeon Platinum 8480+"
    vendor: intel
    sockets: 2
    cores_per_socket: 60
    memory_gb: 512
    tdp_w: 350                    # per socket
    cost:
      value: 4000                 # per socket
    memory_cost_per_gb: 5         # optional; total memory cost = 5 * 512 = $2560
```

### Node-Level Cooling

Optional. Models a node-attached or chassis-level cooling unit (e.g. direct liquid cooling
per chassis). This is in addition to rack-level and datacenter-level (PUE) cooling.

```yaml
  cooling:
    tdp_w: 200                    # power draw of the cooling unit
    cost:
      value: 1500
```

---

## `network` Block

The `network` block contains two sub-blocks: `scale_up` for the intra-node fabric and
`scale_out` for the inter-node fabric. Both are optional.

```yaml
network:
  scale_up:   { ... }   # intra-node NVLink fabric
  scale_out:  { ... }   # inter-node network
```

---

### `network.scale_up` (Intra-Node)

Describes the intra-node GPU interconnect. One NVSwitch per node is assumed.
The `switch` field specifies the NVSwitch hardware using `SwitchSpec`.

**`SwitchSpec` fields** (used for both `scale_up.switch` and `scale_out` leaf/spine switches):

| Field | Type | Description |
|---|---|---|
| `from` | string | Profile name to inherit from |
| `name` | string | Display name |
| `vendor` | string | e.g. `nvidia` |
| `port_count` | int | Number of switch ports |
| `port_speed` | bandwidth | Per-port bandwidth, e.g. `2880Gbps`. For NVSwitch, this is the NVLink bandwidth. |
| `latency` | time | Switch propagation latency, e.g. `0.000025ms` |
| `buffer_per_port` | string | Per-port TX queue buffer (NS-3 backend only), e.g. `32MB` |
| `queue_discipline` | string | `drop_tail` \| `red` \| `codel` \| `fq_codel` (NS-3 backend only) |
| `tdp_w` | float | Power draw per switch chip |
| `cost` | cost | Cost per switch chip |

```yaml
network:
  scale_up:
    switch:                       # NVSwitch specification (1 per node)
      name: NVSwitch3
      port_speed: 2880Gbps        # NVLink bandwidth per port
      latency: 0.000025ms         # propagation latency
      tdp_w: 110
      cost:
        value: 3000
```

---

### `network.scale_out` (Inter-Node)

Describes the inter-node network. Contains the NIC spec, optional switch specs for leaf
and spine tiers, and the topology configuration.

**NIC fields:**

| Field | Type | Description |
|---|---|---|
| `from` | string | Profile name |
| `name` | string | Display name |
| `vendor` | string | e.g. `mellanox`, `broadcom`, `intel` |
| `speed` | bandwidth | Line rate per NIC port |
| `latency` | time | End-to-end NIC latency contribution |
| `tdp_w` | float | Power draw per NIC |
| `cost` | cost | Cost per NIC |
| `bandwidth_efficiency` | float | Effective bandwidth fraction (0.0–1.0). Default: `0.85`. |

The number of GPUs sharing each NIC is set via `node.gpus_per_nic` (default: `1`).

```yaml
network:
  scale_out:
    nic:
      from: ConnectX-7              # optional profile reference
      # --- OR inline ---
      name: "Mellanox ConnectX-7"
      vendor: mellanox
      speed: 400Gbps
      latency: 0.0005ms
      bandwidth_efficiency: 0.85
      tdp_w: 25
      cost:
        value: 2000

    leaf_switch:                    # optional; absent for topologies without leaf tier
      name: Spectrum-4
      port_count: 64
      port_speed: 400Gbps
      buffer_per_port: 32MB
      queue_discipline: fq_codel
      cost:
        value: 50000

    spine_switch:                   # optional; absent for topologies without spine tier
      name: Spectrum-4
      port_count: 128
      cost:
        value: 80000

    topology:
      type: fat_tree                # see Topology Templates below
      params:
        k: 64
        oversubscription: 1.0
```

---

### Topology Templates

The `type` field selects one of the built-in topology generators. With a template selected,
any field under `params` that is omitted is either derived automatically (from `num_nodes`,
`gpus_per_node`, NIC speed, etc.) or set to a documented default.

If `type: custom` is used, no generator is invoked and the user must supply a full topology
description (format TBD).

---

#### `spectrum_x`

NVIDIA Spectrum-X rail-optimized topology. Two-tier fabric (leaf + spine) with optional
dual-ToR and dual-plane configurations.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `nics_per_leaf` | int | derived | NICs connected to each leaf switch |
| `num_leaf_switches` | int | derived | Total number of leaf switches |
| `num_spine_switches` | int | derived | Total number of spine switches |
| `leaf_to_spine_bandwidth` | bandwidth | = NIC speed | Uplink bandwidth from leaf to spine |
| `switches_per_spine` | int | derived | Leaf switches per spine switch |
| `nvlink_switches_per_node` | int | derived | NVLink switch chips per node (scale-up) |
| `dual_tor` | bool | `false` | Enable dual Top-of-Rack for redundancy |
| `dual_plane` | bool | `false` | Enable dual network planes |

```yaml
      type: spectrum_x
      params:
        nics_per_leaf: 64
        num_leaf_switches: 8
        num_spine_switches: 16
        leaf_to_spine_bandwidth: 400Gbps
        dual_tor: false
        dual_plane: false
```

---

#### `alibaba_hpn`

Alibaba High-Performance Network topology. Three-tier design with optional dual-ToR and dual-plane.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `nics_per_leaf` | int | derived | NICs per leaf switch |
| `num_leaf_switches` | int | derived | Number of leaf switches |
| `num_spine_switches` | int | derived | Number of spine switches |
| `leaf_to_spine_bandwidth` | bandwidth | = NIC speed | Leaf–spine uplink bandwidth |
| `dual_tor` | bool | `false` | Enable dual ToR |
| `dual_plane` | bool | `false` | Enable dual plane |

```yaml
      type: alibaba_hpn
      params:
        nics_per_leaf: 64
        num_leaf_switches: 16
        num_spine_switches: 8
        leaf_to_spine_bandwidth: 400Gbps
        dual_tor: true
        dual_plane: false
```

---

#### `dcn_plus`

Enhanced datacenter network (DCN+) topology.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `nics_per_leaf` | int | derived | NICs per leaf switch |
| `num_leaf_switches` | int | derived | Number of leaf switches |
| `num_spine_switches` | int | derived | Number of spine switches |
| `uplink_bandwidth` | bandwidth | = NIC speed | Leaf–spine uplink bandwidth |
| `dual_tor` | bool | `false` | Enable dual ToR |

```yaml
      type: dcn_plus
      params:
        nics_per_leaf: 32
        num_leaf_switches: 16
        num_spine_switches: 4
        uplink_bandwidth: 400Gbps
        dual_tor: false
```

---

#### `fat_tree`

Classic k-ary fat-tree (Leiserson / Al-Fares). Supports 2-tier and 3-tier configurations.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `k` | int | derived | Switch radix (ports per switch). Determines tier sizes. |
| `num_tiers` | int | `3` | Number of network tiers: `2` (leaf+spine) or `3` (leaf+aggregate+core) |
| `oversubscription` | float | `1.0` | Downlink : uplink port ratio at leaf tier (1.0 = non-blocking) |

```yaml
      type: fat_tree
      params:
        k: 64
        num_tiers: 3
        oversubscription: 2.0
```

---

#### `rail_optimized`

Simple rail-optimized topology. Each GPU is assigned to a rail (a single leaf switch);
all rails connect to shared spine switches.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_rails` | int | = `gpus_per_node` | Number of rails (leaf switches) |
| `nodes_per_rail` | int | derived | Nodes attached to each rail |
| `num_spine_switches` | int | `1` | Number of spine switches |
| `rail_to_spine_links` | int | `1` | Uplinks per rail switch to spine tier |

```yaml
      type: rail_optimized
      params:
        num_rails: 8
        nodes_per_rail: 8
        num_spine_switches: 2
        rail_to_spine_links: 2
```

---

#### `dragonfly`

Dragonfly topology. GPUs are grouped into router groups; intra-group links are dense,
inter-group links are sparse.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `group_size` | int | required | Number of routers per group |
| `nodes_per_router` | int | derived | Compute nodes per router |
| `intra_group_links` | int | derived | Links between routers within a group |
| `inter_group_links` | int | `1` | Links between router groups |

```yaml
      type: dragonfly
      params:
        group_size: 8
        nodes_per_router: 2
        intra_group_links: 7
        inter_group_links: 2
```

---

#### `custom`

No generator is invoked. The full topology must be provided via an external description file
(format TBD).

```yaml
      type: custom
      topology_file: "./my_topology.txt"
```

---

## Full Example

```yaml
datacenter:
  name: "Research Cluster A"
  location: "EU-West"
  profiles_dir: "./templates"
  pue: 1.3
  electricity_cost_per_kwh: 0.08
  rack:
    nodes_per_rack: 8
    rack_units: 42
    max_power_kw: 120
    cost:
      value: 12000
    cooling:
      capacity_kw: 120
      tdp_w: 600
      cost:
        value: 9000

cluster:
  num_nodes: 64

node:
  gpus_per_node: 8
  gpus_per_nic: 1
  gpu:
    from: H100
    flops_multiplier: 1.0
    tdp_w: 700
    cost:
      value: 30000
      min: 25000
      max: 35000

  cpu:
    name: "Intel Xeon Platinum 8480+"
    vendor: intel
    sockets: 2
    cores_per_socket: 60
    memory_gb: 512
    tdp_w: 350
    cost:
      value: 4000
    memory_cost_per_gb: 5

  cooling:
    tdp_w: 200
    cost:
      value: 1000

network:
  scale_up:
    switch:
      name: NVSwitch3
      port_speed: 2880Gbps
      latency: 0.000025ms
      tdp_w: 110
      cost:
        value: 3000

  scale_out:
    nic:
      from: ConnectX-7
      speed: 400Gbps
      latency: 0.0005ms
      tdp_w: 25
      cost:
        value: 2000

    leaf_switch:
      from: Spectrum-4
      buffer_per_port: 32MB
      queue_discipline: fq_codel
      tdp_w: 300
      cost:
        value: 50000

    topology:
      type: fat_tree
      params:
        k: 64
        oversubscription: 1.0
```
