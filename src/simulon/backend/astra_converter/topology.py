"""Converter for datacenter configurations to ASTRA-Sim network topology format."""

from dataclasses import dataclass

from simulon.config.dc import DatacenterConfig


@dataclass
class NetworkNode:
    """A node in the network topology."""

    node_id: int
    node_type: str  # "gpu", "nvswitch", "leaf", "spine", "aggregation"


@dataclass
class NetworkLink:
    """A link between two nodes in the network."""

    source: int
    dest: int
    bandwidth_gbps: float
    latency_ns: float
    error_rate: float = 0.0


@dataclass
class NetworkTopology:
    """Complete network topology specification for ASTRA-Sim."""

    nodes: list[NetworkNode]
    links: list[NetworkLink]
    gpus_per_server: int
    nv_switch_num: int
    switches_excluding_nvswitch: int
    gpu_type: str
    nvlink_bandwidth_efficiency: float = 1.0  # Scale-up (intra-node) efficiency
    nic_bandwidth_efficiency: float = 0.85  # Scale-out (inter-node) efficiency


def _parse_bandwidth(bw_str: str) -> float:
    """Parse bandwidth string (e.g., '900Gbps') to Gbps float."""
    bw_str = bw_str.strip().upper()
    if bw_str.endswith("GBPS"):
        return float(bw_str[:-4])
    elif bw_str.endswith("TBPS"):
        return float(bw_str[:-4]) * 1000
    elif bw_str.endswith("MBPS"):
        return float(bw_str[:-4]) / 1000
    else:
        try:
            return float(bw_str)
        except ValueError:
            raise ValueError(f"Cannot parse bandwidth: {bw_str}")


def _parse_latency(lat_str: str) -> float:
    """Parse latency string (e.g., '0.000025ms', '5us') to nanoseconds."""
    lat_str = lat_str.strip().lower()
    if lat_str.endswith("ns"):
        return float(lat_str[:-2])
    elif lat_str.endswith("us"):
        return float(lat_str[:-2]) * 1000
    elif lat_str.endswith("ms"):
        return float(lat_str[:-2]) * 1_000_000
    elif lat_str.endswith("s"):
        return float(lat_str[:-1]) * 1_000_000_000
    else:
        try:
            return float(lat_str)
        except ValueError:
            raise ValueError(f"Cannot parse latency: {lat_str}")


class TopologyConverter:
    """Converts DatacenterConfig to NetworkTopology for ASTRA-Sim."""

    def convert(self, datacenter: DatacenterConfig) -> NetworkTopology:
        """Convert datacenter configuration to ASTRA-Sim network topology.

        Args:
            datacenter: The datacenter configuration to convert

        Returns:
            NetworkTopology suitable for passing to ASTRA-Sim

        Raises:
            ValueError: If topology type is not supported
        """
        nodes: list[NetworkNode] = []
        links: list[NetworkLink] = []

        num_nodes = datacenter.cluster.num_nodes
        gpus_per_node = datacenter.node.gpus_per_node
        total_gpus = num_nodes * gpus_per_node

        # Track next available node ID
        next_node_id = 0

        # 1. Create GPU nodes (IDs: [0, total_gpus))
        for i in range(total_gpus):
            nodes.append(NetworkNode(node_id=i, node_type="gpu"))
        next_node_id = total_gpus

        # 2. Create intra-node topology (scale-up network)
        nvswitch_ids_per_node: list[list[int]] = []
        num_nvswitches_per_node = 0

        if datacenter.scale_up and datacenter.scale_up.topology.value == "switched":
            # Create NVSwitch nodes for each server
            num_nvswitches_per_node = datacenter.node.num_switches_per_node or 0

            # Parse bandwidth and latency
            link_bandwidth_gbps = _parse_bandwidth(datacenter.scale_up.link_bandwidth)
            link_latency_ns = _parse_latency(datacenter.scale_up.link_latency)

            for node_idx in range(num_nodes):
                node_nvswitch_ids = []
                for switch_idx in range(num_nvswitches_per_node):
                    nvswitch_id = next_node_id
                    nodes.append(NetworkNode(node_id=nvswitch_id, node_type="nvswitch"))
                    node_nvswitch_ids.append(nvswitch_id)
                    next_node_id += 1
                nvswitch_ids_per_node.append(node_nvswitch_ids)

                # Connect GPUs to NVSwitches (full bipartite graph)
                gpu_start = node_idx * gpus_per_node
                gpu_end = gpu_start + gpus_per_node

                for gpu_id in range(gpu_start, gpu_end):
                    for nvswitch_id in node_nvswitch_ids:
                        # Bidirectional links
                        links.append(
                            NetworkLink(
                                source=gpu_id,
                                dest=nvswitch_id,
                                bandwidth_gbps=link_bandwidth_gbps,
                                latency_ns=link_latency_ns,
                            )
                        )
                        links.append(
                            NetworkLink(
                                source=nvswitch_id,
                                dest=gpu_id,
                                bandwidth_gbps=link_bandwidth_gbps,
                                latency_ns=link_latency_ns,
                            )
                        )

        elif datacenter.scale_up and datacenter.scale_up.topology.value == "p2p":
            # Direct GPU-to-GPU connections (all-to-all within node)
            link_bandwidth_gbps = _parse_bandwidth(datacenter.scale_up.link_bandwidth)
            link_latency_ns = _parse_latency(datacenter.scale_up.link_latency)

            for node_idx in range(num_nodes):
                gpu_start = node_idx * gpus_per_node
                gpu_end = gpu_start + gpus_per_node

                for src_gpu in range(gpu_start, gpu_end):
                    for dst_gpu in range(gpu_start, gpu_end):
                        if src_gpu != dst_gpu:
                            links.append(
                                NetworkLink(
                                    source=src_gpu,
                                    dest=dst_gpu,
                                    bandwidth_gbps=link_bandwidth_gbps,
                                    latency_ns=link_latency_ns,
                                )
                            )

        # 3. Create inter-node topology (scale-out network)
        total_nvswitches = num_nodes * num_nvswitches_per_node

        if datacenter.scale_out and datacenter.scale_out.topology:
            topology_type = datacenter.scale_out.topology.type.value
            if topology_type == "fat_tree":
                network_switches = self._create_fat_tree_topology(
                    datacenter, num_nodes, gpus_per_node, next_node_id, nodes, links
                )
            else:
                raise ValueError(
                    f"Unsupported scale-out topology: {topology_type}. "
                    "Currently only 'fat_tree' is implemented."
                )
        else:
            network_switches = 0

        # Determine GPU type
        gpu_spec = datacenter.node.gpu
        if isinstance(gpu_spec, str):
            gpu_type = gpu_spec.upper()
        else:
            gpu_type = (gpu_spec.name or "UNKNOWN").upper()

        # Extract bandwidth efficiency from config
        nvlink_efficiency = datacenter.scale_up.bandwidth_efficiency if datacenter.scale_up else 1.0
        nic_efficiency = datacenter.scale_out.nic.bandwidth_efficiency if datacenter.scale_out and datacenter.scale_out.nic else 0.85

        return NetworkTopology(
            nodes=nodes,
            links=links,
            gpus_per_server=gpus_per_node,
            nv_switch_num=total_nvswitches,
            switches_excluding_nvswitch=network_switches,
            gpu_type=gpu_type,
            nvlink_bandwidth_efficiency=nvlink_efficiency,
            nic_bandwidth_efficiency=nic_efficiency,
        )

    def _create_fat_tree_topology(
        self,
        datacenter: DatacenterConfig,
        num_nodes: int,
        gpus_per_node: int,
        next_node_id: int,
        nodes: list[NetworkNode],
        links: list[NetworkLink],
    ) -> int:
        """Create a fat-tree topology for the scale-out network.

        Args:
            datacenter: Datacenter configuration
            num_nodes: Number of compute nodes
            gpus_per_node: GPUs per compute node
            next_node_id: Next available node ID for switches
            nodes: List to append switch nodes to
            links: List to append links to

        Returns:
            Number of network switches created (excluding NVSwitches)
        """
        params = datacenter.scale_out.topology.params or {}
        k = params.get("k", 4)
        oversubscription = params.get("oversubscription", 1.0)

        # Fat-tree parameters
        # Each pod has k/2 leaf switches and k/2 aggregation switches
        # Total of k pods
        # (k/2)^2 core switches
        num_pods = k
        leaf_per_pod = k // 2
        agg_per_pod = k // 2
        num_core = (k // 2) ** 2

        total_leaf = num_pods * leaf_per_pod
        total_agg = num_pods * agg_per_pod

        # Ensure we have enough leaf switches for the number of nodes
        # Each leaf switch can connect to k/2 servers
        servers_per_leaf = k // 2
        required_leaf = (num_nodes + servers_per_leaf - 1) // servers_per_leaf

        if required_leaf > total_leaf:
            raise ValueError(
                f"Fat-tree with k={k} supports max {total_leaf * servers_per_leaf} nodes, "
                f"but datacenter has {num_nodes} nodes"
            )

        # Create leaf switches
        leaf_switch_ids = []
        for i in range(total_leaf):
            switch_id = next_node_id
            nodes.append(NetworkNode(node_id=switch_id, node_type="leaf"))
            leaf_switch_ids.append(switch_id)
            next_node_id += 1

        # Create aggregation switches
        agg_switch_ids = []
        for pod_idx in range(num_pods):
            pod_agg_ids = []
            for i in range(agg_per_pod):
                switch_id = next_node_id
                nodes.append(NetworkNode(node_id=switch_id, node_type="aggregation"))
                pod_agg_ids.append(switch_id)
                next_node_id += 1
            agg_switch_ids.append(pod_agg_ids)

        # Create core switches
        core_switch_ids = []
        for i in range(num_core):
            switch_id = next_node_id
            nodes.append(NetworkNode(node_id=switch_id, node_type="spine"))
            core_switch_ids.append(switch_id)
            next_node_id += 1

        # Get NIC parameters and parse them
        nic_spec = datacenter.scale_out.nic
        if isinstance(nic_spec, str):
            raise ValueError("NIC spec must be inline, not a reference")

        nic_speed_gbps = _parse_bandwidth(nic_spec.speed)
        nic_latency_ns = _parse_latency(nic_spec.latency)

        # Apply oversubscription to switch-to-switch links
        switch_bandwidth_gbps = nic_speed_gbps / oversubscription

        # Connect nodes to leaf switches (via their first GPU as representative)
        # In ASTRA-Sim, we typically connect one GPU per node to the network
        for node_idx in range(num_nodes):
            leaf_idx = node_idx // servers_per_leaf
            if leaf_idx >= len(leaf_switch_ids):
                break  # Don't exceed available leaf switches

            leaf_switch_id = leaf_switch_ids[leaf_idx]
            # Connect first GPU of this node to leaf switch
            gpu_id = node_idx * gpus_per_node

            # Bidirectional links
            links.append(
                NetworkLink(
                    source=gpu_id,
                    dest=leaf_switch_id,
                    bandwidth_gbps=nic_speed_gbps,
                    latency_ns=nic_latency_ns,
                )
            )
            links.append(
                NetworkLink(
                    source=leaf_switch_id,
                    dest=gpu_id,
                    bandwidth_gbps=nic_speed_gbps,
                    latency_ns=nic_latency_ns,
                )
            )

        # Connect leaf switches to aggregation switches within each pod
        for pod_idx in range(num_pods):
            pod_leaf_start = pod_idx * leaf_per_pod
            pod_leaf_end = pod_leaf_start + leaf_per_pod

            for leaf_local_idx in range(leaf_per_pod):
                leaf_id = leaf_switch_ids[pod_leaf_start + leaf_local_idx]

                # Each leaf connects to all aggregation switches in its pod
                for agg_id in agg_switch_ids[pod_idx]:
                    links.append(
                        NetworkLink(
                            source=leaf_id,
                            dest=agg_id,
                            bandwidth_gbps=switch_bandwidth_gbps,
                            latency_ns=nic_latency_ns,
                        )
                    )
                    links.append(
                        NetworkLink(
                            source=agg_id,
                            dest=leaf_id,
                            bandwidth_gbps=switch_bandwidth_gbps,
                            latency_ns=nic_latency_ns,
                        )
                    )

        # Connect aggregation switches to core switches
        for pod_idx in range(num_pods):
            for agg_local_idx, agg_id in enumerate(agg_switch_ids[pod_idx]):
                # Each aggregation switch connects to k/2 core switches
                # Distribute connections across core switches
                for core_local_idx in range(k // 2):
                    core_idx = agg_local_idx * (k // 2) + core_local_idx
                    if core_idx >= len(core_switch_ids):
                        break
                    core_id = core_switch_ids[core_idx]

                    links.append(
                        NetworkLink(
                            source=agg_id,
                            dest=core_id,
                            bandwidth_gbps=switch_bandwidth_gbps,
                            latency_ns=nic_latency_ns,
                        )
                    )
                    links.append(
                        NetworkLink(
                            source=core_id,
                            dest=agg_id,
                            bandwidth_gbps=switch_bandwidth_gbps,
                            latency_ns=nic_latency_ns,
                        )
                    )

        return total_leaf + total_agg + num_core
