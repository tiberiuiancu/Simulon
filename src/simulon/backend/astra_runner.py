"""ASTRA-Sim simulation runner that executes the compiled binary."""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional

from simulon.backend.astra_converter import TopologyConverter, WorkloadConverter
from simulon.backend.astra_converter.file_writer import (
    create_astra_sim_config,
    write_topology_file,
    write_workload_file,
)
from simulon.config.scenario import ScenarioConfig

try:
    from simulon._sim import NetworkTopology, WorkloadTrace
except ImportError:
    # Fallback - will fail at runtime if C++ bindings not available
    NetworkTopology = None  # type: ignore
    WorkloadTrace = None  # type: ignore


class AstraSimRunner:
    """Runs ASTRA-Sim simulations using the compiled binaries."""

    def __init__(
        self,
        network_backend: str = "analytical",
        astra_binary_path: Optional[Path] = None,
    ):
        """Initialize the ASTRA-Sim runner.

        Args:
            network_backend: "analytical" or "ns3"
            astra_binary_path: Path to ASTRA-Sim binary. If None, will try to
                find it in the project structure.
        """
        self.network_backend = network_backend

        if astra_binary_path is None:
            # Try to find the binary in the project structure
            project_root = Path(__file__).parent.parent.parent.parent
            if network_backend == "analytical":
                binary_name = "SimAI_analytical"
                binary_path = (
                    project_root
                    / "csrc"
                    / "astra-sim-build"
                    / "simai_analytical"
                    / "build"
                    / "simai_analytical"
                    / binary_name
                )
            else:
                raise NotImplementedError(f"NS-3 backend not yet implemented")

            if not binary_path.exists():
                raise FileNotFoundError(
                    f"ASTRA-Sim binary not found at {binary_path}. "
                    f"Please compile it first."
                )
            self.binary_path = binary_path
        else:
            self.binary_path = astra_binary_path

    def run(
        self,
        scenario: ScenarioConfig,
        output_dir: Optional[Path] = None,
    ) -> Dict:
        """Run ASTRA-Sim simulation for the given scenario.

        Args:
            scenario: Scenario configuration
            output_dir: Directory for outputs. If None, uses a temp directory.

        Returns:
            Dictionary with simulation results
        """
        # Convert scenario to ASTRA-Sim format
        topo_converter = TopologyConverter()
        workload_converter = WorkloadConverter()

        network_topo = topo_converter.convert(scenario.datacenter)
        workload_trace = workload_converter.convert(
            scenario.workload, scenario.datacenter
        )

        # Create temporary directory for simulation files
        if output_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="astra_sim_")
            output_dir = Path(temp_dir)
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Write workload file
        workload_file = output_dir / "workload.txt"
        write_workload_file(workload_trace, workload_file)

        # Write topology file if needed (analytical backend may not need it)
        if self.network_backend != "analytical":
            topology_file = output_dir / "topology.txt"
            write_topology_file(network_topo, topology_file)
        else:
            topology_file = None

        # Create results directory
        results_dir = output_dir / "results"
        results_dir.mkdir(exist_ok=True)

        # Build command line arguments for ASTRA-Sim
        # Note: We omit -nv and -nic for now as the CSV files may cause parsing issues
        # ASTRA-Sim will use default values
        cmd = [
            str(self.binary_path),
            "-w",
            str(workload_file),
            "-g",
            str(workload_trace.all_gpus),
            "-g_p_s",
            str(network_topo.gpus_per_server),
            "-r",
            str(results_dir),
            "-g_type",
            network_topo.gpu_type,
        ]

        # Run the simulation
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=output_dir,
            )
            stdout = result.stdout
            stderr = result.stderr
            returncode = 0
        except subprocess.CalledProcessError as e:
            stdout = e.stdout
            stderr = e.stderr
            returncode = e.returncode

        # Parse results
        results = {
            "status": "success" if returncode == 0 else "error",
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "output_dir": str(output_dir),
            "results_dir": str(results_dir),
            "workload_file": str(workload_file),
        }

        if topology_file:
            results["topology_file"] = str(topology_file)

        # Try to extract timing information from output
        for line in stdout.split("\n"):
            if "finished" in line.lower() or "time" in line.lower():
                results["timing_info"] = line

        return results
