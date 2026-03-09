"""Custom hatchling build hook to compile C++ extensions with CMake."""

import os
import subprocess
import sys
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Build hook that compiles C++ extensions using CMake."""

    def initialize(self, version, build_data):
        """Run CMake to build C++ extension before packaging."""
        if self.target_name not in ["wheel", "sdist"]:
            return

        print("Running CMake to build C++ extension...")

        # Get project root
        root = Path(self.root)
        build_dir = root / "build"

        # Find pybind11 cmake directory
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pybind11", "--cmakedir"],
                capture_output=True,
                text=True,
                check=True,
            )
            pybind11_dir = result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to find pybind11: {e}")

        # Configure CMake with explicit Python version
        subprocess.run(
            [
                "cmake",
                "-B", str(build_dir),
                "-S", str(root),
                f"-Dpybind11_DIR={pybind11_dir}",
                f"-DPYTHON_EXECUTABLE={sys.executable}",
                "-DCMAKE_BUILD_TYPE=Release",
            ],
            check=True,
        )

        # Build
        subprocess.run(
            ["cmake", "--build", str(build_dir), "--config", "Release"],
            check=True,
        )

        # Copy .so to package directory
        so_files = list(build_dir.glob("_sim*.so"))
        if not so_files:
            raise RuntimeError("No .so file found after build")

        target_dir = root / "src" / "simulon"
        for so_file in so_files:
            import shutil
            shutil.copy2(so_file, target_dir / so_file.name)
            print(f"Copied {so_file.name} to {target_dir}")

        # Ensure .so files are included in the wheel
        if "force_include" not in build_data:
            build_data["force_include"] = {}

        for so_file in so_files:
            rel_path = f"src/simulon/{so_file.name}"
            build_data["force_include"][rel_path] = f"simulon/{so_file.name}"
