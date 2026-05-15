import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import ClassVar

import pybind11
from setuptools import Command, Extension, setup
from setuptools.command.build_py import build_py

ext = Extension(
    "simulon._mocknccl",
    sources=[
        "csrc/bindings.cpp",
        "csrc/mocknccl/MockNcclGroup.cc",
        "csrc/mocknccl/MockNcclChannel.cc",
        "csrc/mocknccl/MockNcclLog.cc",
    ],
    include_dirs=[pybind11.get_include(), "csrc"],
    extra_compile_args=["-std=c++17", "-O2"],
    language="c++",
)

_BUILD_TARGETS: list[tuple[str, str]] = [
    ("vendor/atlahs/sim/LogGOPSim", "LogGOPSim"),
    ("vendor/atlahs/sim/LogGOPSim", "txt2bin"),
    ("vendor/atlahs/sim/htsim-backend/sim/datacenter", "htsim_uec"),
    ("vendor/atlahs/goal_gen/hpc/Schedgen", "schedgen"),
]


def _get_platform_tag() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "darwin":
        if machine in ("arm64", "aarch64"):
            return "darwin_arm64"
        if machine in ("x86_64", "amd64"):
            return "darwin_x86_64"
    elif system == "linux":
        if machine in ("x86_64", "amd64"):
            return "linux_x86_64"
    raise RuntimeError(
        f"Unsupported platform: {system=}, {machine=}. " +
        "Supported: darwin_arm64, darwin_x86_64, linux_x86_64."
    )


class BuildAtlahsCommand(Command):
    description: ClassVar[str] = "Build ATLAHS simulator binaries from vendor/atlahs source"
    user_options = []  # type: ignore[override]

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def _find_in_path(self, name: str) -> str | None:
        return shutil.which(name)

    def _run_make(self, src_dir: Path, binary_name: str) -> Path:
        if not src_dir.exists():
            raise RuntimeError(f"Source directory does not exist: {src_dir}")

        makefile = src_dir / "Makefile"
        if not makefile.exists():
            raise RuntimeError(f"No Makefile found in {src_dir}")

        self.announce(f"Building {binary_name} in {src_dir} ...")
        result = subprocess.run(
            ["make", "-C", str(src_dir)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            self.announce(result.stdout, level=2)
            self.announce(result.stderr, level=2)
            raise RuntimeError(f"make failed in {src_dir} (rc={result.returncode})")

        binary_path = src_dir / binary_name
        if not binary_path.exists():
            for candidate in src_dir.rglob(binary_name):
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    binary_path = candidate
                    break
            else:
                raise RuntimeError(
                    f"Binary '{binary_name}' not found after make in {src_dir}"
                )

        return binary_path.resolve()

    def _patch_txt2bin_macos(self) -> None:
        if platform.system() != "Darwin":
            return
        repo_root = Path(__file__).resolve().parent
        txt2bin_path = repo_root / "vendor" / "atlahs" / "sim" / "LogGOPSim" / "txt2bin.re"
        if not txt2bin_path.exists():
            return
        content = txt2bin_path.read_text()
        if "main(int argc, char **argv){" in content and "int main(int argc, char **argv){" not in content:
            self.announce("Patching txt2bin.re for macOS Clang compatibility...")
            content = content.replace("main(int argc, char **argv){", "int main(int argc, char **argv){")
            txt2bin_path.write_text(content)
            self.announce("  -> txt2bin.re patched")

    def run(self) -> None:
        self.announce("Checking for build dependencies")
        if self._find_in_path("re2c") is None or self._find_in_path("gengetopt") is None:
            raise RuntimeError(
                "ATLAHS build requires re2c and gengetopt. Install: " +
                "macOS: brew install re2c gengetopt; " +
                "Ubuntu/Debian: sudo apt install re2c gengetopt"
            )

        self._patch_txt2bin_macos()

        self.announce("Detecting platform")
        repo_root = Path(__file__).resolve().parent
        platform_tag = _get_platform_tag()
        output_dir = (
            repo_root / "src" / "simulon" / "backend" / "atlahs_binaries" / platform_tag
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        unique_dirs: dict[Path, list[str]] = {}
        for rel_dir, binary_name in _BUILD_TARGETS:
            src = repo_root / rel_dir
            unique_dirs.setdefault(src, []).append(binary_name)

        for src_dir, binaries in unique_dirs.items():
            for binary_name in binaries:
                built = self._run_make(src_dir, binary_name)
                dest = output_dir / binary_name
                _ = shutil.copy2(built, dest)
                dest.chmod(dest.stat().st_mode | 0o111)
                self.announce(f"  -> {dest}")

        self.announce(f"\nAll ATLAHS binaries staged in {output_dir}")


class BuildPyWithAtlahs(build_py):
    def run(self) -> None:
        self.run_command("build_atlahs")
        super().run()


_ = setup(
    name="simulon",
    version="0.1.0",
    ext_modules=[ext],
    cmdclass={
        "build_py": BuildPyWithAtlahs,
        "build_atlahs": BuildAtlahsCommand,
    },
    package_dir={"": "src"},
    zip_safe=False,
)
