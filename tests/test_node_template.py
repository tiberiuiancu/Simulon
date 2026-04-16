"""Tests for node template loading and resolution: load_node_template,
resolve_node_spec, resolve_nccl_profile, resolve_scale_out, and the
`simulon profile node` CLI command.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from simulon.cli import app
from simulon.config.dc import (
    ClusterSpec,
    DatacenterConfig,
    DatacenterMeta,
    NodeSpec,
    ScaleOutSpec,
    ScaleUpSpec,
    SwitchSpec,
)
from simulon.config.nccl_profile import NcclProfile
from simulon.config.resolve import (
    load_node_template,
    resolve_nccl_profile,
    resolve_node_spec,
    resolve_scale_out,
)

runner = CliRunner()


# ---------------------------------------------------------------------------
# Minimal YAML content for a node template
# ---------------------------------------------------------------------------

_MINIMAL_NODE_YAML = """\
name: test-node
gpus_per_node: 4
gpu: h100
scale_up:
  switch:
    port_speed: 2554Gbps
    latency: 0.000025ms
"""

_NODE_WITH_NCCL_YAML = """\
name: test-node-nccl
gpus_per_node: 4
gpu: h100
scale_up:
  switch:
    port_speed: 2554Gbps
    latency: 0.000025ms
nccl:
  name: test-node-nccl
  gpus_per_node: 4
  AllReduce:
    ring:
      - {size_bytes: 8388608, bus_bw_GBps: 200.0}
      - {size_bytes: 16777216, bus_bw_GBps: 220.0}
"""

_MINIMAL_NCCL_JSON = {
    "config": {"devices": [0, 1, 2, 3]},
    "results": [
        {"size": 8388608, "out_of_place": {"bus_bw": 200.0}},
        {"size": 16777216, "out_of_place": {"bus_bw": 220.0}},
    ],
}


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def node_templates_dir(tmp_path: Path, monkeypatch) -> Path:
    """Create a temporary templates/node/ directory and chdir into tmp_path."""
    tdir = tmp_path / "templates" / "node"
    tdir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    return tdir


def _make_dc(
    node: NodeSpec, scale_out: ScaleOutSpec | None = None, network=None
) -> DatacenterConfig:
    return DatacenterConfig(
        datacenter=DatacenterMeta(name="test"),
        cluster=ClusterSpec(num_nodes=1),
        node=node,
        scale_out=scale_out,
        network=network,
    )


# ---------------------------------------------------------------------------
# load_node_template
# ---------------------------------------------------------------------------


class TestLoadNodeTemplate:
    def test_loads_valid_template(self, node_templates_dir: Path) -> None:
        """load_node_template returns a NodeSpec when the template file exists."""
        (node_templates_dir / "test-node.yaml").write_text(_MINIMAL_NODE_YAML)
        spec = load_node_template("test-node")
        assert spec.name == "test-node"
        assert spec.gpus_per_node == 4
        assert spec.gpu == "h100"

    def test_raises_for_missing_template(self, node_templates_dir: Path) -> None:
        """load_node_template raises FileNotFoundError for unknown names."""
        with pytest.raises(FileNotFoundError, match="Node template not found"):
            load_node_template("does-not-exist")

    def test_case_insensitive_fallback(self, node_templates_dir: Path) -> None:
        """load_node_template finds a template when name case differs from filename."""
        (node_templates_dir / "Test-Node.yaml").write_text(_MINIMAL_NODE_YAML)
        spec = load_node_template("test-node")
        assert spec.gpus_per_node == 4

    def test_raises_when_templates_dir_missing(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """load_node_template raises FileNotFoundError when templates/node/ dir is absent."""
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            load_node_template("any-name")


# ---------------------------------------------------------------------------
# resolve_node_spec
# ---------------------------------------------------------------------------


class TestResolveNodeSpec:
    def test_bare_string_coercion(self, node_templates_dir: Path) -> None:
        """resolve_node_spec loads the template when node is a bare string reference."""
        (node_templates_dir / "test-node.yaml").write_text(_MINIMAL_NODE_YAML)
        dc = _make_dc(NodeSpec.model_validate({"from": "test-node"}))
        spec = resolve_node_spec(dc)
        assert spec.name == "test-node"
        assert spec.gpus_per_node == 4

    def test_from_with_shallow_override(self, node_templates_dir: Path) -> None:
        """resolve_node_spec applies top-level field overrides from the inline spec."""
        (node_templates_dir / "test-node.yaml").write_text(_MINIMAL_NODE_YAML)
        dc = _make_dc(
            NodeSpec.model_validate({"from": "test-node", "gpus_per_node": 8})
        )
        spec = resolve_node_spec(dc)
        assert spec.gpus_per_node == 8  # override applied
        assert spec.gpu == "h100"  # from base template

    def test_from_with_nested_field_override(self, node_templates_dir: Path) -> None:
        """resolve_node_spec deep-merges nested scale_up.switch overrides without clobbering sibling fields."""
        (node_templates_dir / "test-node.yaml").write_text(_MINIMAL_NODE_YAML)
        # Override only latency; port_speed from base template must survive.
        override_data = {
            "from": "test-node",
            "scale_up": {"switch": {"latency": "0.0001ms"}},
        }
        dc = _make_dc(NodeSpec.model_validate(override_data))
        spec = resolve_node_spec(dc)
        assert spec.scale_up is not None
        switch = spec.scale_up.switch
        assert isinstance(switch, SwitchSpec)
        assert switch.latency == "0.0001ms"
        assert switch.port_speed == "2554Gbps"  # sibling field preserved from base

    def test_deep_merge_does_not_clobber_siblings(
        self, node_templates_dir: Path
    ) -> None:
        """Partial nested override of scale_up.switch does not wipe port_speed from the base."""
        (node_templates_dir / "test-node.yaml").write_text(_MINIMAL_NODE_YAML)
        override_data = {
            "from": "test-node",
            "scale_up": {"switch": {"port_speed": "7200Gbps"}},
        }
        dc = _make_dc(NodeSpec.model_validate(override_data))
        spec = resolve_node_spec(dc)
        switch = spec.scale_up.switch
        assert isinstance(switch, SwitchSpec)
        assert switch.port_speed == "7200Gbps"
        assert switch.latency == "0.000025ms"  # untouched base field

    def test_inline_spec_returned_as_is(self) -> None:
        """resolve_node_spec returns the inline NodeSpec directly when no from_ is set."""
        node = NodeSpec(name="custom", gpus_per_node=2)
        dc = _make_dc(node)
        spec = resolve_node_spec(dc)
        assert spec.name == "custom"
        assert spec.gpus_per_node == 2


# ---------------------------------------------------------------------------
# resolve_nccl_profile
# ---------------------------------------------------------------------------


class TestResolveNcclProfile:
    def test_embedded_nccl_takes_priority(self, node_templates_dir: Path) -> None:
        """resolve_nccl_profile returns embedded node.nccl over any companion file."""
        (node_templates_dir / "test-node.yaml").write_text(_NODE_WITH_NCCL_YAML)
        dc = _make_dc(NodeSpec.model_validate({"from": "test-node"}))
        profile = resolve_nccl_profile(dc)
        assert profile is not None
        assert profile.name == "test-node-nccl"
        assert len(profile.AllReduce.ring) == 2

    def test_fallback_to_companion_nccl_yaml(
        self, node_templates_dir: Path, tmp_path: Path
    ) -> None:
        """resolve_nccl_profile falls back to companion .nccl.yaml when node has no embedded profile."""
        # Provide a minimal node template without embedded nccl.
        (node_templates_dir / "bare-node.yaml").write_text(_MINIMAL_NODE_YAML)

        # Create a companion gpu template and a .nccl.yaml next to it.
        gpu_dir = tmp_path / "templates" / "gpu"
        gpu_dir.mkdir(parents=True)
        (gpu_dir / "h100.yaml").write_text("name: h100\n")
        nccl_data = {
            "name": "h100",
            "gpus_per_node": 4,
            "AllReduce": {"ring": [{"size_bytes": 8388608, "bus_bw_GBps": 199.0}]},
        }
        (gpu_dir / "h100.nccl.yaml").write_text(yaml.dump(nccl_data))

        dc = _make_dc(NodeSpec.model_validate({"from": "bare-node"}))
        profile = resolve_nccl_profile(dc)
        assert profile is not None
        assert profile.AllReduce.ring[0].bus_bw_GBps == pytest.approx(199.0)

    def test_returns_none_when_no_profile(
        self, node_templates_dir: Path, tmp_path: Path
    ) -> None:
        """resolve_nccl_profile returns None when neither embedded nor companion profile exists."""
        # Node template with a gpu string but no nccl profile.
        (node_templates_dir / "bare-node.yaml").write_text(_MINIMAL_NODE_YAML)
        gpu_dir = tmp_path / "templates" / "gpu"
        gpu_dir.mkdir(parents=True)
        (gpu_dir / "h100.yaml").write_text("name: h100\n")
        # No .nccl.yaml written.
        dc = _make_dc(NodeSpec.model_validate({"from": "bare-node"}))
        profile = resolve_nccl_profile(dc)
        assert profile is None

    def test_inline_node_with_embedded_nccl(self) -> None:
        """resolve_nccl_profile returns the inline nccl when the node is fully inline."""
        nccl = NcclProfile(gpus_per_node=4, name="inline-nccl")
        node = NodeSpec(gpus_per_node=4, gpu="h100", nccl=nccl)
        dc = _make_dc(node)
        profile = resolve_nccl_profile(dc)
        assert profile is not None
        assert profile.name == "inline-nccl"


# ---------------------------------------------------------------------------
# resolve_scale_out
# ---------------------------------------------------------------------------


class TestResolveScaleOut:
    def test_returns_top_level_scale_out(self) -> None:
        """resolve_scale_out returns dc.scale_out when it is set."""
        so = ScaleOutSpec()
        dc = _make_dc(NodeSpec(gpus_per_node=4), scale_out=so)
        result = resolve_scale_out(dc)
        assert result is so

    def test_fallback_to_network_scale_out_with_warning(self) -> None:
        """resolve_scale_out falls back to dc.network.scale_out with a DeprecationWarning."""
        from simulon.config.dc import NetworkSpec

        so = ScaleOutSpec()
        network = NetworkSpec(scale_out=so)
        dc = _make_dc(NodeSpec(gpus_per_node=4), network=network)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = resolve_scale_out(dc)
        assert result is so
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "deprecated" in str(w[0].message).lower()

    def test_returns_none_when_neither_set(self) -> None:
        """resolve_scale_out returns None when neither dc.scale_out nor dc.network.scale_out is set."""
        dc = _make_dc(NodeSpec(gpus_per_node=4))
        assert resolve_scale_out(dc) is None

    def test_top_level_takes_priority_over_network(self) -> None:
        """resolve_scale_out prefers top-level dc.scale_out over dc.network.scale_out."""
        from simulon.config.dc import NetworkSpec

        so_top = ScaleOutSpec()
        so_net = ScaleOutSpec()
        network = NetworkSpec(scale_out=so_net)
        dc = _make_dc(NodeSpec(gpus_per_node=4), scale_out=so_top, network=network)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = resolve_scale_out(dc)
        assert result is so_top
        assert len(w) == 0  # no deprecation warning when top-level is used


# ---------------------------------------------------------------------------
# CLI: simulon profile node
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data))


class TestProfileNodeCli:
    def test_input_json_generates_template(self, tmp_path: Path) -> None:
        """profile node --input-json writes a valid NodeSpec YAML with correct data."""
        json_dir = tmp_path / "nccl_results"
        json_dir.mkdir()
        _write_json(json_dir / "allreduce.json", _MINIMAL_NCCL_JSON)

        result = runner.invoke(
            app,
            [
                "profile",
                "node",
                "--gpu",
                "h100",
                "--input-json",
                str(json_dir),
                "--out",
                str(tmp_path / "templates" / "node" / "h100-4g.yaml"),
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        out_path = tmp_path / "templates" / "node" / "h100-4g.yaml"
        assert out_path.exists()

        data = yaml.safe_load(out_path.read_text())
        spec = NodeSpec.model_validate(data)
        assert spec.gpus_per_node == 4
        assert spec.nccl is not None
        assert len(spec.nccl.AllReduce.ring) == 2
        assert spec.nccl.AllReduce.ring[0].size_bytes == 8388608
        assert spec.nccl.AllReduce.ring[0].bus_bw_GBps == pytest.approx(200.0)

    def test_dry_run_prints_yaml_without_writing(self, tmp_path: Path) -> None:
        """profile node --dry-run prints YAML to stdout and does not write any file."""
        json_dir = tmp_path / "nccl_results"
        json_dir.mkdir()
        _write_json(json_dir / "allreduce.json", _MINIMAL_NCCL_JSON)

        out_path = tmp_path / "templates" / "node" / "h100-4g.yaml"
        result = runner.invoke(
            app,
            [
                "profile",
                "node",
                "--gpu",
                "h100",
                "--input-json",
                str(json_dir),
                "--out",
                str(out_path),
                "--dry-run",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert not out_path.exists()
        # YAML content should appear in stdout
        assert "gpus_per_node" in result.output

    def test_missing_input_exits_with_error(self, tmp_path: Path) -> None:
        """profile node exits with code 1 when neither --input-json nor --nccl-tests-dir is given."""
        result = runner.invoke(
            app,
            ["profile", "node", "--gpu", "h100"],
        )
        assert result.exit_code != 0
        assert "Error" in result.output or (
            result.stderr_bytes and b"Error" in result.stderr_bytes
        )

    def test_latency_always_written(self, tmp_path: Path) -> None:
        """profile node always writes the latency field regardless of whether --port-speed was passed."""
        json_dir = tmp_path / "nccl_results"
        json_dir.mkdir()
        _write_json(json_dir / "allreduce.json", _MINIMAL_NCCL_JSON)

        result = runner.invoke(
            app,
            [
                "profile",
                "node",
                "--gpu",
                "h100",
                "--input-json",
                str(json_dir),
                "--dry-run",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "latency" in result.output

    def test_port_speed_included_only_when_provided(self, tmp_path: Path) -> None:
        """profile node omits port_speed from scale_up.switch when --port-speed is not passed."""
        json_dir = tmp_path / "nccl_results"
        json_dir.mkdir()
        _write_json(json_dir / "allreduce.json", _MINIMAL_NCCL_JSON)

        result_without = runner.invoke(
            app,
            [
                "profile",
                "node",
                "--gpu",
                "h100",
                "--input-json",
                str(json_dir),
                "--dry-run",
            ],
            catch_exceptions=False,
        )
        result_with = runner.invoke(
            app,
            [
                "profile",
                "node",
                "--gpu",
                "h100",
                "--input-json",
                str(json_dir),
                "--port-speed",
                "7200Gbps",
                "--dry-run",
            ],
            catch_exceptions=False,
        )
        assert result_without.exit_code == 0
        assert result_with.exit_code == 0
        assert "port_speed" not in result_without.output
        assert "7200Gbps" in result_with.output

    def test_multiple_json_files_warns_and_uses_first(self, tmp_path: Path) -> None:
        """profile node emits a warning and uses the first file when multiple JSONs match a collective."""
        json_dir = tmp_path / "nccl_results"
        json_dir.mkdir()
        data_a = {
            **_MINIMAL_NCCL_JSON,
            "results": [{"size": 8388608, "out_of_place": {"bus_bw": 111.0}}],
        }
        data_b = {
            **_MINIMAL_NCCL_JSON,
            "results": [{"size": 8388608, "out_of_place": {"bus_bw": 222.0}}],
        }
        # Both filenames match *allreduce*.json
        _write_json(json_dir / "allreduce_run1.json", data_a)
        _write_json(json_dir / "allreduce_run2.json", data_b)

        result = runner.invoke(
            app,
            [
                "profile",
                "node",
                "--gpu",
                "h100",
                "--input-json",
                str(json_dir),
                "--dry-run",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        # Warning must appear in stderr or stdout
        combined = result.output + (
            result.stderr_bytes.decode() if result.stderr_bytes else ""
        )
        assert "Warning" in combined or "warning" in combined.lower()

    def test_custom_name_and_latency(self, tmp_path: Path) -> None:
        """profile node respects --name and --latency overrides in the generated template."""
        json_dir = tmp_path / "nccl_results"
        json_dir.mkdir()
        _write_json(json_dir / "allreduce.json", _MINIMAL_NCCL_JSON)

        result = runner.invoke(
            app,
            [
                "profile",
                "node",
                "--gpu",
                "h100",
                "--input-json",
                str(json_dir),
                "--name",
                "my-custom-node",
                "--latency",
                "0.0001ms",
                "--dry-run",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "my-custom-node" in result.output
        assert "0.0001ms" in result.output

    def test_all_four_collectives_in_output(self, tmp_path: Path) -> None:
        """profile node includes all measured collectives in the nccl profile."""
        json_dir = tmp_path / "nccl_results"
        json_dir.mkdir()
        for coll in ("allreduce", "allgather", "reducescatter", "alltoall"):
            _write_json(json_dir / f"{coll}.json", _MINIMAL_NCCL_JSON)

        out_path = tmp_path / "templates" / "node" / "h100-4g.yaml"
        result = runner.invoke(
            app,
            [
                "profile",
                "node",
                "--gpu",
                "h100",
                "--input-json",
                str(json_dir),
                "--out",
                str(out_path),
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        data = yaml.safe_load(out_path.read_text())
        spec = NodeSpec.model_validate(data)
        assert spec.nccl is not None
        assert len(spec.nccl.AllReduce.ring) == 2
        assert len(spec.nccl.AllGather.ring) == 2
        assert len(spec.nccl.ReduceScatter.ring) == 2
        assert len(spec.nccl.AllToAll.ring) == 2
