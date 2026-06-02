"""Tests for node template loading and resolution: load_node_template,
resolve_node_spec, resolve_nccl_profile, and resolve_scale_out.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from simulon.config.dc import DatacenterConfig, DatacenterMeta, NodeSpec, ScaleOutSpec, SwitchSpec
from simulon.config.nccl_profile import NcclProfile
from simulon.config.resolve import (
    load_node_template,
    resolve_nccl_profile,
    resolve_node_spec,
    resolve_scale_out,
)

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


def _make_dc(node: NodeSpec) -> DatacenterConfig:
    return DatacenterConfig(datacenter=DatacenterMeta(name="test"), num_nodes=1, node=node)


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

    def test_raises_when_templates_dir_missing(self, tmp_path: Path, monkeypatch) -> None:
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
        dc = _make_dc(NodeSpec.model_validate({"from": "test-node", "gpus_per_node": 8}))
        spec = resolve_node_spec(dc)
        assert spec.gpus_per_node == 8  # override applied
        assert spec.gpu == "h100"  # from base template

    def test_from_with_nested_field_override(self, node_templates_dir: Path) -> None:
        """resolve_node_spec deep-merges nested scale_up.switch overrides without clobbering sibling fields."""
        (node_templates_dir / "test-node.yaml").write_text(_MINIMAL_NODE_YAML)
        # Override only latency; port_speed from base template must survive.
        override_data = {"from": "test-node", "scale_up": {"switch": {"latency": "0.0001ms"}}}
        dc = _make_dc(NodeSpec.model_validate(override_data))
        spec = resolve_node_spec(dc)
        assert spec.scale_up is not None
        switch = spec.scale_up.switch
        assert isinstance(switch, SwitchSpec)
        assert switch.latency == "0.0001ms"
        assert switch.port_speed == "2554Gbps"  # sibling field preserved from base

    def test_deep_merge_does_not_clobber_siblings(self, node_templates_dir: Path) -> None:
        """Partial nested override of scale_up.switch does not wipe port_speed from the base."""
        (node_templates_dir / "test-node.yaml").write_text(_MINIMAL_NODE_YAML)
        override_data = {"from": "test-node", "scale_up": {"switch": {"port_speed": "7200Gbps"}}}
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

    def test_returns_none_when_no_profile(self, node_templates_dir: Path, tmp_path: Path) -> None:
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
    def test_returns_node_scale_out(self) -> None:
        so = ScaleOutSpec()
        node = NodeSpec(gpus_per_node=4, scale_out=so)
        dc = _make_dc(node)
        result = resolve_scale_out(dc)
        assert result is so

    def test_returns_none_when_node_has_no_scale_out(self) -> None:
        dc = _make_dc(NodeSpec(gpus_per_node=4))
        assert resolve_scale_out(dc) is None
