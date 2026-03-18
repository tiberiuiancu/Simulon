"""Unit tests for simulon.profiling.models (no GPU required)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from simulon.profiling.models import load_model_template, model_to_kernel_params

# ---------------------------------------------------------------------------
# load_model_template
# ---------------------------------------------------------------------------

_DENSE_YAML = """\
name: llama-7b
hidden_size: 4096
num_heads: 32
ffn_hidden_size: 11008
vocab_size: 32000
"""

_MOE_YAML = """\
name: mixtral-8x7b
hidden_size: 4096
num_heads: 32
ffn_hidden_size: 14336
vocab_size: 32000
num_experts: 8
top_k: 2
swiglu: true
"""


def test_load_model_template_dense(tmp_path):
    (tmp_path / "llama-7b.yaml").write_text(_DENSE_YAML)
    with patch("simulon.profiling.models._TEMPLATES_DIR", tmp_path):
        tmpl = load_model_template("llama-7b")
    assert tmpl["hidden_size"] == 4096
    assert tmpl["vocab_size"] == 32000


def test_load_model_template_moe(tmp_path):
    (tmp_path / "mixtral-8x7b.yaml").write_text(_MOE_YAML)
    with patch("simulon.profiling.models._TEMPLATES_DIR", tmp_path):
        tmpl = load_model_template("mixtral-8x7b")
    assert tmpl["num_experts"] == 8
    assert tmpl["swiglu"] is True


def test_load_model_template_not_found(tmp_path):
    with patch("simulon.profiling.models._TEMPLATES_DIR", tmp_path):
        with pytest.raises(FileNotFoundError, match="wrong-model"):
            load_model_template("wrong-model")


# ---------------------------------------------------------------------------
# model_to_kernel_params
# ---------------------------------------------------------------------------


def test_model_to_kernel_params_dense():
    tmpl = {
        "name": "llama-7b",
        "hidden_size": 4096,
        "num_heads": 32,
        "ffn_hidden_size": 11008,
        "vocab_size": 32000,
    }
    params = model_to_kernel_params(tmpl)
    assert params == {
        "hidden_size": 4096,
        "num_heads": 32,
        "ffn_hidden_size": 11008,
        "vocab_size": 32000,
    }
    # Non-kernel fields are excluded
    assert "name" not in params


def test_model_to_kernel_params_moe():
    tmpl = {
        "hidden_size": 4096,
        "num_heads": 32,
        "ffn_hidden_size": 14336,
        "vocab_size": 32000,
        "num_experts": 8,
        "top_k": 2,
        "swiglu": True,
    }
    params = model_to_kernel_params(tmpl)
    assert params["num_experts"] == 8
    assert params["top_k"] == 2
    assert params["swiglu"] is True


def test_model_to_kernel_params_excludes_none():
    tmpl = {
        "hidden_size": 4096,
        "num_heads": 32,
        "ffn_hidden_size": 11008,
        "vocab_size": 32000,
        "num_experts": None,
        "top_k": None,
    }
    params = model_to_kernel_params(tmpl)
    assert "num_experts" not in params
    assert "top_k" not in params


def test_model_to_kernel_params_swiglu_false_included():
    tmpl = {
        "hidden_size": 4096,
        "num_heads": 32,
        "ffn_hidden_size": 11008,
        "vocab_size": 32000,
        "swiglu": False,
    }
    params = model_to_kernel_params(tmpl)
    assert "swiglu" in params
    assert params["swiglu"] is False


def test_model_to_kernel_params_minimal():
    """Only required fields — no optional fields present."""
    tmpl = {
        "hidden_size": 8192,
        "num_heads": 64,
        "ffn_hidden_size": 28672,
        "vocab_size": 128000,
    }
    params = model_to_kernel_params(tmpl)
    assert set(params.keys()) == {"hidden_size", "num_heads", "ffn_hidden_size", "vocab_size"}
