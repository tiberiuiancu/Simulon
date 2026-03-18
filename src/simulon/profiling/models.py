"""Utilities for loading and converting model architecture templates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Root of the templates/model/ directory relative to the package install.
_TEMPLATES_DIR = Path(__file__).parent.parent.parent.parent / "templates" / "model"


def load_model_template(name: str) -> dict[str, Any]:
    """Load a model template YAML from templates/model/<name>.yaml.

    Raises FileNotFoundError if the template does not exist.
    """
    path = _TEMPLATES_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Model template '{name}' not found at {path}. "
            "Create templates/model/<name>.yaml or pass arch fields directly."
        )
    with open(path) as f:
        return yaml.safe_load(f)


def model_to_kernel_params(tmpl: dict[str, Any]) -> dict[str, Any]:
    """Extract kernel parameter fields from a model template dict."""
    keys = ["hidden_size", "num_heads", "ffn_hidden_size", "vocab_size",
            "num_experts", "top_k", "swiglu"]
    return {k: tmpl[k] for k in keys if k in tmpl and tmpl[k] is not None}
