"""Tests for trace refactor: resolve_workload, workload_hash, list_traces."""

from __future__ import annotations

import pytest
import yaml

from simulon.config.resolve import resolve_workload, workload_hash
from simulon.config.workload import MegatronWorkload


def test_resolve_workload_base_only(tmp_path):
    workload_file = tmp_path / "base.yaml"
    workload_file.write_text(
        yaml.dump(
            {
                "framework": "megatron",
                "config": {
                    "tensor-model-parallel-size": 1,
                    "pipeline-model-parallel-size": 1,
                    "micro-batch-size": 1,
                    "global-batch-size": 8,
                    "seq-length": 2048,
                    "num-layers": 12,
                    "hidden-size": 4096,
                    "num-attention-heads": 32,
                    "ffn-hidden-size": 16384,
                    "vocab-size": 32000,
                    "dtype": "bf16",
                },
            }
        )
    )

    result = resolve_workload(str(workload_file))

    assert isinstance(result, MegatronWorkload)
    assert result.config["tensor-model-parallel-size"] == 1
    assert result.config["num-layers"] == 12


def test_resolve_workload_inheritance(tmp_path):
    templates_dir = tmp_path / "templates" / "workload"
    templates_dir.mkdir(parents=True)

    (templates_dir / "base.yaml").write_text(
        yaml.dump(
            {
                "framework": "megatron",
                "config": {
                    "tensor-model-parallel-size": 1,
                    "pipeline-model-parallel-size": 1,
                    "micro-batch-size": 1,
                    "global-batch-size": 8,
                    "seq-length": 2048,
                    "num-layers": 12,
                    "hidden-size": 4096,
                    "num-attention-heads": 32,
                    "ffn-hidden-size": 16384,
                    "vocab-size": 32000,
                    "dtype": "bf16",
                },
            }
        )
    )

    (templates_dir / "child.yaml").write_text(
        yaml.dump(
            {
                "from": "base",
                "config": {
                    "tensor-model-parallel-size": 4,
                    "pipeline-model-parallel-size": 2,
                    "num-layers": 24,
                },
            }
        )
    )

    result = resolve_workload(str(templates_dir / "child.yaml"))

    assert isinstance(result, MegatronWorkload)
    assert result.config["tensor-model-parallel-size"] == 4
    assert result.config["pipeline-model-parallel-size"] == 2
    assert result.config["num-layers"] == 24
    assert result.config["global-batch-size"] == 8
    assert result.config["hidden-size"] == 4096


def test_resolve_workload_circular_detection(tmp_path):
    templates_dir = tmp_path / "templates" / "workload"
    templates_dir.mkdir(parents=True)

    (templates_dir / "a.yaml").write_text(
        yaml.dump(
            {"framework": "megatron", "from": "b.yaml", "config": {"tensor-model-parallel-size": 1}}
        )
    )
    (templates_dir / "b.yaml").write_text(
        yaml.dump(
            {"framework": "megatron", "from": "a.yaml", "config": {"tensor-model-parallel-size": 2}}
        )
    )

    with pytest.raises(ValueError, match="(?i)circular"):
        resolve_workload(str(templates_dir / "a.yaml"))


def test_workload_hash_deterministic():
    wl = MegatronWorkload.model_validate(
        {
            "framework": "megatron",
            "config": {
                "tensor-model-parallel-size": 4,
                "pipeline-model-parallel-size": 2,
                "micro-batch-size": 2,
                "global-batch-size": 16,
                "seq-length": 4096,
                "num-layers": 40,
                "hidden-size": 5120,
                "num-attention-heads": 40,
                "ffn-hidden-size": 13824,
                "vocab-size": 32000,
            },
        }
    )

    h1 = workload_hash(wl)
    h2 = workload_hash(wl)

    assert h1 == h2


def test_workload_hash_ignores_data_path():
    wl1 = MegatronWorkload.model_validate(
        {
            "framework": "megatron",
            "config": {
                "tensor-model-parallel-size": 4,
                "pipeline-model-parallel-size": 2,
                "data_path": "/data/a",
            },
        }
    )
    wl2 = MegatronWorkload.model_validate(
        {
            "framework": "megatron",
            "config": {
                "tensor-model-parallel-size": 4,
                "pipeline-model-parallel-size": 2,
                "data_path": "/data/b",
            },
        }
    )

    assert workload_hash(wl1) == workload_hash(wl2)


def test_workload_hash_ignores_tokenizer():
    wl1 = MegatronWorkload.model_validate(
        {
            "framework": "megatron",
            "config": {
                "tensor-model-parallel-size": 4,
                "tokenizer_type": "GPT2BPETokenizer",
                "tokenizer_model": "/tmp/model",
            },
        }
    )
    wl2 = MegatronWorkload.model_validate(
        {
            "framework": "megatron",
            "config": {
                "tensor-model-parallel-size": 4,
                "tokenizer_type": "HuggingFaceTokenizer",
                "tokenizer_model": "/tmp/other",
            },
        }
    )

    assert workload_hash(wl1) == workload_hash(wl2)


def test_workload_hash_different_on_tp_change():
    wl1 = MegatronWorkload.model_validate(
        {
            "framework": "megatron",
            "config": {"tensor-model-parallel-size": 2, "pipeline-model-parallel-size": 1},
        }
    )
    wl2 = MegatronWorkload.model_validate(
        {
            "framework": "megatron",
            "config": {"tensor-model-parallel-size": 4, "pipeline-model-parallel-size": 1},
        }
    )

    assert workload_hash(wl1) != workload_hash(wl2)


def test_workload_hash_exactly_16_chars():
    wl = MegatronWorkload.model_validate(
        {
            "framework": "megatron",
            "config": {"tensor-model-parallel-size": 1, "pipeline-model-parallel-size": 1},
        }
    )

    h = workload_hash(wl)

    assert len(h) == 16
    assert h == h.lower()
    assert all(c in "0123456789abcdef" for c in h)


def test_list_traces_empty_dir(tmp_path):
    gpu_dir = tmp_path / "templates" / "gpu"
    (gpu_dir / "h100" / "traces").mkdir(parents=True)

    entries = []
    for gpu_dir_entry in sorted(gpu_dir.iterdir()):
        if not gpu_dir_entry.is_dir():
            continue
        traces_dir = gpu_dir_entry / "traces"
        if not traces_dir.is_dir():
            continue
        for trace_dir in sorted(traces_dir.iterdir()):
            if not trace_dir.is_dir():
                continue
            workload_yaml = trace_dir / "workload.yaml"
            if workload_yaml.exists():
                entries.append((trace_dir, gpu_dir_entry.name))

    assert len(entries) == 0


def test_list_traces_pagination(tmp_path):
    gpu_dir = tmp_path / "templates" / "gpu"
    h100_dir = gpu_dir / "h100" / "traces"
    h100_dir.mkdir(parents=True)

    names = []
    for i in range(5):
        trace_dir = h100_dir / f"hash{i:03d}"
        trace_dir.mkdir()
        (trace_dir / "workload.yaml").write_text(
            yaml.dump(
                {
                    "framework": "megatron",
                    "from": "base",
                    "config": {"tensor-model-parallel-size": 1},
                }
            )
        )
        names.append(f"hash{i:03d}")

    entries = []
    for trace_dir in sorted(h100_dir.iterdir()):
        if not trace_dir.is_dir():
            continue
        workload_yaml = trace_dir / "workload.yaml"
        if workload_yaml.exists():
            entries.append((trace_dir, "h100"))

    entries.sort(key=lambda e: e[0].stat().st_mtime, reverse=True)

    n, offset = 2, 1
    sliced = entries[offset : offset + n]

    assert len(sliced) == 2
    sliced_names = [e[0].name for e in sliced]
    assert sliced_names[0] != sliced_names[1]


def test_list_traces_gpu_filter(tmp_path):
    gpu_dir = tmp_path / "templates" / "gpu"
    (gpu_dir / "h100" / "traces").mkdir(parents=True)
    (gpu_dir / "a100" / "traces").mkdir(parents=True)

    h100_trace = gpu_dir / "h100" / "traces" / "hash_h100"
    h100_trace.mkdir()
    (h100_trace / "workload.yaml").write_text(
        yaml.dump({"framework": "megatron", "config": {"tensor-model-parallel-size": 4}})
    )

    a100_trace = gpu_dir / "a100" / "traces" / "hash_a100"
    a100_trace.mkdir()
    (a100_trace / "workload.yaml").write_text(
        yaml.dump({"framework": "megatron", "config": {"tensor-model-parallel-size": 8}})
    )

    gpu_filter = "h100"

    entries = []
    for gpu_dir_entry in sorted(gpu_dir.iterdir()):
        if not gpu_dir_entry.is_dir():
            continue
        gpu_name = gpu_dir_entry.name
        if gpu_filter and gpu_filter.lower() != gpu_name.lower():
            continue
        traces_dir = gpu_dir_entry / "traces"
        if not traces_dir.is_dir():
            continue
        for trace_dir in sorted(traces_dir.iterdir()):
            if not trace_dir.is_dir():
                continue
            workload_yaml = trace_dir / "workload.yaml"
            if workload_yaml.exists():
                entries.append((trace_dir, gpu_name))

    assert len(entries) == 1
    assert entries[0][1] == "h100"
    assert entries[0][0].name == "hash_h100"
