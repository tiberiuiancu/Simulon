"""The synthesized distributed-optimizer collectives, pinned to measurement.

Real-process-group traces are captured at DP=1, where the dist-optimizer's grad
reduce-scatter and param all-gather are no-ops, so `_inject_dp_grad_sync` synthesizes them
when such a trace is replayed at DP>1. Job 1854038 (2026-09-17) is the first capture with
those collectives instrumented (`param_and_grad_buffer._record_dp_collective`), so the
synthesis can be checked against recorded bytes instead of reasoning:

    cell                    stage  layers  measured AllGather bytes
    tp4pp2-dp4-*            0/1    32      8,473,165,824 / 8,473,176,064
    tp4pp4-layout-dp2-ogr   0      17      4,815,942,144
                            1,2    16      3,901,038,592
                            3      15      4,328,322,560

The per-stage model below reproduces every one of those to +0.01% (the remainder is bucket
padding). The previous uniform N/(TP*PP) assumption is off by -7.9% to +13.7% on the
production layout and is what these tests exist to prevent coming back.
"""
from __future__ import annotations

import pytest
import yaml

from simulon.backend.dag.trace_tracer import (
    ParallelConfig,
    _dp_grad_sync_bytes,
    _pipeline_layout_groups,
    _stage_param_count,
)
from simulon.config.workload import MegatronWorkload

LAYOUT = "Et*5|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*3L"

BASE = {
    "num-layers": 64, "hidden-size": 5120, "ffn-hidden-size": 25600,
    "num-attention-heads": 64, "num-query-groups": 8, "kv-channels": 128,
    "seq-length": 4096, "padded-vocab-size": 262144, "swiglu": True,
    "untie-embeddings-and-output-weights": True,
}


def _wl(**over) -> MegatronWorkload:
    return MegatronWorkload.model_validate(
        {"framework": "megatron", "config": {**BASE, **over}})


def _cfg(tp: int, pp: int, dp: int) -> ParallelConfig:
    return ParallelConfig(tp=tp, cp=1, ep=1, dp=dp, pp=pp, etp=tp, edp=dp,
                          num_gpus=tp * pp * dp)


def test_layout_parses_to_groups_layers_and_endpoints():
    g = _pipeline_layout_groups(LAYOUT)
    assert len(g) == 16
    assert sum(n for n, _, _ in g) == 64                 # every layer accounted for
    assert g[0] == (5, True, False)                      # Et*5 -- embedding + 5 layers
    assert g[-1] == (3, False, True)                     # t*3L -- 3 layers + output head
    assert all(n == 4 and not e and not l for n, e, l in g[1:-1])


def test_backslash_escaped_layout_parses_identically():
    """Hydra configs carry the layout escaped (Et\\*5\\|t\\*4...)."""
    assert _pipeline_layout_groups(LAYOUT.replace("*", "\\*").replace("|", "\\|")) \
        == _pipeline_layout_groups(LAYOUT)


@pytest.mark.parametrize("stage,layers", [(0, 17), (1, 16), (2, 16), (3, 15)])
def test_layout_stage_owns_the_chunks_megatron_assigns_it(stage, layers):
    """Chunk c of stage s is layout group s + c*pp."""
    wl = _wl(**{"pipeline-model-parallel-layout": LAYOUT})
    n = _stage_param_count(wl, _cfg(4, 4, 2), stage)
    from simulon.backend.memory import ModelDims, embedding_param_count, layer_param_count
    d = ModelDims.from_config(wl.config)
    embeds = 1 if stage in (0, 3) else 0                 # E on stage 0, L on stage 3
    assert n == layers * layer_param_count(d) + embeds * embedding_param_count(d)


# (stage, measured all-gather bytes) from job 1854038, tp4pp4-layout-dp2-ogr
@pytest.mark.parametrize("stage,measured", [
    (0, 4_815_942_144), (1, 3_901_038_592), (2, 3_901_038_592), (3, 4_328_322_560)])
def test_layout_allgather_bytes_match_the_measured_trace(stage, measured):
    wl = _wl(**{"pipeline-model-parallel-layout": LAYOUT})
    _rs, ag = _dp_grad_sync_bytes(wl, _cfg(4, 4, 2), stage)
    assert abs(ag / measured - 1) < 0.001, f"stage {stage}: {ag:,} vs measured {measured:,}"


@pytest.mark.parametrize("stage,measured", [(0, 8_473_165_824), (1, 8_473_176_064)])
def test_uniform_pp2_allgather_bytes_match_the_measured_trace(stage, measured):
    """No layout: uniform layers, embedding on stage 0, output head on the last stage.
    With untied embeddings those tie, which is why PP=2 alone could not catch the bug."""
    _rs, ag = _dp_grad_sync_bytes(_wl(), _cfg(4, 2, 4), stage)
    assert abs(ag / measured - 1) < 0.001


def test_reduce_scatter_is_fp32_unless_grad_reduce_in_bf16():
    rs32, ag = _dp_grad_sync_bytes(_wl(), _cfg(4, 2, 4), 0)
    rs16, _ = _dp_grad_sync_bytes(_wl(**{"grad-reduce-in-bf16": True}), _cfg(4, 2, 4), 0)
    assert rs32 == 2 * ag          # fp32 grads vs bf16 params
    assert rs16 == ag              # bf16 grads
    assert abs(rs32 / 16_946_331_648 - 1) < 0.001      # measured, job 1854038


def test_fp8_param_gather_halves_the_all_gather():
    _rs, ag = _dp_grad_sync_bytes(_wl(), _cfg(4, 2, 4), 0)
    _rs8, ag8 = _dp_grad_sync_bytes(_wl(**{"fp8-param-gather": True}), _cfg(4, 2, 4), 0)
    assert ag8 == ag // 2


def test_uniform_split_puts_embedding_first_and_head_last():
    """PP=4, no layout: interior stages carry layers only."""
    from simulon.backend.memory import ModelDims, embedding_param_count, layer_param_count
    wl = _wl()
    d = ModelDims.from_config(wl.config)
    interior = _stage_param_count(wl, _cfg(4, 4, 2), 1)
    assert interior == 16 * layer_param_count(d)
    assert _stage_param_count(wl, _cfg(4, 4, 2), 0) == interior + embedding_param_count(d)
    assert _stage_param_count(wl, _cfg(4, 4, 2), 3) == interior + embedding_param_count(d)
