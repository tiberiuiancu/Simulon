"""Interleaved-1F1B (VPP / layout) synthesis, gated against measurement.

`interleave_synth` builds an interleaved trace from a plain-PP one, which is what unblocks
98 of the 141 measured Qwen3-32B configurations -- every production run. Two things have to
hold, and both were checked against the captured layout trace (job 1854038,
`tp4pp4-layout-chunks`, the production 16-chunk layout at PP=4, M=64):

  * the SCHEDULE must be Megatron's, slot for slot. The recorded (direction, virtual
    microbatch, model chunk) sequence is reproduced exactly for all 512 slots on every
    stage; the constants pinned below are taken from that recording.
  * per-chunk COMPUTE must follow the chunk's layer count. Synthesizing the production
    layout from the plain PP4 trace and comparing against the measured layout trace gives
    mean |error| 5.0%, max 15.7% over 16 chunks -- the same order as the +/-6% spread of
    t_layer across parallelism strategies, i.e. at the level of the intrinsic variation.
    The residual is structured: per-stage hardware variation (which one per-layer rate
    cannot express) and an under-derived output-head term.
"""
from __future__ import annotations

import pytest

from simulon.backend.dag.interleave_synth import (
    chunk_id,
    group_of,
    interleaved_actions,
    layer_span,
    num_warmup,
    parse_layout,
    uniform_layout,
)

PROD_LAYOUT = "Et*5|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*3L"


def test_layout_parses_groups_layers_and_endpoints():
    g = parse_layout(PROD_LAYOUT)
    assert len(g) == 16
    assert sum(x.n_layers for x in g) == 64
    assert (g[0].n_layers, g[0].has_embedding, g[0].has_head) == (5, True, False)
    assert (g[-1].n_layers, g[-1].has_embedding, g[-1].has_head) == (3, False, True)
    assert all(x.n_layers == 4 and not x.has_embedding and not x.has_head for x in g[1:-1])


def test_escaped_layout_parses_identically():
    assert parse_layout(PROD_LAYOUT.replace("*", "\\*").replace("|", "\\|")) == parse_layout(PROD_LAYOUT)


def test_uniform_layout_matches_num_layers_per_virtual_pipeline_stage():
    g = uniform_layout(64, pp=4, chunks=4)
    assert len(g) == 16 and all(x.n_layers == 4 for x in g)
    assert g[0].has_embedding and g[-1].has_head


@pytest.mark.parametrize("stage,expected", [(0, [5, 4, 4, 4]), (3, [4, 4, 4, 3])])
def test_stage_owns_the_groups_megatron_assigns(stage, expected):
    """Chunk c of stage s is layout group s + c*pp -- measured: stage 0 holds 17 layers
    and the embedding, stage 3 holds 15 and the output head."""
    g = parse_layout(PROD_LAYOUT)
    assert [g[group_of(stage, c, 4)].n_layers for c in range(4)] == expected
    assert g[group_of(0, 0, 4)].has_embedding
    assert g[group_of(3, 3, 4)].has_head


def test_layer_spans_tile_the_model_without_gaps_or_overlap():
    g = parse_layout(PROD_LAYOUT)
    covered = [i for j in range(len(g)) for i in layer_span(g, j)]
    assert covered == list(range(64))


def test_chunk_id_rule_matches_the_recorded_trace():
    """Forward runs chunks in order, pp virtual microbatches at a time; backward reverses."""
    first12 = [(v, chunk_id(v, True, 4, 4)) for v in range(12)]
    assert first12 == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 1), (5, 1), (6, 1), (7, 1),
                       (8, 2), (9, 2), (10, 2), (11, 2)]
    assert [chunk_id(v, False, 4, 4) for v in range(4)] == [3, 3, 3, 3]


@pytest.mark.parametrize("stage,warm", [(0, 18), (3, 12)])
def test_warmup_depth_matches_the_recorded_trace(stage, warm):
    """Measured forwards before the first backward were 19 and 13: the warmup loop plus the
    steady loop's opening forward."""
    assert num_warmup(stage, pp=4, chunks=4, n_mb=64) == warm
    acts = interleaved_actions(stage, 4, 4, 64)
    first_bwd = next(i for i, a in enumerate(acts) if a[0] == "bwd")
    assert first_bwd == warm + 1


def test_schedule_is_complete_and_well_formed():
    for stage in range(4):
        acts = interleaved_actions(stage, 4, 4, n_mb=64)
        assert len(acts) == 2 * 64 * 4                       # every microbatch, both directions
        fwd = [(v, c) for d, v, c in acts if d == "fwd"]
        bwd = [(v, c) for d, v, c in acts if d == "bwd"]
        assert sorted(v for v, _ in fwd) == list(range(256))
        assert sorted(v for v, _ in bwd) == list(range(256))
        # a microbatch's backward never precedes its forward on the same stage
        fwd_at = {v: i for i, (d, v, _) in enumerate(acts) if d == "fwd"}
        bwd_at = {v: i for i, (d, v, _) in enumerate(acts) if d == "bwd"}
        assert all(bwd_at[v] > fwd_at[v] for v in range(256))


def test_no_vpp_reduces_to_plain_1f1b_warmup():
    """chunks=1 must degenerate to the NON-interleaved schedule.

    Regression test. This originally asserted the interleaved form `(pp-stage-1)*2` and so
    certified the bug it was meant to catch: at chunks=1 that fills the pipeline twice as
    deep, which produced a 31 s bubble against a 5.7 s measurement (8 nodes, PP8, a layout
    with exactly pp groups -- an uneven plain split, not an interleaved schedule). The
    correct rule is pp_synth's, measured against real PP2/PP4 traces: `pp - stage` forwards
    precede the first backward.
    """
    for pp in (2, 4, 8):
        for stage in range(pp):
            acts = interleaved_actions(stage, pp, 1, 64)
            assert all(c == 0 for _, _, c in acts)
            first_bwd = next(i for i, a in enumerate(acts) if a[0] == "bwd")
            assert first_bwd == pp - stage, f"pp{pp} stage{stage}: {first_bwd} != {pp - stage}"


def test_chunks_one_matches_pp_synth_exactly():
    """The degenerate case must reproduce the existing, separately-validated plain path."""
    from simulon.backend.dag.pp_synth import _oneF1B_actions
    for pp in (2, 4):
        for stage in range(pp):
            mine = [(d, v) for d, v, _c in interleaved_actions(stage, pp, 1, 16)]
            assert mine == _oneF1B_actions(stage, pp, 16)
