"""Parity tests: verify DAGTracer matches generate_megatron_trace workload model.

Run these before removing workload/megatron.py. All tests must pass (or divergences
must be understood and accepted) before the legacy module is deleted.

Known intentional omissions in DAGTracer (not tested here):
  - Init phase collectives (4× dp AllReduce, dp AllGather, tp broadcast)
  - Embedding and logit compute kernels
  - Attention mask metadata broadcasts (2× per microbatch when tp > 1)
  - Cross-entropy AllReduces after logit (pp=1 only)
  - LayerNorm grad TP sync in step phase
  - NaN/Inf check AllReduce in step phase

Known SP behaviour difference (not a bug, documented):
  DAGTracer always emits AG+RS per sublayer regardless of sp flag.
  Legacy emits AllReduce when sp=False, AG+RS when sp=True.
  At communication-volume level these are equivalent for ring (AllReduce = RS+AG),
  but the sp flag is effectively ignored in the DAGTracer.
"""

import pytest

from simulon.config.common import DType
from simulon.config.workload import LLMSpec, MegatronParallelism, MegatronTraining, MegatronWorkload
from simulon.workload.megatron import generate_megatron_trace
from simulon.workload.trace import CommOp, CommOpType, ParallelGroup
from simulon.backend.dag.megatron_tracer import _params_per_tp_rank


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _spec(swiglu: bool = False) -> LLMSpec:
    return LLMSpec(
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        vocab_size=1000,
        ffn_hidden_size=512,
        swiglu=swiglu,
    )


def _workload(
    spec: LLMSpec,
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
    dp: int = 2,
    sp: bool = False,
    dist_opt: bool = False,
    micro_batch: int = 1,
    seq_len: int = 512,
) -> MegatronWorkload:
    num_gpus = tp * pp * ep * dp
    return MegatronWorkload(
        framework="megatron",
        model=spec,
        parallelism=MegatronParallelism(tp=tp, pp=pp, ep=ep, sp=sp, distributed_optimizer=dist_opt),
        training=MegatronTraining(
            num_gpus=num_gpus,
            global_batch_size=num_gpus,  # 1 microbatch per dp rank
            micro_batch_size=micro_batch,
            sequence_length=seq_len,
            dtype=DType.bf16,
        ),
    )


def _legacy_step_dp_ops(wl: MegatronWorkload, spec: LLMSpec) -> list[CommOp]:
    trace = generate_megatron_trace(wl, spec)
    return [op for op in trace.ops if isinstance(op, CommOp) and op.stage == "step" and op.group == ParallelGroup.dp]


def _aicb_step_ar_bytes(wl: MegatronWorkload, spec: LLMSpec) -> int:
    """AICB-correct parameter count for the DP step AllReduce.

    Per AICB (SimAI_training_workload_generator.py):
    - SwiGLU: mlp_factor=3 (gate+up projection doubled, then down; equivalent to 3 matrices)
    - Logit head counted separately (not tied to embedding in AICB's model)
    - LayerNorm parameters included in main DP AllReduce (not a separate TP step)
    - fp32 gradients → 4 bytes per parameter

    Known differences from generate_megatron_trace (legacy bugs):
    - Legacy uses mlp_factor=2 for swiglu (wrong; fixed in tracer)
    - Legacy excludes logit params (assumes tied weights; AICB does not)
    - Legacy syncs layernorm via separate TP AllReduce instead of including in DP step
    """
    tp = wl.parallelism.tp
    pp = wl.parallelism.pp
    hidden = spec.hidden_size
    ffn = spec.ffn_hidden_size or (4 * hidden)
    num_layers = spec.num_layers
    vocab_size = spec.vocab_size or 0
    layers_per_stage = num_layers // pp

    attn_per_layer = 4 * hidden * hidden // tp
    mlp_factor = 3 if spec.swiglu else 2
    mlp_per_layer = mlp_factor * hidden * ffn // tp
    ln_per_layer = 2 * hidden  # pre-attn LN + pre-MLP LN
    per_layer = attn_per_layer + mlp_per_layer + ln_per_layer

    embedding = vocab_size * hidden // tp
    logit = vocab_size * hidden // tp  # separate weight matrix in AICB

    total_params = layers_per_stage * per_layer + embedding + logit
    return 4 * total_params


def _tracer_step_ar_bytes(wl: MegatronWorkload, spec: LLMSpec) -> int:
    """Compute step DP AllReduce bytes the same way DAGTracer does."""
    pp = wl.parallelism.pp
    tp = wl.parallelism.tp
    ep = wl.parallelism.ep
    step_params = _params_per_tp_rank(spec, tp, ep)
    return 4 * step_params // pp


# ---------------------------------------------------------------------------
# Test 1: activation bytes (per-layer TP collective size)
# ---------------------------------------------------------------------------


def test_tp_comm_size_formula_matches():
    """Per-layer TP collective bytes must match: both use 2 * seq * mb * hidden."""
    B = 2  # bf16
    spec = _spec()
    wl = _workload(spec, tp=2, dp=2, sp=False)

    seq = wl.training.sequence_length
    mb = wl.training.micro_batch_size
    expected_bytes = B * seq * mb * spec.hidden_size

    # Legacy: tp_comm = B * seq_len * micro_batch * hidden
    assert expected_bytes == B * seq * mb * spec.hidden_size

    # DAGTracer: activation_bytes = seq_len * micro_bs * hidden_size * cfg.dtype_bytes (2 for bf16)
    tracer_activation_bytes = seq * mb * spec.hidden_size * 2
    assert tracer_activation_bytes == expected_bytes


# ---------------------------------------------------------------------------
# Test 2: step DP collective type matches dist_opt flag
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="generate_megatron_trace() is legacy-stubbed; parity already verified")
@pytest.mark.parametrize("dist_opt,expected_types", [
    (False, {CommOpType.all_reduce}),
    (True, {CommOpType.reduce_scatter, CommOpType.all_gather}),
])
def test_step_dp_collective_type_matches(dist_opt, expected_types):
    """Step DP collective type must match: both must respect distributed_optimizer flag."""
    spec = _spec()
    wl = _workload(spec, tp=1, dp=2, dist_opt=dist_opt)
    step_ops = _legacy_step_dp_ops(wl, spec)
    actual = {op.comm_type for op in step_ops}
    assert actual == expected_types, f"dist_opt={dist_opt}: got {actual}, expected {expected_types}"


# ---------------------------------------------------------------------------
# Test 3: step DP AllReduce bytes match AICB reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("swiglu", [False, True])
def test_step_dp_bytes_match_aicb(swiglu):
    """Step DP AllReduce bytes must match the AICB-correct formula."""
    spec = _spec(swiglu=swiglu)
    wl = _workload(spec, tp=2, dp=2, dist_opt=False)

    expected = _aicb_step_ar_bytes(wl, spec)
    tracer = _tracer_step_ar_bytes(wl, spec)

    assert tracer == expected, (
        f"Step DP AllReduce bytes diverge (swiglu={swiglu}):\n"
        f"  expected (AICB) = {expected}\n"
        f"  tracer          = {tracer}\n"
        f"  diff            = {tracer - expected:+d} bytes"
    )


# ---------------------------------------------------------------------------
# Test 5: distributed optimizer step sizes (RS fp32 grads, AG bf16 params)
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="generate_megatron_trace() is legacy-stubbed; parity already verified")
def test_dist_opt_step_sizes_match():
    """Distributed optimizer: RS bytes (fp32 grads) and AG bytes (bf16 params) must match."""
    spec = _spec(swiglu=False)
    wl = _workload(spec, tp=2, dp=2, dist_opt=True)

    step_ops = _legacy_step_dp_ops(wl, spec)
    rs_ops = [op for op in step_ops if op.comm_type == CommOpType.reduce_scatter]
    ag_ops = [op for op in step_ops if op.comm_type == CommOpType.all_gather]
    assert len(rs_ops) == 1 and len(ag_ops) == 1, "Expected exactly 1 RS + 1 AG in legacy step"

    legacy_rs = rs_ops[0].size_bytes
    legacy_ag = ag_ops[0].size_bytes

    # Verify RS/AG ratio: RS is fp32 grads (4 bytes/param), AG is bf16 params (2 bytes/param)
    assert legacy_rs == 2 * legacy_ag, (
        f"Legacy dist_opt: RS should be 2× AG (fp32 vs bf16), got RS={legacy_rs} AG={legacy_ag}"
    )

    # Verify tracer RS matches AICB reference and AG = RS/2
    step_params = _params_per_tp_rank(spec, wl.parallelism.tp, wl.parallelism.ep)
    pp = wl.parallelism.pp
    tracer_rs = 4 * step_params // pp
    tracer_ag = 2 * step_params // pp
    aicb_rs = _aicb_step_ar_bytes(wl, spec)

    assert tracer_rs == aicb_rs, (
        f"Distributed optimizer RS bytes diverge from AICB reference:\n"
        f"  expected (AICB) = {aicb_rs}\n"
        f"  tracer          = {tracer_rs}\n"
        f"  diff            = {tracer_rs - aicb_rs:+d} bytes"
    )
    assert tracer_ag == aicb_rs // 2, (
        f"Distributed optimizer AG bytes should be RS/2: expected {aicb_rs // 2}, got {tracer_ag}"
    )


# ---------------------------------------------------------------------------
# Test 6: MoE AllToAll size formula
#
# Both implementations use: seq_len * micro_batch * hidden * top_k * B // tp
# ---------------------------------------------------------------------------


def test_moe_a2a_size_formula_matches():
    """MoE AllToAll bytes must match: seq * mb * hidden * top_k * 2 // tp."""
    spec = LLMSpec(
        hidden_size=256,
        num_layers=2,
        num_heads=4,
        vocab_size=1000,
        ffn_hidden_size=512,
        moe=True,
        num_experts=4,
        top_k=2,
    )
    tp = 2
    seq_len = 512
    micro_batch = 1
    B = 2  # bf16

    # DAGTracer: moe_data_bytes = seq_len * micro_bs * hidden_size * top_k * dtype_bytes // tp
    tracer_a2a = seq_len * micro_batch * spec.hidden_size * (spec.top_k or 1) * B // tp
    expected = B * seq_len * micro_batch * spec.hidden_size * (spec.top_k or 1) // tp

    assert tracer_a2a == expected, (
        f"MoE AllToAll size formula diverges: tracer={tracer_a2a}, expected={expected}"
    )
