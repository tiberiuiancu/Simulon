"""Unit tests for the Megatron workload trace generator.

No GPU required — uses mock profile loaded from templates/gpu/mock-h100.yaml.
"""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from simulon.config.common import DType
from simulon.config.dc import GPUSpec
from simulon.config.workload import LLMSpec, MegatronParallelism, MegatronTraining, MegatronWorkload
from simulon.workload.megatron import generate_megatron_trace
from simulon.workload.trace import CommOp, CommOpType, ComputeOp, ParallelGroup

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


def _llama7b() -> LLMSpec:
    return LLMSpec(
        name="LLaMA-7B",
        hidden_size=4096,
        num_layers=32,
        num_heads=32,
        vocab_size=32000,
        ffn_hidden_size=11008,
        swiglu=True,
    )


def _workload(
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
    sp: bool = False,
    dist_opt: bool = False,
    num_gpus: int = 8,
    global_batch: int = 8,
    micro_batch: int = 1,
    seq_len: int = 2048,
    iterations: int = 1,
) -> MegatronWorkload:
    return MegatronWorkload(
        framework="megatron",
        model=_llama7b(),
        parallelism=MegatronParallelism(
            tp=tp, pp=pp, ep=ep, sp=sp, distributed_optimizer=dist_opt
        ),
        training=MegatronTraining(
            num_gpus=num_gpus,
            global_batch_size=global_batch,
            micro_batch_size=micro_batch,
            sequence_length=seq_len,
            dtype=DType.bf16,
            iterations=iterations,
        ),
    )


def _mock_profile() -> GPUSpec:
    path = TEMPLATES_DIR / "gpu" / "mock-h100.yaml"
    data = yaml.safe_load(path.read_text())
    return GPUSpec.model_validate(data)


# ---------------------------------------------------------------------------
# Test 1: Basic trace shape
# ---------------------------------------------------------------------------


def test_basic_trace_shape():
    """LLaMA-7B, tp=2 pp=1 dp=4, 8 GPUs — verify comm op count."""
    tp, pp, ep = 2, 1, 1
    num_gpus = 8
    global_batch = 8
    micro_batch = 1
    seq_len = 2048

    dp = num_gpus // (tp * pp * ep)  # = 4
    num_microbatches = global_batch // (micro_batch * dp)  # = 2
    layers = 32
    sp = False
    dist_opt = False

    wl = _workload(tp=tp, pp=pp, num_gpus=num_gpus, global_batch=global_batch,
                   micro_batch=micro_batch, seq_len=seq_len)
    trace = generate_megatron_trace(wl, _llama7b())

    comm_ops = [op for op in trace.ops if isinstance(op, CommOp)]

    # Expected comm count formula
    init = 4 + 1 + 1  # 4× dp_allreduce + 1× dp_allgather + 1× tp_broadcast (pp=1, no emb sync)
    per_fwd_mb = (2 if tp > 1 else 0)  # attention mask broadcasts
    per_fwd_mb += layers * (2 if not sp else 3)  # per-layer TP comms
    per_fwd_mb += 4  # logit: 3 tp_allreduce + 1 dp_allreduce (pp=1)
    per_fwd_mb += 0  # no ISend/IRecv (pp=1)

    per_bwd_mb = layers * 2  # 2 AllReduce tp per layer (sp=False)

    step = 1 + 2  # 1× dp_allreduce + 2× tp_allreduce (layernorm + NaN)

    expected = init + num_microbatches * (per_fwd_mb + per_bwd_mb) + step
    assert len(comm_ops) == expected, f"Expected {expected} comm ops, got {len(comm_ops)}"

    # Verify trace metadata
    assert trace.framework == "megatron"
    assert trace.tp == tp
    assert trace.pp == pp
    assert trace.dp == dp
    assert trace.num_gpus == num_gpus


# ---------------------------------------------------------------------------
# Test 2: GPU profile fills compute_time_us
# ---------------------------------------------------------------------------


def test_with_gpu_profile():
    """With mock-h100 profile, all ComputeOps should have non-None compute_time_us."""
    wl = _workload(tp=1, pp=1, num_gpus=4, global_batch=4)
    profile = _mock_profile()
    trace = generate_megatron_trace(wl, _llama7b(), gpu_profile=profile)

    compute_ops = [op for op in trace.ops if isinstance(op, ComputeOp)]
    assert len(compute_ops) > 0

    for op in compute_ops:
        assert op.compute_time_us is not None, (
            f"kernel={op.kernel!r} has no compute_time_us with profile"
        )
        assert op.compute_time_us > 0


# ---------------------------------------------------------------------------
# Test 3: SP flag changes comm pattern
# ---------------------------------------------------------------------------


def test_sp_changes_comm_pattern():
    """With sp=True, forward RowLinear ops are reduce_scatter not all_reduce."""
    seq_len, micro_batch, hidden = 2048, 1, 4096
    tp_comm_size = 2 * seq_len * micro_batch * hidden  # bf16 bytes

    wl_sp = _workload(tp=2, pp=1, sp=True, num_gpus=8, global_batch=8,
                      micro_batch=micro_batch, seq_len=seq_len)
    wl_no_sp = _workload(tp=2, pp=1, sp=False, num_gpus=8, global_batch=8,
                         micro_batch=micro_batch, seq_len=seq_len)

    trace_sp = generate_megatron_trace(wl_sp, _llama7b())
    trace_no_sp = generate_megatron_trace(wl_no_sp, _llama7b())

    # RowLinear comm ops are identified by their size (tp_comm_size)
    def layer_tp_ops(trace):
        return [
            op for op in trace.ops
            if isinstance(op, CommOp)
            and op.stage == "forward"
            and op.group == ParallelGroup.tp
            and op.size_bytes == tp_comm_size
        ]

    sp_layer_ops = layer_tp_ops(trace_sp)
    no_sp_layer_ops = layer_tp_ops(trace_no_sp)

    sp_types = {op.comm_type for op in sp_layer_ops}
    no_sp_types = {op.comm_type for op in no_sp_layer_ops}

    # SP: RowLinear → ReduceScatter; ColumnLinear backward during fwd → AllGather
    assert CommOpType.reduce_scatter in sp_types
    assert CommOpType.all_gather in sp_types
    assert CommOpType.all_reduce not in sp_types

    # No SP: RowLinear → AllReduce; no AllGather/ReduceScatter
    assert CommOpType.all_reduce in no_sp_types
    assert CommOpType.reduce_scatter not in no_sp_types
    assert CommOpType.all_gather not in no_sp_types


# ---------------------------------------------------------------------------
# Test 4: Distributed optimizer changes step ops
# ---------------------------------------------------------------------------


def test_distributed_optimizer_step():
    """dist_opt=True → step has reduce_scatter + all_gather; False → all_reduce."""
    wl_dist = _workload(tp=2, pp=1, dist_opt=True, num_gpus=8, global_batch=8)
    wl_plain = _workload(tp=2, pp=1, dist_opt=False, num_gpus=8, global_batch=8)

    def step_dp_ops(wl):
        trace = generate_megatron_trace(wl, _llama7b())
        return [
            op for op in trace.ops
            if isinstance(op, CommOp) and op.stage == "step" and op.group == ParallelGroup.dp
        ]

    dist_ops = step_dp_ops(wl_dist)
    plain_ops = step_dp_ops(wl_plain)

    assert len(dist_ops) == 2
    assert dist_ops[0].comm_type == CommOpType.reduce_scatter
    assert dist_ops[1].comm_type == CommOpType.all_gather

    assert len(plain_ops) == 1
    assert plain_ops[0].comm_type == CommOpType.all_reduce


# ---------------------------------------------------------------------------
# Test 5: Pipeline parallelism includes ISend/IRecv
# ---------------------------------------------------------------------------


def test_pipeline_parallelism_isend_irecv():
    """pp=2 config → trace contains isend and irecv ops in pp group."""
    # tp=2, pp=2, dp=2, num_gpus=8
    wl = _workload(tp=2, pp=2, num_gpus=8, global_batch=8, micro_batch=1)
    trace = generate_megatron_trace(wl, _llama7b())

    pp_ops = [op for op in trace.ops if isinstance(op, CommOp) and op.group == ParallelGroup.pp]
    assert len(pp_ops) > 0

    types = {op.comm_type for op in pp_ops}
    assert CommOpType.isend in types
    assert CommOpType.irecv in types


# ---------------------------------------------------------------------------
# Test 6: extra="forbid" enforcement
# ---------------------------------------------------------------------------


def test_extra_forbid_megatron():
    """A MegatronWorkload dict with an extra key raises ValidationError."""
    data = {
        "framework": "megatron",
        "model": {
            "hidden_size": 4096, "num_layers": 32, "num_heads": 32, "vocab_size": 32000
        },
        "parallelism": {"tp": 1},
        "training": {
            "num_gpus": 8, "global_batch_size": 8, "micro_batch_size": 1, "sequence_length": 2048
        },
        "inference": {"num_gpus": 8, "batch_size": 1, "seq_length": 2048},  # stray key
    }
    with pytest.raises(ValidationError):
        MegatronWorkload.model_validate(data)


def test_extra_forbid_does_not_reject_valid():
    """A valid MegatronWorkload dict parses without error."""
    data = {
        "framework": "megatron",
        "model": {
            "hidden_size": 4096, "num_layers": 32, "num_heads": 32, "vocab_size": 32000
        },
        "parallelism": {"tp": 1},
        "training": {
            "num_gpus": 4, "global_batch_size": 4, "micro_batch_size": 1, "sequence_length": 2048
        },
    }
    wl = MegatronWorkload.model_validate(data)
    assert wl.framework == "megatron"


# ---------------------------------------------------------------------------
# Test 7: Comm size correctness
# ---------------------------------------------------------------------------


def test_comm_size_tp_allreduce():
    """TP AllReduce after attention layer should be 2 * seq_len * micro_batch * hidden."""
    tp, seq_len, micro_batch, hidden = 2, 2048, 1, 4096
    B = 2  # bf16 bytes per element
    expected_size = B * seq_len * micro_batch * hidden

    wl = _workload(tp=tp, pp=1, sp=False, num_gpus=8, global_batch=8,
                   micro_batch=micro_batch, seq_len=seq_len)
    trace = generate_megatron_trace(wl, _llama7b())

    fwd_tp_allreduce = [
        op for op in trace.ops
        if isinstance(op, CommOp)
        and op.stage == "forward"
        and op.group == ParallelGroup.tp
        and op.comm_type == CommOpType.all_reduce
    ]
    # All layer TP AllReduces (excluding logit cross-entropy which has different size)
    layer_comms = [op for op in fwd_tp_allreduce if op.size_bytes == expected_size]
    assert len(layer_comms) > 0, "No TP AllReduce with expected size found"
    for op in layer_comms:
        assert op.size_bytes == expected_size


def test_comm_size_step_dp_allreduce():
    """Step DP AllReduce size should equal 4 * total_params // pp."""
    tp, pp = 2, 1
    hidden, vocab_size = 4096, 32000
    ffn_hidden = 11008
    num_layers = 32

    embedding_params = vocab_size * hidden // tp
    mlp_factor = 3  # swiglu
    attn_per = 4 * hidden * hidden // tp
    mlp_per = mlp_factor * hidden * ffn_hidden // tp
    params_per_stage = (attn_per + mlp_per) * num_layers
    total_params = embedding_params + pp * params_per_stage
    expected = 4 * total_params // pp

    wl = _workload(tp=tp, pp=pp, dist_opt=False, num_gpus=8, global_batch=8)
    trace = generate_megatron_trace(wl, _llama7b())

    step_dp = [
        op for op in trace.ops
        if isinstance(op, CommOp) and op.stage == "step" and op.group == ParallelGroup.dp
    ]
    assert len(step_dp) == 1
    assert step_dp[0].size_bytes == expected
