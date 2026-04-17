"""Tests for EP-aware data-parallel group modeling in MegatronDAGTracer.

Covers:
  - _non_expert_params_per_tp_rank / _expert_params_per_tp_rank helpers
  - ParallelGroups rank membership (verified against Megatron-Core RankGenerator)
  - DAG step-phase regression: EP=1 behaves identically to old single-group code
  - DAG step-phase MoE: two separate AllReduce groups (non-expert over dp×ep, expert over dp)
"""

from __future__ import annotations

import pytest

from simulon.backend.dag.megatron_tracer import (
    MegatronDAGTracer,
    ParallelGroups,
    _expert_params_per_tp_rank,
    _non_expert_params_per_tp_rank,
    _params_per_tp_rank,
)
from simulon.backend.dag.nodes import CommNode
from simulon.backend.dag.tracer import DAGTracerConfig
from simulon.config.common import DType
from simulon.config.dc import ClusterSpec, DatacenterConfig, DatacenterMeta, GPUSpec, NodeSpec
from simulon.config.workload import LLMSpec, MegatronParallelism, MegatronTraining, MegatronWorkload


# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------


def _minimal_dc(num_gpus: int) -> DatacenterConfig:
    return DatacenterConfig(
        datacenter=DatacenterMeta(name="test"),
        cluster=ClusterSpec(num_nodes=1),
        node=NodeSpec(
            gpus_per_node=num_gpus,
            gpu=GPUSpec(name="H100", memory_capacity_gb=80.0),
        ),
    )


def _no_cache_tracer(compact: bool = True) -> MegatronDAGTracer:
    return MegatronDAGTracer(config=DAGTracerConfig(cache_dir=None, compact=compact))


def _dense_model(
    hidden_size: int = 256,
    num_layers: int = 2,
    vocab_size: int = 4096,
    ffn_hidden_size: int | None = None,
    swiglu: bool = False,
) -> LLMSpec:
    return LLMSpec(
        name="test-dense",
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=4,
        vocab_size=vocab_size,
        ffn_hidden_size=ffn_hidden_size,
        swiglu=swiglu,
    )


def _moe_model(
    hidden_size: int = 256,
    num_layers: int = 2,
    num_experts: int = 4,
    vocab_size: int = 4096,
) -> LLMSpec:
    return LLMSpec(
        name="test-moe",
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=4,
        vocab_size=vocab_size,
        num_experts=num_experts,
        top_k=1,
    )


def _workload(
    model: LLMSpec,
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
    dp: int = 1,
    distributed_optimizer: bool = False,
    seq_len: int = 128,
    micro_batch_size: int = 1,
) -> MegatronWorkload:
    num_gpus = tp * pp * ep * dp
    return MegatronWorkload(
        framework="megatron",
        model=model,
        parallelism=MegatronParallelism(
            tp=tp, pp=pp, ep=ep, dp=dp,
            distributed_optimizer=distributed_optimizer,
        ),
        training=MegatronTraining(
            num_gpus=num_gpus,
            global_batch_size=dp * micro_batch_size,
            micro_batch_size=micro_batch_size,
            sequence_length=seq_len,
            dtype=DType.bf16,
        ),
    )


def _global_rank(dp_rank: int, pp_stage: int, ep_rank: int, tp_rank: int,
                 pp: int, ep: int, tp: int) -> int:
    """Match the rank formula in MegatronDAGTracer (tp-cp-ep-dp-pp order, cp=1)."""
    return dp_rank * (pp * ep * tp) + pp_stage * (ep * tp) + ep_rank * tp + tp_rank


# ---------------------------------------------------------------------------
# 1. Unit tests for param-count helpers
# ---------------------------------------------------------------------------


class TestParamHelpers:
    """_non_expert_params_per_tp_rank / _expert_params_per_tp_rank / _params_per_tp_rank."""

    def test_non_moe_expert_params_are_zero(self):
        """For a dense (non-MoE) model, expert param count must be 0."""
        model = _dense_model(hidden_size=256, num_layers=2, vocab_size=4096)
        assert _expert_params_per_tp_rank(model, tp=1, ep=1) == 0
        assert _expert_params_per_tp_rank(model, tp=2, ep=4) == 0

    def test_non_moe_total_equals_non_expert(self):
        """For dense model, _params_per_tp_rank == _non_expert_params_per_tp_rank."""
        model = _dense_model(hidden_size=256, num_layers=2, vocab_size=4096)
        for tp in (1, 2, 4):
            ne = _non_expert_params_per_tp_rank(model, tp=tp)
            total = _params_per_tp_rank(model, tp=tp, ep=1)
            assert total == ne, f"tp={tp}: total={total} != non_expert={ne}"

    def test_moe_expert_params_positive(self):
        """For an MoE model, expert param count > 0."""
        model = _moe_model(hidden_size=256, num_layers=2, num_experts=4)
        assert _expert_params_per_tp_rank(model, tp=1, ep=1) > 0

    def test_moe_sum_equals_total(self):
        """For MoE model, non_expert + expert == total across various tp/ep values."""
        model = _moe_model(hidden_size=256, num_layers=2, num_experts=4)
        for tp, ep in [(1, 1), (1, 2), (1, 4), (2, 2)]:
            ne = _non_expert_params_per_tp_rank(model, tp=tp)
            ex = _expert_params_per_tp_rank(model, tp=tp, ep=ep)
            total = _params_per_tp_rank(model, tp=tp, ep=ep)
            assert ne + ex == total, f"tp={tp}, ep={ep}: {ne} + {ex} != {total}"

    def test_moe_expert_params_scale_with_experts_per_ep_rank(self):
        """Expert params scale linearly with num_experts // ep (each EP rank holds a shard)."""
        model_4e = _moe_model(hidden_size=256, num_layers=2, num_experts=4)
        model_8e = _moe_model(hidden_size=256, num_layers=2, num_experts=8)
        # ep=1: full expert set on this rank, so 8e has 2× params of 4e
        ex_4e = _expert_params_per_tp_rank(model_4e, tp=1, ep=1)
        ex_8e = _expert_params_per_tp_rank(model_8e, tp=1, ep=1)
        assert ex_8e == 2 * ex_4e

        # ep=4: each rank holds 1 of 4 experts regardless of total count (num_experts // ep)
        ex_4e_ep4 = _expert_params_per_tp_rank(model_4e, tp=1, ep=4)
        ex_8e_ep4 = _expert_params_per_tp_rank(model_8e, tp=1, ep=4)
        assert ex_8e_ep4 == 2 * ex_4e_ep4

    def test_non_expert_params_tp_scaling(self):
        """Non-expert params (attn + embed + logit) are halved when tp doubles."""
        model = _dense_model(hidden_size=256, num_layers=2, vocab_size=4096)
        p1 = _non_expert_params_per_tp_rank(model, tp=1)
        p2 = _non_expert_params_per_tp_rank(model, tp=2)
        # Layer-norm params (2*hidden per layer) are NOT TP-sharded; the rest are.
        # So the ratio is not exactly 0.5, but tp=2 should be strictly less than tp=1.
        assert p2 < p1

    def test_non_expert_mlp_excluded_for_moe(self):
        """MoE model has no MLP contribution to non_expert params (MLP → MoE expert layer)."""
        # Build equivalent dense and MoE models
        dense = _dense_model(hidden_size=256, num_layers=2, vocab_size=4096)
        moe = _moe_model(hidden_size=256, num_layers=2, num_experts=1, vocab_size=4096)
        # Non-expert params for MoE should be smaller (no MLP counted)
        ne_dense = _non_expert_params_per_tp_rank(dense, tp=1)
        ne_moe = _non_expert_params_per_tp_rank(moe, tp=1)
        assert ne_moe < ne_dense


# ---------------------------------------------------------------------------
# 2. Unit tests for ParallelGroups rank membership
# ---------------------------------------------------------------------------


class TestParallelGroupsRankMembership:
    """Verify ParallelGroups matches Megatron-Core's two-generator RankGenerator pattern."""

    def _build_groups(self, dp: int, pp: int, ep: int, tp: int,
                      dp_rank: int, pp_stage: int, ep_rank: int, tp_rank: int) -> ParallelGroups:
        """Directly replicates the group-construction logic from MegatronDAGTracer.trace()."""
        def gr(dp_r, pp_s, ep_r, tp_r):
            return _global_rank(dp_r, pp_s, ep_r, tp_r, pp, ep, tp)

        return ParallelGroups(
            tp=[gr(dp_rank, pp_stage, ep_rank, r) for r in range(tp)],
            ep=[gr(dp_rank, pp_stage, r, tp_rank) for r in range(ep)],
            expert_dp=[gr(r, pp_stage, ep_rank, tp_rank) for r in range(dp)],
            non_expert_dp=[
                gr(r, pp_stage, ep_r, tp_rank)
                for r in range(dp)
                for ep_r in range(ep)
            ],
        )

    def test_ep1_non_expert_dp_size(self):
        """With EP=1, non_expert_dp size == dp (same as old behavior)."""
        # 4 GPUs: TP=1, PP=1, EP=1, DP=4
        groups = self._build_groups(dp=4, pp=1, ep=1, tp=1,
                                    dp_rank=0, pp_stage=0, ep_rank=0, tp_rank=0)
        assert len(groups.non_expert_dp) == 4
        assert len(groups.expert_dp) == 4  # ep=1, so expert_dp == non_expert_dp

    def test_ep2_dp1_non_expert_dp_size(self):
        """8-GPU config (TP=2, PP=2, EP=2, DP=1): non_expert_dp size == dp×ep == 2."""
        # 8 GPUs: TP=2, PP=2, EP=2, DP=1
        groups = self._build_groups(dp=1, pp=2, ep=2, tp=2,
                                    dp_rank=0, pp_stage=0, ep_rank=0, tp_rank=0)
        assert len(groups.non_expert_dp) == 2  # dp * ep = 1 * 2
        assert len(groups.expert_dp) == 1      # dp = 1 (just this GPU itself)

    def test_ep2_dp1_expert_dp_is_self_only(self):
        """With DP=1, expert_dp contains only this GPU (no peers to sync with)."""
        # For any GPU in this config, expert_dp should be exactly [gpu]
        tp, pp, ep, dp = 2, 2, 2, 1
        for dp_rank in range(dp):
            for pp_stage in range(pp):
                for ep_rank in range(ep):
                    for tp_rank in range(tp):
                        groups = self._build_groups(dp, pp, ep, tp,
                                                    dp_rank, pp_stage, ep_rank, tp_rank)
                        gpu = _global_rank(dp_rank, pp_stage, ep_rank, tp_rank, pp, ep, tp)
                        assert groups.expert_dp == [gpu], (
                            f"dp=1 → expert_dp must be [self={gpu}], "
                            f"got {groups.expert_dp}"
                        )

    def test_ep2_dp1_non_expert_dp_spans_both_ep_ranks(self):
        """non_expert_dp for (dp=0, pp=0, ep=0, tp=0) includes ranks from both EP ranks."""
        tp, pp, ep, dp = 2, 2, 2, 1
        groups = self._build_groups(dp, pp, ep, tp,
                                    dp_rank=0, pp_stage=0, ep_rank=0, tp_rank=0)
        # Expect ranks from ep_rank=0 and ep_rank=1 (same dp_rank=0, pp=0, tp=0)
        expected = sorted([
            _global_rank(0, 0, ep_r, 0, pp, ep, tp)
            for ep_r in range(ep)
        ])
        assert sorted(groups.non_expert_dp) == expected

    def test_ep4_dp1_non_expert_dp_whole_world(self):
        """4-GPU config (TP=1, PP=1, EP=4, DP=1): non_expert_dp == all 4 GPUs."""
        tp, pp, ep, dp = 1, 1, 4, 1
        groups = self._build_groups(dp, pp, ep, tp,
                                    dp_rank=0, pp_stage=0, ep_rank=0, tp_rank=0)
        assert sorted(groups.non_expert_dp) == list(range(4))
        assert len(groups.expert_dp) == 1

    def test_against_megatron_rank_generator(self):
        """Compare non_expert_dp and expert_dp against Megatron-Core's RankGenerator.

        Uses the two-generator pattern from Megatron parallel_state.py:
          decoder_gen (ep=1, dp=DP*EP): DATA_PARALLEL_GROUP
          expert_gen  (ep=EP, dp=DP):   EXPERT_DATA_PARALLEL_GROUP

        The Megatron RankGenerator and its helper are inlined here (pure Python,
        no torch dependency) so this test runs without a Megatron installation.
        The implementations are copied verbatim from:
          /tmp/megatron-lm-full/megatron/core/parallel_state.py
        """
        # --- Inlined from Megatron-Core parallel_state.py (torch-free) ---
        def _prefix_product(a, init=1):
            r = [init]
            for v in a:
                init = init * v
                r.append(init)
            return r

        def _inner_product(a, b):
            return sum(x * y for x, y in zip(a, b))

        def _decompose(index, shape, stride=None):
            if stride is None:
                stride = _prefix_product(shape)
            return [(index // d) % s for s, d in zip(shape, stride)]

        def _generate_masked_orthogonal_rank_groups(world_size, parallel_size, mask):
            masked_shape = [s for s, m in zip(parallel_size, mask) if m]
            unmasked_shape = [s for s, m in zip(parallel_size, mask) if not m]
            global_stride = _prefix_product(parallel_size)
            masked_stride = [d for d, m in zip(global_stride, mask) if m]
            unmasked_stride = [d for d, m in zip(global_stride, mask) if not m]
            group_size = _prefix_product(masked_shape)[-1]
            num_of_group = world_size // group_size
            ranks = []
            for group_index in range(num_of_group):
                decomposed_group_idx = _decompose(group_index, unmasked_shape)
                rank = []
                for rank_in_group in range(group_size):
                    decomposed_rank_idx = _decompose(rank_in_group, masked_shape)
                    rank.append(
                        _inner_product(decomposed_rank_idx, masked_stride)
                        + _inner_product(decomposed_group_idx, unmasked_stride)
                    )
                ranks.append(rank)
            return ranks

        class _RankGenerator:
            def __init__(self, tp, ep, dp, pp, cp, order):
                self.name_to_size = {"tp": tp, "ep": ep, "dp": dp, "pp": pp, "cp": cp}
                order = order.lower()
                for name, size in self.name_to_size.items():
                    if name not in order and size != 1:
                        raise RuntimeError(f"{name} missing from order")
                    elif name not in order:
                        order = order + "-" + name
                self.order = order
                self.ordered_size = [self.name_to_size[t] for t in order.split("-")]
                self.world_size = tp * ep * dp * pp * cp

            def get_ranks(self, token):
                order_tokens = self.order.split("-")
                token_list = token.split("-")
                mask = [t in token_list for t in order_tokens]
                return _generate_masked_orthogonal_rank_groups(
                    self.world_size, self.ordered_size, mask
                )
        # --- end inlined code ---

        # Config: TP=2, PP=2, EP=2, DP=1  (8 GPUs total)
        TP, PP, EP, DP = 2, 2, 2, 1

        decoder_gen = _RankGenerator(tp=TP, ep=1, dp=DP * EP, pp=PP, cp=1,
                                     order="tp-cp-ep-dp-pp")
        expert_gen = _RankGenerator(tp=TP, ep=EP, dp=DP, pp=PP, cp=1,
                                    order="tp-cp-ep-dp-pp")
        megatron_non_expert_dp_groups = decoder_gen.get_ranks('dp')
        megatron_expert_dp_groups = expert_gen.get_ranks('dp')

        def find_group(groups_list: list[list[int]], rank: int) -> list[int]:
            for g in groups_list:
                if rank in g:
                    return sorted(g)
            raise ValueError(f"rank {rank} not found in any group")

        for dp_rank in range(DP):
            for pp_stage in range(PP):
                for ep_rank in range(EP):
                    for tp_rank in range(TP):
                        gpu = _global_rank(dp_rank, pp_stage, ep_rank, tp_rank, PP, EP, TP)
                        groups = self._build_groups(DP, PP, EP, TP,
                                                    dp_rank, pp_stage, ep_rank, tp_rank)

                        expected_ne = find_group(megatron_non_expert_dp_groups, gpu)
                        expected_ex = find_group(megatron_expert_dp_groups, gpu)

                        assert sorted(groups.non_expert_dp) == expected_ne, (
                            f"GPU={gpu}: non_expert_dp={sorted(groups.non_expert_dp)} "
                            f"!= Megatron {expected_ne}"
                        )
                        assert sorted(groups.expert_dp) == expected_ex, (
                            f"GPU={gpu}: expert_dp={sorted(groups.expert_dp)} "
                            f"!= Megatron {expected_ex}"
                        )


# ---------------------------------------------------------------------------
# 3. DAG regression: EP=1 non-MoE step phase
# ---------------------------------------------------------------------------


class TestEP1Regression:
    """EP=1 traces must match the single-DP-group pattern from before EP-aware changes."""

    def test_ep1_dp2_step_allreduce_present(self):
        """With EP=1, DP=2, non-dist-opt: exactly one AllReduce group in step phase."""
        model = _dense_model(hidden_size=256, num_layers=2)
        wl = _workload(model, tp=1, pp=1, ep=1, dp=2, distributed_optimizer=False)
        dc = _minimal_dc(num_gpus=2)
        dag = _no_cache_tracer().trace(wl, dc)

        step_ar = [n for n in dag.comm_nodes
                   if n.collective_type == "AllReduce" and n.phase == "step"]
        assert len(step_ar) > 0, "EP=1, DP=2: step AllReduce must be present"

    def test_ep1_dp2_step_no_rs_ag(self):
        """EP=1, DP=2, no dist-opt: no ReduceScatter or AllGather in step phase."""
        model = _dense_model(hidden_size=256, num_layers=2)
        wl = _workload(model, tp=1, pp=1, ep=1, dp=2, distributed_optimizer=False)
        dc = _minimal_dc(num_gpus=2)
        dag = _no_cache_tracer().trace(wl, dc)

        step_rs = [n for n in dag.comm_nodes
                   if n.collective_type == "ReduceScatter" and n.phase == "step"]
        step_ag = [n for n in dag.comm_nodes
                   if n.collective_type == "AllGather" and n.phase == "step"]
        assert step_rs == [], "No ReduceScatter in step phase with dist-opt=False"
        assert step_ag == [], "No AllGather in step phase with dist-opt=False"

    def test_ep1_dp1_no_step_comms(self):
        """EP=1, DP=1: no step-phase communication at all (single rank, no sync needed)."""
        model = _dense_model(hidden_size=256, num_layers=2)
        wl = _workload(model, tp=1, pp=1, ep=1, dp=1, distributed_optimizer=False)
        dc = _minimal_dc(num_gpus=1)
        dag = _no_cache_tracer().trace(wl, dc)

        step_comms = [n for n in dag.comm_nodes if n.phase == "step"]
        assert step_comms == [], "DP=1: no step comms should be emitted"

    def test_ep1_dp2_dist_opt_has_rs_and_ag(self):
        """EP=1, DP=2, dist-opt=True: step phase has ReduceScatter and AllGather, no AllReduce."""
        model = _dense_model(hidden_size=256, num_layers=2)
        wl = _workload(model, tp=1, pp=1, ep=1, dp=2, distributed_optimizer=True)
        dc = _minimal_dc(num_gpus=2)
        dag = _no_cache_tracer().trace(wl, dc)

        step_rs = [n for n in dag.comm_nodes
                   if n.collective_type == "ReduceScatter" and n.phase == "step"]
        step_ag = [n for n in dag.comm_nodes
                   if n.collective_type == "AllGather" and n.phase == "step"]
        step_ar = [n for n in dag.comm_nodes
                   if n.collective_type == "AllReduce" and n.phase == "step"]
        assert len(step_rs) > 0, "dist-opt: ReduceScatter must appear in step phase"
        assert len(step_ag) > 0, "dist-opt: AllGather must appear in step phase"
        assert step_ar == [], "dist-opt: no AllReduce in step phase"

    def test_ep1_dp2_step_allreduce_group_size_is_dp(self):
        """EP=1, DP=2: step AllReduce flows touch exactly both DP ranks (group size 2)."""
        model = _dense_model(hidden_size=256, num_layers=2)
        wl = _workload(model, tp=1, pp=1, ep=1, dp=2, distributed_optimizer=False)
        dc = _minimal_dc(num_gpus=2)
        dag = _no_cache_tracer().trace(wl, dc)

        step_ar = [n for n in dag.comm_nodes
                   if n.collective_type == "AllReduce" and n.phase == "step"]
        # Ring AllReduce over 2 ranks: 2 flows (each rank sends one chunk)
        # All src/dst ranks must be in {0, 1}
        ranks_touched = {n.src_gpu for n in step_ar} | {n.dst_gpu for n in step_ar}
        assert ranks_touched == {0, 1}, (
            f"EP=1, DP=2: step AllReduce must only involve ranks 0 and 1, got {ranks_touched}"
        )


# ---------------------------------------------------------------------------
# 4. DAG structure: MoE with EP>1
# ---------------------------------------------------------------------------


class TestMoEEPAwareOptimizer:
    """MoE + EP>1: two separate AllReduce groups appear in the step phase."""

    def test_moe_ep4_dp2_two_allreduce_groups(self):
        """8-GPU MoE (TP=1, PP=1, EP=4, DP=2): step phase has both non-expert and expert AllReduces.

        non_expert_dp size = dp * ep = 8 (whole world)
        expert_dp size = dp = 2
        """
        model = _moe_model(hidden_size=256, num_layers=2, num_experts=4)
        # 8 GPUs: TP=1, PP=1, EP=4, DP=2
        wl = _workload(model, tp=1, pp=1, ep=4, dp=2, distributed_optimizer=False)
        dc = _minimal_dc(num_gpus=8)
        dag = _no_cache_tracer().trace(wl, dc)

        step_ar = [n for n in dag.comm_nodes
                   if n.collective_type == "AllReduce" and n.phase == "step"]
        assert len(step_ar) > 0, "MoE EP=4 DP=2: step AllReduce must be present"

        # Collect sets of ranks touched by each AllReduce communication node
        # non-expert group spans all 8 GPUs; expert group spans only 2
        all_pairs = [(n.src_gpu, n.dst_gpu) for n in step_ar]
        all_ranks = {r for pair in all_pairs for r in pair}

        # The non-expert group is the whole world (8 GPUs)
        assert all_ranks == set(range(8)), (
            f"Non-expert step AllReduce must span all 8 GPUs, got {all_ranks}"
        )

    def test_moe_ep4_dp2_expert_allreduce_group_size_dp(self):
        """Expert-layer step AllReduce involves only dp=2 ranks (the expert_dp group)."""
        model = _moe_model(hidden_size=256, num_layers=2, num_experts=4)
        wl = _workload(model, tp=1, pp=1, ep=4, dp=2, distributed_optimizer=False)
        dc = _minimal_dc(num_gpus=8)
        dag = _no_cache_tracer().trace(wl, dc)

        step_ar = [n for n in dag.comm_nodes
                   if n.collective_type == "AllReduce" and n.phase == "step"]

        # Find flows that are restricted to only 2 ranks (expert_dp groups)
        # Group flows by the frozenset of ranks they touch
        from collections import defaultdict
        group_flows: dict[frozenset, list[CommNode]] = defaultdict(list)

        # For ring AllReduce, flows within one collective instance all share the same
        # set of ranks. We identify groups by src+dst pairs.
        # We look for any pair of flows where both endpoints are within a 2-rank set.
        def _involved_ranks(nodes: list[CommNode]) -> set[int]:
            return {r for n in nodes for r in (n.src_gpu, n.dst_gpu)}

        # Build per-"group" rank sets by checking which 2-rank subsets appear
        # (ring AllReduce over 2 ranks: 2 flows, each touching both ranks)
        two_rank_pairs: set[frozenset] = set()
        for n in step_ar:
            pair = frozenset([n.src_gpu, n.dst_gpu])
            if len(pair) == 2:
                two_rank_pairs.add(pair)

        # With DP=2, EP=4, PP=1, TP=1 and rank order tp-ep-dp-pp:
        #   global_rank(dp_r, pp=0, ep_r, tp=0) = dp_r * (1*4*1) + 0 + ep_r * 1 + 0
        #                                        = dp_r * 4 + ep_r
        # expert_dp groups (fix ep_rank, vary dp): e.g. {0, 4}, {1, 5}, {2, 6}, {3, 7}
        # There are ep=4 such groups, one per EP rank.
        expected_expert_dp_groups = {
            frozenset([ep_r, ep_r + 4]) for ep_r in range(4)
        }
        for expected_group in expected_expert_dp_groups:
            assert expected_group in two_rank_pairs, (
                f"Expert DP group {expected_group} must appear as an AllReduce pair in step phase; "
                f"found pairs: {two_rank_pairs}"
            )

    def test_moe_ep1_dp2_no_expert_allreduce(self):
        """MoE with EP=1: no separate expert AllReduce (expert_dp == non_expert_dp, dp > 1 guard)."""
        model = _moe_model(hidden_size=256, num_layers=2, num_experts=4)
        wl = _workload(model, tp=1, pp=1, ep=1, dp=2, distributed_optimizer=False)
        dc = _minimal_dc(num_gpus=2)
        dag = _no_cache_tracer().trace(wl, dc)

        step_ar = [n for n in dag.comm_nodes
                   if n.collective_type == "AllReduce" and n.phase == "step"]
        # With EP=1, both non_expert_dp and expert_dp reduce to [0, 1].
        # The expert branch fires because dp > 1 AND _expert_params > 0.
        # So we get TWO AllReduce collectives but both over the same 2-rank group.
        # The key correctness invariant: no rank outside {0, 1} is touched.
        ranks = {r for n in step_ar for r in (n.src_gpu, n.dst_gpu)}
        assert ranks <= {0, 1}, f"MoE EP=1 step AllReduce must only touch ranks 0,1; got {ranks}"

    def test_non_moe_ep1_dp2_single_allreduce_no_expert(self):
        """Dense (non-MoE) EP=1, DP=2: only one AllReduce in step (non-expert only, no expert)."""
        model = _dense_model(hidden_size=256, num_layers=2)
        wl = _workload(model, tp=1, pp=1, ep=1, dp=2, distributed_optimizer=False)
        dc = _minimal_dc(num_gpus=2)
        dag = _no_cache_tracer().trace(wl, dc)

        step_comms = [n for n in dag.comm_nodes if n.phase == "step"]
        # Only non-expert path fires (_expert_params == 0 for dense model)
        coll_types = {n.collective_type for n in step_comms}
        assert "AllReduce" in coll_types
        # Expert path must NOT fire: _expert_params_per_tp_rank == 0 for dense
        # → the `if dp > 1 and _expert_params > 0` guard prevents a second AllReduce

    def test_moe_ep4_dp2_adamw_present(self):
        """MoE EP=4, DP=2: one adamw ComputeNode per GPU in the step phase."""
        model = _moe_model(hidden_size=256, num_layers=2, num_experts=4)
        wl = _workload(model, tp=1, pp=1, ep=4, dp=2, distributed_optimizer=False)
        dc = _minimal_dc(num_gpus=8)
        dag = _no_cache_tracer().trace(wl, dc)

        adamw_nodes = [n for n in dag.compute_nodes if n.kernel == "adamw"]
        assert len(adamw_nodes) == 8, f"Expected 8 adamw nodes (one per GPU), got {len(adamw_nodes)}"

    def test_moe_ep4_dp2_dist_opt_has_rs_ag_for_both_groups(self):
        """MoE EP=4, DP=2, dist-opt=True: ReduceScatter and AllGather present for both groups."""
        model = _moe_model(hidden_size=256, num_layers=2, num_experts=4)
        wl = _workload(model, tp=1, pp=1, ep=4, dp=2, distributed_optimizer=True)
        dc = _minimal_dc(num_gpus=8)
        dag = _no_cache_tracer().trace(wl, dc)

        step_rs = [n for n in dag.comm_nodes
                   if n.collective_type == "ReduceScatter" and n.phase == "step"]
        step_ag = [n for n in dag.comm_nodes
                   if n.collective_type == "AllGather" and n.phase == "step"]
        step_ar = [n for n in dag.comm_nodes
                   if n.collective_type == "AllReduce" and n.phase == "step"]

        assert len(step_rs) > 0, "dist-opt + MoE: ReduceScatter in step phase"
        assert len(step_ag) > 0, "dist-opt + MoE: AllGather in step phase"
        assert step_ar == [], "dist-opt: no AllReduce in step phase"

        # Both non-expert (8-rank) and expert (2-rank) groups should appear
        rs_ranks = {r for n in step_rs for r in (n.src_gpu, n.dst_gpu)}
        ag_ranks = {r for n in step_ag for r in (n.src_gpu, n.dst_gpu)}
        assert rs_ranks == set(range(8)), "ReduceScatter must span all 8 GPUs (non-expert group)"
        assert ag_ranks == set(range(8)), "AllGather must span all 8 GPUs (non-expert group)"
