import pytest

from simulon.backend.dag.trace_tracer import (
    ParallelConfig,
    _decompose_rank,
    _get_cp_group_ranks,
    _get_dp_group_ranks,
    _get_ep_group_ranks,
    _get_tp_group_ranks,
    _global_rank,
    _ranks_in_same_cp_group,
    _ranks_in_same_dp_group,
    _ranks_in_same_ep_group,
    _ranks_in_same_tp_group,
)


def _pc(tp=1, cp=1, ep=1, dp=1, pp=1, num_gpus=None):
    if num_gpus is None:
        num_gpus = tp * cp * ep * dp * pp
    return ParallelConfig(tp=tp, cp=cp, ep=ep, dp=dp, pp=pp, num_gpus=num_gpus)


@pytest.mark.parametrize(
    "config",
    [
        _pc(tp=2, cp=1, ep=1, dp=2, pp=4, num_gpus=16),
        _pc(tp=4, pp=1, ep=1, dp=4, num_gpus=16),
        _pc(tp=2, cp=1, ep=2, dp=2, pp=2, num_gpus=16),
        _pc(tp=1, cp=1, ep=1, dp=1, pp=1, num_gpus=1),
    ],
)
def test_round_trip_identity(config):
    for r in range(config.world_size):
        assert _global_rank(*_decompose_rank(r, config), config) == r


def test_dp_group_positive_same_pp_tp_ep_different_dp():
    config = _pc(tp=2, pp=2, ep=2, dp=2, num_gpus=16)
    assert _ranks_in_same_dp_group([0, 4], 0, config) is True


def test_dp_group_negative_different_tp():
    config = _pc(tp=2, pp=2, ep=2, dp=2, num_gpus=16)
    assert _ranks_in_same_dp_group([0, 1], 0, config) is False


def test_dp_group_negative_different_ep():
    config = _pc(tp=2, pp=2, ep=2, dp=2, num_gpus=16)
    assert _ranks_in_same_dp_group([0, 2], 0, config) is False


def test_dp_group_negative_different_pp():
    config = _pc(tp=2, pp=2, ep=2, dp=2, num_gpus=16)
    assert _ranks_in_same_dp_group([0, 8], 0, config) is False


def test_dp_group_another_positive():
    config = _pc(tp=4, pp=1, ep=1, dp=4, num_gpus=16)
    assert _ranks_in_same_dp_group([0, 4, 8, 12], 0, config) is True


def test_dp_group_vacuously_true_for_empty_list():
    config = _pc(tp=2, pp=2, ep=2, dp=2, num_gpus=16)
    assert _ranks_in_same_dp_group([], 0, config) is False


def test_dp_group_single_element():
    config = _pc(tp=2, pp=2, ep=2, dp=2, num_gpus=16)
    assert _ranks_in_same_dp_group([0], 0, config) is False


def test_dp_group_full_group():
    config = _pc(tp=2, pp=2, ep=2, dp=2, num_gpus=16)
    assert _ranks_in_same_dp_group([0, 4], 0, config) is True


def test_dp_group_missing_element():
    config = _pc(tp=4, pp=1, ep=1, dp=4, num_gpus=16)
    assert _ranks_in_same_dp_group([0, 4, 8], 0, config) is False


def test_ep_group_positive_same_tp_cp_dp_pp_different_ep():
    config = _pc(tp=2, pp=1, ep=4, dp=2, num_gpus=16)
    assert _ranks_in_same_ep_group([0, 2, 4, 6], 0, config) is True


def test_ep_group_negative_different_tp():
    config = _pc(tp=2, pp=1, ep=4, dp=2, num_gpus=16)
    assert _ranks_in_same_ep_group([0, 1], 0, config) is False


def test_ep_group_negative_different_dp():
    config = _pc(tp=2, pp=1, ep=4, dp=2, num_gpus=16)
    assert _ranks_in_same_ep_group([0, 8], 0, config) is False


def test_tp_group_positive_same_cp_ep_dp_pp_different_tp():
    config = _pc(tp=4, pp=2, ep=2, dp=2, num_gpus=32)
    assert _ranks_in_same_tp_group([0, 1, 2, 3], 0, config) is True


def test_tp_group_negative_different_ep():
    config = _pc(tp=4, pp=2, ep=2, dp=2, num_gpus=32)
    assert _ranks_in_same_tp_group([0, 4], 0, config) is False


def test_cp_group_single_element():
    config = _pc(tp=2, pp=2, ep=2, dp=2, num_gpus=16)
    assert _ranks_in_same_cp_group([0], 0, config) is True


def test_cp_group_multi_rank_different_ep():
    config = _pc(tp=2, pp=2, ep=2, dp=2, num_gpus=16)
    assert _ranks_in_same_cp_group([0, 2], 0, config) is False


def test_get_dp_group_ranks():
    config = _pc(tp=2, cp=1, ep=1, dp=2, pp=4, num_gpus=16)
    assert _get_dp_group_ranks(0, config) == [0, 2]
    assert _get_dp_group_ranks(1, config) == [1, 3]


def test_get_ep_group_ranks():
    config = _pc(tp=2, pp=1, ep=4, dp=2, num_gpus=16)
    assert _get_ep_group_ranks(0, config) == [0, 2, 4, 6]


def test_get_tp_group_ranks():
    config = _pc(tp=4, pp=2, ep=2, dp=2, num_gpus=32)
    assert _get_tp_group_ranks(0, config) == [0, 1, 2, 3]


def test_get_cp_group_ranks_with_cp_one():
    config = _pc(tp=2, pp=2, ep=2, dp=2, num_gpus=16)
    assert _get_cp_group_ranks(0, config) == [0]


def test_decompose_rank_raises_on_invalid_rank():
    config = _pc(tp=2, dp=2, pp=2, num_gpus=8)
    with pytest.raises(ValueError):
        _decompose_rank(8, config)


def test_global_rank_raises_on_out_of_range_tp():
    config = _pc(tp=2, dp=2, pp=2, num_gpus=8)
    with pytest.raises(ValueError):
        _global_rank(2, 0, 0, 0, 0, config)
