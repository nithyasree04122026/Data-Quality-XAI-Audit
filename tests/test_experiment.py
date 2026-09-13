import numpy as np
import pytest

from dq_xai.experiment import Config, corrupt_training, rank_correlation, run, split_data


def test_splits_are_disjoint_in_size_and_stratified():
    train, val, test = split_data(42)
    assert sum(len(y) for _, y in (train, val, test)) == 569
    assert all(0.55 < y.mean() < 0.7 for _, y in (train, val, test))


def test_corruption_does_not_mutate_inputs_and_is_deterministic():
    x, y = split_data(42)[0]
    original_x, original_y = x.copy(), y.copy()
    cfg = Config(seed=42, missing_rate=0.2, label_flip_rate=0.1)
    a = corrupt_training(x, y, cfg)
    b = corrupt_training(x, y, cfg)
    np.testing.assert_array_equal(x, original_x)
    np.testing.assert_array_equal(y, original_y)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert a[2] > 0 and a[3] == round(0.1 * len(y))


def test_clean_and_damaged_runs_are_reproducible():
    cfg = Config(seed=42, missing_rate=0.2, label_flip_rate=0.1, repeats=2)
    assert run(cfg) == run(cfg)
    result = run(cfg)
    assert 0 <= result["test_accuracy"] <= 1
    assert 0 <= result["test_roc_auc"] <= 1
    assert -1 <= result["validation_explanation_rank_correlation"] <= 1


def test_invalid_rates_rejected():
    with pytest.raises(ValueError):
        Config(missing_rate=-0.1).validate()
    with pytest.raises(ValueError):
        Config(label_flip_rate=1.1).validate()


def test_rank_correlation_identity():
    a = np.array([0.1, 0.2, 0.3])
    assert rank_correlation(a, a) == pytest.approx(1.0)
