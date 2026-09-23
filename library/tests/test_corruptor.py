import numpy as np
import pandas as pd
import pytest

from tsadquality import Corruptor
from tsadquality.corruption import CORRUPTIONS

SETTINGS = {
    "noise": {"snr_db": 10},
    "spikes": {"fraction": 0.05, "multiplier": 5},
    "stuck": {"fraction": 0.1, "stuck_blocks": 2},
    "point_swap": {"fraction": 0.2},
    "segment_swap": {"fraction": 0.2, "num_swaps": 2},
    "permutation": {"num_permutations": 4},
    "point_missing": {"fraction": 0.05, "mechanism": "MNAR_extreme"},
    "burst_missing": {"fraction": 0.05, "num_bursts": 2},
    "gilbert_elliott": {"a": 0.05, "b": 0.3},
}


@pytest.fixture
def df():
    t = np.arange(1000)
    labels = np.zeros(t.size, dtype=int)
    labels[500:520] = 1
    return pd.DataFrame(
        {"value": np.sin(2 * np.pi * t / 50) + t / 500, "label": labels, "other": "x"},
        index=pd.date_range("2024-01-01", periods=t.size, freq="min"),
    )


def test_settings_cover_every_corruption():
    assert set(SETTINGS) == set(CORRUPTIONS)


@pytest.mark.parametrize("kind", SETTINGS)
def test_each_corruption_changes_only_the_value_column(df, kind):
    original = df.copy()
    corrupted, mask = Corruptor(seed=1).add(kind, **SETTINGS[kind]).corrupt(
        df, value_col="value", label_col="label", return_mask=True
    )

    pd.testing.assert_frame_equal(df, original)  # input untouched
    assert corrupted.index.equals(df.index)
    pd.testing.assert_series_equal(corrupted["label"], df["label"])
    pd.testing.assert_series_equal(corrupted["other"], df["other"])
    changed = ~np.isclose(corrupted["value"], df["value"], equal_nan=False)
    assert changed.any()
    assert mask.index.equals(df.index) and mask.any()


def test_same_seed_same_output_on_every_call(df):
    corruptor = Corruptor(seed=7).add("noise", snr_db=5).add("point_swap", fraction=0.1)
    first = corruptor.corrupt(df)
    pd.testing.assert_frame_equal(first, corruptor.corrupt(df))
    pd.testing.assert_frame_equal(first, Corruptor(seed=7).add("noise", snr_db=5).add("point_swap", fraction=0.1).corrupt(df))
    assert not first.equals(Corruptor(seed=8).add("noise", snr_db=5).add("point_swap", fraction=0.1).corrupt(df))


def test_global_numpy_random_state_is_untouched(df):
    np.random.seed(123)
    expected = np.random.random(3)
    np.random.seed(123)
    Corruptor(seed=1).add("segment_swap", fraction=0.2, num_swaps=5).add("noise", snr_db=10).corrupt(df)
    assert np.array_equal(np.random.random(3), expected)


@pytest.mark.parametrize("seed", range(5))
def test_swaps_deliver_the_requested_fraction_across_the_series(seed):
    n = 18000
    df = pd.DataFrame({"value": np.arange(n, dtype=float)})

    out = Corruptor(seed=seed).add("point_swap", fraction=0.2).corrupt(df)["value"].to_numpy()
    moved = np.flatnonzero(out != np.arange(n))
    assert len(moved) == int(0.2 * n)
    # Swapped points should travel ~n/3 on average, not to their immediate neighbour.
    assert np.median(np.abs(out[moved] - moved)) > n / 10

    for fraction, num_swaps in [(0.3, 1), (0.1, 5), (0.01, 20)]:
        out = Corruptor(seed=seed).add("segment_swap", fraction=fraction, num_swaps=num_swaps).corrupt(df)["value"].to_numpy()
        swap_length = int(fraction * n / (2 * num_swaps))
        assert (out != np.arange(n)).sum() == 2 * num_swaps * swap_length


def test_steps_run_in_order(df):
    noise_then_missing = Corruptor(seed=3).add("noise", snr_db=10).add("point_missing", fraction=0.1).corrupt(df)
    assert noise_then_missing["value"].isna().sum() == 100


def test_works_without_labels(df):
    corrupted = Corruptor(seed=1).add("spikes", fraction=0.05, multiplier=5).corrupt(df[["value"]])
    assert list(corrupted.columns) == ["value"]


def test_integer_values_are_corrupted_as_floats():
    df = pd.DataFrame({"value": np.arange(200)})
    corrupted = Corruptor(seed=1).add("noise", snr_db=0).corrupt(df)
    assert corrupted["value"].dtype.kind == "f"


@pytest.mark.parametrize(
    "call, error",
    [
        (lambda c: c.add("blur", sigma=1), ValueError),
        (lambda c: c.add("noise"), TypeError),
        (lambda c: c.add("noise", snr_db=10, fraction=0.1), TypeError),
        (lambda c: c.add("point_swap", fraction=0.1, num_swaps=3), TypeError),
        (lambda c: c.add("noise", snr_db=10, target="somewhere"), ValueError),
        (lambda c: c.add("point_missing", fraction=0.1, mechanism="MAR"), ValueError),
    ],
)
def test_invalid_configuration_fails_early(call, error):
    with pytest.raises(error):
        call(Corruptor())


def test_corrupt_validates_input(df):
    with pytest.raises(ValueError, match="add"):
        Corruptor().corrupt(df)
    with pytest.raises(KeyError):
        Corruptor().add("noise", snr_db=10).corrupt(df, value_col="missing")
    with pytest.raises(ValueError, match="label_col"):
        Corruptor().add("noise", snr_db=10, target="near_anomaly").corrupt(df)


def test_targets_restrict_where_corruption_happens(df):
    _, mask = Corruptor(seed=1).add("noise", snr_db=0, target="only_normal").corrupt(
        df, label_col="label", return_mask=True
    )
    assert not mask[df["label"] == 1].any()
    assert mask[df["label"] == 0].all()


def test_repr_lists_steps():
    corruptor = Corruptor(seed=42).add("noise", snr_db=10).add("spikes", fraction=0.01, multiplier=5)
    assert repr(corruptor) == "Corruptor(seed=42, steps=[noise(snr_db=10), spikes(fraction=0.01, multiplier=5)])"
