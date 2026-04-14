import os
import sys

import numpy as np

# Allow running as a script from this subfolder:
#   python .\tests_misaligned\test_time_offset_misaligned.py
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from correlation import estimate_time_offset


def _make_misaligned_heads(
    *,
    start_ps: int,
    window_ps: int,
    true_time_diff_ps: int,
    n_events: int,
    drop_initial_ps: int,
    drop_fraction: float,
    noise_events: int,
    seed: int,
):
    """
    Build a synthetic pair (signal_head, idler_head) where:
      - idler is shifted by `true_time_diff_ps` (idler + timeDiff ~= signal)
      - idler misses an initial segment (so first timestamps do not correspond)
      - idler randomly drops additional events and adds some noise events
    This simulates "timestamps not fully corresponding".
    """
    if window_ps <= 0:
        raise ValueError("window_ps must be > 0")
    if n_events <= 0:
        raise ValueError("n_events must be > 0")
    if not (0.0 <= drop_fraction < 1.0):
        raise ValueError("drop_fraction must be in [0, 1)")

    rng = np.random.default_rng(seed)

    # Shared "ground truth" event times in [0, window_ps). Use integers (ps).
    base = rng.integers(0, window_ps, size=n_events, dtype=np.int64)
    base.sort()

    sig = start_ps + base

    # Construct idler such that idler + timeDiff ~= signal.
    idl = sig - int(true_time_diff_ps)

    # Drop early idler events to make the head misaligned (first event differs).
    if drop_initial_ps > 0:
        keep = base >= int(drop_initial_ps)
        idl = idl[keep]

    # Randomly drop a fraction of remaining idler events.
    if drop_fraction > 0 and idl.size > 0:
        keep = rng.random(idl.size) >= drop_fraction
        idl = idl[keep]

    # Add some "noise" idler events that are not present in signal.
    if noise_events > 0:
        noise_base = rng.integers(0, window_ps, size=noise_events, dtype=np.int64)
        noise = start_ps + noise_base - int(true_time_diff_ps)
        idl = np.concatenate([idl, noise])
        idl.sort()

    # Ensure dtype/int sorting contract matches the real code expectations.
    return sig.astype(np.int64), idl.astype(np.int64)


def test_estimate_time_offset_with_partial_overlap_and_noise():
    # Keep n_frames modest so the test is fast, but choose a clean integer bin width.
    # bin_width_ps = window_ps / n_frames
    n_frames = 20_000
    window_ps = n_frames * 1_000  # 1000 ps per bin (integer)
    start_ps = 1_000_000_000_000

    true_time_diff_ps = 123_000  # multiple of 1000 ps to avoid rounding ambiguity

    sig_head, idl_head = _make_misaligned_heads(
        start_ps=start_ps,
        window_ps=window_ps,
        true_time_diff_ps=true_time_diff_ps,
        n_events=200_000,
        drop_initial_ps=2_000_000,  # drop first 2e6 ps worth of idler events
        drop_fraction=0.35,
        noise_events=15_000,
        seed=7,
    )

    est = estimate_time_offset(sig_head, idl_head, window_ps, n_frames)

    # Allow 1-bin tolerance. With the way we choose parameters, it should usually be exact.
    assert abs(est - true_time_diff_ps) <= 1_000


def test_estimate_time_offset_when_idler_starts_earlier_than_signal():
    n_frames = 10_000
    window_ps = n_frames * 2_000  # 2000 ps per bin
    start_ps = 5_000_000_000_000

    # Negative timeDiff means idler + timeDiff ~= signal still holds:
    # If timeDiff=-20000, idler is ahead of signal by 20000 ps.
    true_time_diff_ps = -20_000  # multiple of 2000 ps per bin

    sig_head, idl_head = _make_misaligned_heads(
        start_ps=start_ps,
        window_ps=window_ps,
        true_time_diff_ps=true_time_diff_ps,
        n_events=120_000,
        drop_initial_ps=1_000_000,
        drop_fraction=0.25,
        noise_events=8_000,
        seed=99,
    )

    est = estimate_time_offset(sig_head, idl_head, window_ps, n_frames)
    assert abs(est - true_time_diff_ps) <= 2_000


def _run_as_script():
    # Script-mode runner so you can do: python tests_misaligned/test_time_offset_misaligned.py
    # without requiring pytest.
    test_estimate_time_offset_with_partial_overlap_and_noise()
    test_estimate_time_offset_when_idler_starts_earlier_than_signal()
    print("[OK] misaligned timestamp tests passed")


if __name__ == "__main__":
    _run_as_script()
