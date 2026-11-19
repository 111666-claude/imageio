"""Tests for imageio.core.animation (the animation time view)"""

import math

import pytest
from imageio.core.animation import AnimationTimeline


def test_timeline_basic():
    timeline = AnimationTimeline([100, 200, 50])

    assert timeline.n_frames == 3
    assert timeline.duration == 350
    assert timeline.delays == [100, 200, 50]
    assert timeline.starts == [0, 100, 300]


def test_timeline_frame_index_at():
    timeline = AnimationTimeline([100, 200, 50])

    # half-open intervals: a boundary belongs to the frame that starts there
    assert timeline.frame_index_at(0) == 0
    assert timeline.frame_index_at(99.9) == 0
    assert timeline.frame_index_at(100) == 1
    assert timeline.frame_index_at(299.9) == 1
    assert timeline.frame_index_at(300) == 2
    assert timeline.frame_index_at(349.9) == 2

    # the exact end of a play-through shows the end state
    assert timeline.frame_index_at(350) == 2


def test_timeline_out_of_range():
    timeline = AnimationTimeline([100, 200, 50])

    with pytest.raises(IndexError):
        timeline.frame_index_at(-0.1)
    with pytest.raises(IndexError):
        timeline.frame_index_at(350.1)


def test_timeline_loop():
    timeline = AnimationTimeline([100, 200, 50])

    assert timeline.frame_index_at(350, loop=True) == 0
    assert timeline.frame_index_at(450, loop=True) == 1
    assert timeline.frame_index_at(700, loop=True) == 0
    assert timeline.frame_index_at(701, loop=True) == 0
    assert timeline.frame_index_at(800, loop=True) == 1

    # negative times wrap around, too
    assert timeline.frame_index_at(-1, loop=True) == 2
    assert timeline.frame_index_at(-350, loop=True) == 0


def test_timeline_zero_delay():
    # zero-delay frames occupy no playback time and are skipped
    timeline = AnimationTimeline([0, 100, 0, 50])

    assert timeline.duration == 150
    assert timeline.frame_index_at(0) == 1
    assert timeline.frame_index_at(99.9) == 1
    assert timeline.frame_index_at(100) == 3
    assert timeline.frame_index_at(150) == 3


def test_timeline_all_zero_delays():
    # a zero-duration animation collapses to its end state
    timeline = AnimationTimeline([0, 0, 0])

    assert timeline.duration == 0
    assert timeline.frame_index_at(0) == 2
    assert timeline.frame_index_at(123, loop=True) == 2

    with pytest.raises(IndexError):
        timeline.frame_index_at(1)


def test_timeline_empty():
    timeline = AnimationTimeline([])

    assert timeline.n_frames == 0
    assert timeline.duration == 0

    with pytest.raises(IndexError):
        timeline.frame_index_at(0)
    with pytest.raises(IndexError):
        timeline.frame_index_at(0, loop=True)


def test_timeline_sanitizes_delays():
    # missing and negative delays are treated as zero
    timeline = AnimationTimeline([None, -5, 100])

    assert timeline.delays == [0.0, 0.0, 100.0]
    assert timeline.duration == 100
    assert timeline.frame_index_at(0) == 2


@pytest.mark.parametrize("bad_delay", [math.nan, math.inf, -math.inf])
def test_timeline_rejects_nonfinite_delays(bad_delay):
    with pytest.raises(ValueError):
        AnimationTimeline([100, bad_delay])


@pytest.mark.parametrize("bad_time", [math.nan, math.inf, -math.inf])
def test_timeline_rejects_nonfinite_times(bad_time):
    timeline = AnimationTimeline([100])

    with pytest.raises(ValueError):
        timeline.frame_index_at(bad_time)
    with pytest.raises(ValueError):
        timeline.frame_index_at(bad_time, loop=True)
