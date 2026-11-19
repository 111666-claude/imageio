# -*- coding: utf-8 -*-
# imageio is distributed under the terms of the (new) BSD License.

"""Tools to view animations along a time axis.

This module provides :class:`AnimationTimeline`, a small helper that maps
playback time (in milliseconds) onto frame indices of an animation. Plugins
use it to offer time-based random access to animated ImageResources (e.g.
GIF or APNG files) in addition to the index-based access of the v3 plugin
API.
"""

import math
from bisect import bisect_right
from typing import List, Sequence


class AnimationTimeline:
    """Map playback time onto the frames of an animation.

    Parameters
    ----------
    delays : Sequence[float]
        The display time of each frame in milliseconds. ``None`` and
        negative values are treated as 0.

    Attributes
    ----------
    n_frames : int
        The number of frames in the animation.
    duration : float
        The total duration of one play-through of the animation in
        milliseconds.
    delays : List[float]
        The display time of each frame in milliseconds.
    starts : List[float]
        The playback time (in milliseconds) at which each frame starts.

    Notes
    -----
    Frames partition the interval ``[0, duration]`` into half-open
    intervals ``[start, start + delay)``: a frame is shown from the moment
    the previous frame's display time ends until its own display time ends.

    Frames with zero delay occupy an empty interval. They never become the
    current frame during playback, but they still exist and still matter
    when compositing subsequent frames. If all delays are zero the
    animation has ``duration == 0``; in that case only ``time_ms == 0`` is
    valid and it maps to the last frame, i.e. the end state of the
    animation. The same end state is shown when ``time_ms == duration``
    (the exact end of a single play-through).

    """

    def __init__(self, delays: Sequence[float]) -> None:
        self._starts: List[float] = []
        self._delays: List[float] = []

        start = 0.0
        for delay in delays:
            if delay is None:
                delay = 0.0
            delay = float(delay)
            if not math.isfinite(delay):
                raise ValueError("Frame delays must be finite numbers.")
            delay = max(0.0, delay)
            self._starts.append(start)
            self._delays.append(delay)
            start += delay

        self._duration = start

    @property
    def n_frames(self) -> int:
        """The number of frames in the animation."""
        return len(self._delays)

    @property
    def duration(self) -> float:
        """The duration of one play-through in milliseconds."""
        return self._duration

    @property
    def delays(self) -> List[float]:
        """The display time of each frame in milliseconds."""
        return list(self._delays)

    @property
    def starts(self) -> List[float]:
        """The playback time (in milliseconds) at which each frame starts."""
        return list(self._starts)

    def frame_index_at(self, time_ms: float, *, loop: bool = False) -> int:
        """Return the index of the frame shown at the given playback time.

        Parameters
        ----------
        time_ms : float
            The playback time in milliseconds. Must be a finite number.
            Without looping, valid times lie in ``[0, duration]`` where
            ``duration`` is the total duration of one play-through.
        loop : bool
            If True, the animation loops and any (finite) time is valid:
            times outside one play-through wrap around via modulo. If
            False (default), out-of-range times raise an ``IndexError``.

        Returns
        -------
        index : int
            The index of the frame that is visible at ``time_ms``.

        Raises
        ------
        IndexError
            If the animation contains no frames, or if ``time_ms`` lies
            outside the animation and ``loop`` is False.
        ValueError
            If ``time_ms`` is not a finite number.

        """
        if self.n_frames == 0:
            raise IndexError("The animation contains no frames.")

        time_ms = float(time_ms)
        if not math.isfinite(time_ms):
            raise ValueError("`time_ms` must be a finite number.")

        if self._duration > 0:
            if loop:
                time_ms = math.fmod(time_ms, self._duration)
                if time_ms < 0:
                    time_ms += self._duration
            elif time_ms < 0 or time_ms > self._duration:
                raise IndexError(
                    f"Time {time_ms} ms is outside the animation's duration "
                    f"of {self._duration} ms. Pass `loop=True` to wrap "
                    "around instead."
                )

            if time_ms >= self._duration:
                # the exact end of a single play-through: the end state
                return self.n_frames - 1

            index = bisect_right(self._starts, time_ms) - 1
            while self._starts[index] + self._delays[index] <= time_ms:
                # skip frames with zero display time
                index += 1
            return index

        # a zero-duration animation collapses to its end state
        if loop or time_ms == 0:
            return self.n_frames - 1
        raise IndexError(
            f"The animation has zero duration; only time 0 ms is valid "
            f"(got {time_ms} ms)."
        )
