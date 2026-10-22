"""Bounded history shared by checks and plots."""

from collections import deque

from tools.diagnostics.frame import DiagnosticsFrame


class FrameBuffer:
    def __init__(self, capacity: int = 400, seconds: float = 20.0) -> None:
        if capacity <= 0 or seconds <= 0:
            raise ValueError("History limits must be positive.")
        self._frames: deque[DiagnosticsFrame] = deque(maxlen=capacity)
        self.seconds = seconds

    def append(self, frame: DiagnosticsFrame) -> None:
        if self._frames and frame.timestamp < self._frames[-1].timestamp:
            self._frames.clear()
        if self._frames and frame.timestamp == self._frames[-1].timestamp:
            self._frames[-1] = frame
        else:
            self._frames.append(frame)
        while self._frames and frame.timestamp - self._frames[0].timestamp > self.seconds:
            self._frames.popleft()

    def frames(self) -> tuple[DiagnosticsFrame, ...]:
        return tuple(self._frames)

    def clear(self) -> None:
        self._frames.clear()
