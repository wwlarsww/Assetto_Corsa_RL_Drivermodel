"""Freeze a consistent SB3 pair at update boundaries; write it off-thread."""

from collections import deque
from dataclasses import dataclass
from io import BufferedIOBase, BytesIO
from pathlib import Path
import pickle
from threading import Event, Thread

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecNormalize

from assetto_corsa_ai.training_run import TrainingRun


@dataclass(frozen=True)
class FrozenModel:
    payload: bytes

    def save(self, stream: BufferedIOBase, /) -> None:
        stream.write(self.payload)


@dataclass(frozen=True)
class FrozenNormalizer:
    payload: bytes

    def save(self, path: str, /) -> None:
        with Path(path).open("xb") as stream:
            stream.write(self.payload)


class LatestSnapshotWriter:
    """One pending version. No worker reads mutable model/statistics objects."""
    def __init__(self, run: TrainingRun) -> None:
        self.run = run
        self._pending: deque[tuple[int, int, FrozenModel, FrozenNormalizer]] = deque(maxlen=1)
        self._wake, self._stop = Event(), Event()
        self.error: str | None = None
        self._thread = Thread(target=self._write, name="latest-snapshot", daemon=True)
        self._thread.start()

    def submit(self, model: BaseAlgorithm, normalizer: VecNormalize, updates: int) -> None:
        try:
            stream = BytesIO()
            # Public SB3 serialization freezes weights and optimizer state.
            # It costs a copy at the update boundary, but performs no disk I/O.
            model.save(stream)
            frozen = FrozenModel(stream.getvalue())
            normalization = FrozenNormalizer(pickle.dumps(normalizer, protocol=pickle.HIGHEST_PROTOCOL))
            self._pending.append((model.num_timesteps, updates, frozen, normalization))
            self._wake.set()
        except Exception as error:
            self.error = str(error)

    def _write(self) -> None:
        while True:
            self._wake.wait()
            self._wake.clear()
            try:
                step, updates, model, normalizer = self._pending.pop()
            except IndexError:
                if self._stop.is_set():
                    return
                continue
            try:
                self.run.save_latest(step, updates, model, normalizer)
                self.error = None
            except Exception as error:
                self.error = str(error)
            if self._stop.is_set() and not self._pending:
                return

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=5.0)
