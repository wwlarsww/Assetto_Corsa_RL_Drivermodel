"""Optional, low-overhead desktop dashboard for live PPO diagnostics."""

from __future__ import annotations

import csv
import importlib
import json
import logging
import math
import os
import subprocess
import sys
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

try:
    plt: Any = importlib.import_module("matplotlib.pyplot")
except ImportError:  # pragma: no cover - exercised only with a broken installation
    plt = None

LOGGER = logging.getLogger("assetto_corsa_rl")

_BACKGROUND: Final = "#08111f"
_PANEL: Final = "#101c2e"
_GRID: Final = "#2a3a51"
_TEXT: Final = "#e8eef7"
_MUTED: Final = "#8fa1b8"
_CYAN: Final = "#38bdf8"
_GREEN: Final = "#34d399"
_AMBER: Final = "#fbbf24"
_ORANGE: Final = "#fb923c"
_RED: Final = "#fb7185"
_PURPLE: Final = "#a78bfa"


class TrainingMonitorError(RuntimeError):
    """Raised when a requested standalone training monitor cannot start."""


class _Monitor(Protocol):
    @property
    def is_enabled(self) -> bool: ...

    def update(self, metrics: dict[str, Any]) -> None: ...

    def finish(self, *, completed: bool) -> None: ...

    def close(self) -> None: ...

    def wait_for_close(self) -> None: ...

    def process_events(self) -> None: ...


@dataclass(frozen=True, slots=True)
class TrainingRunDescriptor:
    """Display metadata loaded from one immutable PPO run directory."""

    run_directory: Path
    total_environment_steps: int | None
    run_label: str


@dataclass(frozen=True, slots=True)
class _SeriesSpec:
    key: str
    label: str
    colour: str


@dataclass(frozen=True, slots=True)
class _PanelSpec:
    title: str
    hint: str
    series: tuple[_SeriesSpec, ...]
    non_negative: bool = False


_PANELS: Final = (
    _PanelSpec(
        "Reward signal",
        "rollout mean · higher is better",
        (
            _SeriesSpec("environment_reward_mean", "Environment", _CYAN),
            _SeriesSpec("shaped_reward_mean", "Shaped", _GREEN),
        ),
    ),
    _PanelSpec(
        "Policy loss",
        "clipped actor objective",
        (_SeriesSpec("policy_loss", "Policy", _RED),),
    ),
    _PanelSpec(
        "Value loss",
        "critic regression · lower is better",
        (_SeriesSpec("value_loss", "Value", _AMBER),),
        non_negative=True,
    ),
    _PanelSpec(
        "Predictor loss",
        "auxiliary trajectory objective",
        (_SeriesSpec("predictor_loss", "Predictor", _PURPLE),),
        non_negative=True,
    ),
    _PanelSpec(
        "Policy entropy",
        "exploration · watch for collapse",
        (_SeriesSpec("entropy", "Entropy", _CYAN),),
    ),
    _PanelSpec(
        "Approximate KL",
        "policy update size · spikes matter",
        (_SeriesSpec("approximate_kl", "KL", _RED),),
        non_negative=True,
    ),
    _PanelSpec(
        "Clip fraction",
        "share of clipped probability ratios",
        (_SeriesSpec("clip_fraction", "Clip", _ORANGE),),
        non_negative=True,
    ),
    _PanelSpec(
        "Completed episodes",
        "cumulative collection progress",
        (_SeriesSpec("episodes_completed", "Episodes", _GREEN),),
        non_negative=True,
    ),
)


class LiveMonitor:
    """Render PPO metrics in a responsive, non-blocking Matplotlib dashboard.

    Dashboard failures never stop training: if the Matplotlib GUI is
    unavailable, construction logs one warning and produces a no-op object
    while CSV recording continues unaffected.
    """

    def __init__(
        self,
        *,
        total_environment_steps: int | None = None,
        run_label: str = "PPO",
        max_history_points: int = 5_000,
        refresh_interval_s: float = 0.1,
    ) -> None:
        """Create the live dashboard without blocking the training process."""

        if total_environment_steps is not None and total_environment_steps <= 0:
            raise ValueError("total_environment_steps must be positive")
        if max_history_points < 2:
            raise ValueError("max_history_points must be at least two")
        if refresh_interval_s < 0:
            raise ValueError("refresh_interval_s must be non-negative")

        self._enabled = False
        self._total_environment_steps = total_environment_steps
        self._run_label = run_label
        self._refresh_interval_s = refresh_interval_s
        self._history: dict[str, deque[float]] = {
            series.key: deque(maxlen=max_history_points)
            for panel in _PANELS
            for series in panel.series
        }
        self._steps: dict[str, deque[float]] = {
            series.key: deque(maxlen=max_history_points)
            for panel in _PANELS
            for series in panel.series
        }
        self._lines: dict[str, Any] = {}
        self._axes_by_metric: dict[str, Any] = {}
        self._kpi_text: dict[str, Any] = {}
        self._iteration = 0
        self._started_at = time.perf_counter()
        self._last_render_at = 0.0
        self._last_sample_at: float | None = None
        self._last_environment_steps: float | None = None
        self._steps_per_second: float | None = None
        self._latest_metrics: dict[str, float] = {}
        self._fig: Any = None
        self._header: Any = None
        self._status_text: Any = None
        self._meta_text: Any = None

        if plt is None:
            LOGGER.warning("live dashboard unavailable; Matplotlib is not installed")
            return

        try:
            self._build_figure()
        except Exception as error:  # pragma: no cover - backend and OS dependent
            partial_figure = getattr(self, "_fig", None)
            if partial_figure is not None:
                plt.close(partial_figure)
            LOGGER.warning("live dashboard unavailable: %s", error)
            return
        self._enabled = True

    @property
    def is_enabled(self) -> bool:
        """Return whether a live figure is available and still open."""

        return self._enabled

    def update(self, metrics: dict[str, Any]) -> None:
        """Append one PPO iteration and refresh the visible dashboard if due."""

        if not self._is_open():
            return

        self._iteration += 1
        now = time.perf_counter()
        environment_steps = self._finite_float(metrics.get("environment_steps"))
        if environment_steps is None:
            environment_steps = float(self._iteration)
        self._update_throughput(environment_steps, now)

        for key in self._history:
            value = self._finite_float(metrics.get(key))
            if value is None:
                continue
            self._history[key].append(value)
            self._steps[key].append(environment_steps)
            self._latest_metrics[key] = value

        policy_version = self._finite_float(metrics.get("policy_version"))
        self._latest_metrics["environment_steps"] = environment_steps
        if policy_version is not None:
            self._latest_metrics["policy_version"] = policy_version
        self._update_header()
        self._render(force=now - self._last_render_at >= self._refresh_interval_s)

    def finish(self, *, completed: bool) -> None:
        """Mark the run as completed or stopped and force one final redraw."""

        if not self._is_open():
            return
        status = "COMPLETE" if completed else "STOPPED"
        colour = _GREEN if completed else _AMBER
        self._status_text.set_text(f"●  {status}")
        self._status_text.set_color(colour)
        self._render(force=True)

    def close(self) -> None:
        """Close the dashboard and release its GUI resources."""

        if not self._enabled:
            return
        self._enabled = False
        try:
            plt.close(self._fig)
        except Exception:  # pragma: no cover - defensive GUI cleanup
            LOGGER.debug("live dashboard cleanup failed", exc_info=True)

    def wait_for_close(self) -> None:
        """Keep the final dashboard visible until the user closes its window."""

        if not self._is_open():
            return
        try:
            plt.ioff()
            plt.show(block=True)
        except Exception:  # pragma: no cover - backend and OS dependent
            LOGGER.debug("live dashboard could not wait for close", exc_info=True)
            self.close()

    def process_events(self) -> None:
        """Keep a standalone dashboard responsive between metric polls."""

        if not self._is_open():
            return
        try:
            self._fig.canvas.flush_events()
        except Exception:  # pragma: no cover - defensive GUI event handling
            LOGGER.debug("live dashboard event processing failed", exc_info=True)
            self.close()

    def _build_figure(self) -> None:
        plt.ion()
        self._fig = plt.figure(figsize=(16, 9.5), facecolor=_BACKGROUND, layout="constrained")
        grid = self._fig.add_gridspec(
            3,
            4,
            height_ratios=(0.72, 1.0, 1.0),
            hspace=0.2,
            wspace=0.12,
        )
        self._header = self._fig.add_subplot(grid[0, :])
        self._style_header()

        for index, panel in enumerate(_PANELS):
            row = 1 + index // 4
            column = index % 4
            axis = self._fig.add_subplot(grid[row, column])
            self._style_panel(axis, panel)
            for series in panel.series:
                (line,) = axis.plot(
                    [],
                    [],
                    color=series.colour,
                    label=series.label,
                    linewidth=2.0,
                    solid_capstyle="round",
                )
                self._lines[series.key] = line
                self._axes_by_metric[series.key] = axis
            if len(panel.series) > 1:
                legend = axis.legend(
                    loc="upper left",
                    frameon=False,
                    ncols=len(panel.series),
                    fontsize=8,
                )
                for text in legend.get_texts():
                    text.set_color(_MUTED)

        manager = getattr(self._fig.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title(f"AC-RL Training Dashboard — {self._run_label}")
        self._fig.canvas.mpl_connect("close_event", self._on_close)
        self._fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self._fig.canvas.draw_idle()
        if str(plt.get_backend()).casefold() != "agg":
            plt.show(block=False)
        self._fig.canvas.flush_events()

    def _style_header(self) -> None:
        self._header.set_facecolor(_BACKGROUND)
        self._header.set_axis_off()
        self._header.text(
            0.0,
            0.93,
            "AC-RL  /  TRAINING CONTROL",
            transform=self._header.transAxes,
            color=_TEXT,
            fontsize=17,
            fontweight="bold",
            va="top",
        )
        self._meta_text = self._header.text(
            0.0,
            0.68,
            self._run_label,
            transform=self._header.transAxes,
            color=_MUTED,
            fontsize=9,
            va="top",
        )
        self._status_text = self._header.text(
            1.0,
            0.9,
            "●  LIVE",
            transform=self._header.transAxes,
            color=_GREEN,
            fontsize=10,
            fontweight="bold",
            ha="right",
            va="top",
        )

        cards = (
            ("progress", "PROGRESS"),
            ("reward", "SHAPED REWARD"),
            ("policy_loss", "POLICY LOSS"),
            ("entropy", "ENTROPY"),
            ("throughput", "THROUGHPUT"),
            ("episodes", "EPISODES"),
        )
        card_width = 1.0 / len(cards)
        for index, (key, label) in enumerate(cards):
            x = index * card_width
            if index:
                self._header.plot(
                    [x, x],
                    [0.06, 0.5],
                    transform=self._header.transAxes,
                    color=_GRID,
                    linewidth=0.8,
                )
            self._kpi_text[key] = self._header.text(
                x + 0.012,
                0.36,
                "—",
                transform=self._header.transAxes,
                color=_TEXT,
                fontsize=14,
                fontweight="bold",
                va="center",
            )
            self._header.text(
                x + 0.012,
                0.11,
                label,
                transform=self._header.transAxes,
                color=_MUTED,
                fontsize=7.5,
                fontweight="bold",
                va="center",
            )

    @staticmethod
    def _style_panel(axis: Any, panel: _PanelSpec) -> None:
        axis.set_facecolor(_PANEL)
        axis.set_title(
            panel.title,
            loc="left",
            color=_TEXT,
            fontsize=11,
            fontweight="bold",
            pad=21,
        )
        axis.text(
            0.0,
            1.025,
            panel.hint,
            transform=axis.transAxes,
            color=_MUTED,
            fontsize=7.5,
            va="bottom",
        )
        axis.set_xlabel("environment steps", color=_MUTED, fontsize=8, labelpad=5)
        axis.xaxis.set_major_formatter(
            lambda value, _position: LiveMonitor._compact_number(float(value))
        )
        axis.locator_params(axis="x", nbins=5)
        axis.grid(axis="y", color=_GRID, linewidth=0.7, alpha=0.65)
        axis.tick_params(axis="both", colors=_MUTED, labelsize=8, length=0)
        axis.margins(x=0.025, y=0.14)
        for spine in axis.spines.values():
            spine.set_color(_GRID)
            spine.set_linewidth(0.8)

    def _update_header(self) -> None:
        steps = self._latest_metrics.get("environment_steps", 0.0)
        version = int(self._latest_metrics.get("policy_version", 0.0))
        elapsed = max(0.0, time.perf_counter() - self._started_at)
        self._meta_text.set_text(
            f"{self._run_label}   ·   iteration {self._iteration:,}   ·   "
            f"policy v{version:,}   ·   elapsed {self._format_duration(elapsed)}"
        )

        if self._total_environment_steps is None:
            progress = self._compact_number(steps)
        else:
            percent = min(100.0, 100.0 * steps / self._total_environment_steps)
            progress = f"{percent:.1f}%"
        self._kpi_text["progress"].set_text(progress)
        self._kpi_text["reward"].set_text(
            self._format_metric(self._latest_metrics.get("shaped_reward_mean"))
        )
        self._kpi_text["policy_loss"].set_text(
            self._format_metric(self._latest_metrics.get("policy_loss"))
        )
        self._kpi_text["entropy"].set_text(self._format_metric(self._latest_metrics.get("entropy")))
        throughput = (
            f"{self._compact_number(self._steps_per_second)} steps/s"
            if self._steps_per_second is not None
            else "warming up"
        )
        self._kpi_text["throughput"].set_text(throughput)
        episodes = self._latest_metrics.get("episodes_completed")
        self._kpi_text["episodes"].set_text(f"{int(episodes):,}" if episodes is not None else "—")

    def _render(self, *, force: bool) -> None:
        if not force:
            return
        for key, line in self._lines.items():
            line.set_data(tuple(self._steps[key]), tuple(self._history[key]))

        for panel in _PANELS:
            axis = self._axes_by_metric[panel.series[0].key]
            values = [value for series in panel.series for value in self._history[series.key]]
            steps = [step for series in panel.series for step in self._steps[series.key]]
            if not values or not steps:
                continue
            self._set_axis_limits(axis, values, steps, non_negative=panel.non_negative)

        self._last_render_at = time.perf_counter()
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    @staticmethod
    def _set_axis_limits(
        axis: Any,
        values: list[float],
        steps: list[float],
        *,
        non_negative: bool,
    ) -> None:
        x_min, x_max = min(steps), max(steps)
        x_padding = max((x_max - x_min) * 0.025, 1.0)
        axis.set_xlim(x_min - x_padding, x_max + x_padding)

        y_min, y_max = min(values), max(values)
        y_span = y_max - y_min
        y_padding = max(abs(y_max) * 0.08, 0.05) if y_span == 0 else y_span * 0.1
        lower = 0.0 if non_negative and y_min >= 0 else y_min - y_padding
        upper = y_max + y_padding
        if upper <= lower:
            upper = lower + 0.1
        axis.set_ylim(lower, upper)

    def _update_throughput(self, environment_steps: float, now: float) -> None:
        if self._last_sample_at is not None and self._last_environment_steps is not None:
            elapsed = now - self._last_sample_at
            step_delta = environment_steps - self._last_environment_steps
            if elapsed > 0 and step_delta >= 0:
                current = step_delta / elapsed
                if self._steps_per_second is None:
                    self._steps_per_second = current
                else:
                    self._steps_per_second = 0.25 * current + 0.75 * self._steps_per_second
        self._last_sample_at = now
        self._last_environment_steps = environment_steps

    def _is_open(self) -> bool:
        if not self._enabled:
            return False
        if not plt.fignum_exists(self._fig.number):
            self._enabled = False
            return False
        return True

    def _on_close(self, _event: Any) -> None:
        self._enabled = False

    def _on_key_press(self, event: Any) -> None:
        if event.key in {"escape", "q"}:
            self.close()

    @staticmethod
    def _finite_float(value: Any) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    @staticmethod
    def _compact_number(value: float | None) -> str:
        if value is None:
            return "—"
        absolute = abs(value)
        if absolute >= 1_000_000:
            return f"{value / 1_000_000:.2f}m"
        if absolute >= 1_000:
            return f"{value / 1_000:.1f}k"
        if absolute >= 100:
            return f"{value:.0f}"
        if absolute >= 10:
            return f"{value:.1f}"
        return f"{value:.2f}"

    @staticmethod
    def _format_metric(value: float | None) -> str:
        if value is None:
            return "—"
        absolute = abs(value)
        if absolute != 0 and absolute < 0.0001:
            return f"{value:.2e}"
        if absolute >= 100:
            return f"{value:,.1f}"
        if absolute >= 1:
            return f"{value:.3f}"
        return f"{value:.4f}"

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total_seconds = int(seconds)
        hours, remainder = divmod(total_seconds, 3_600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"


def resolve_training_run_directory(
    run_directory: Path | None,
    *,
    runs_root: Path = Path("runs"),
) -> Path:
    """Resolve an explicit PPO run or select the newest recorded PPO run."""

    if run_directory is not None:
        selected = run_directory.parent if run_directory.is_file() else run_directory
        if not selected.is_dir():
            raise TrainingMonitorError(f"training run directory does not exist: {selected}")
        return selected.resolve()

    if not runs_root.is_dir():
        raise TrainingMonitorError(f"training runs root does not exist: {runs_root}")
    candidates = [
        metadata.parent
        for metadata in runs_root.rglob("run.json")
        if metadata.parent.parent.name.startswith("ppo-")
    ]
    if not candidates:
        raise TrainingMonitorError(f"no PPO training run found below {runs_root}")
    return max(
        candidates,
        key=lambda path: (path / "run.json").stat().st_mtime_ns,
    ).resolve()


def describe_training_run(run_directory: Path) -> TrainingRunDescriptor:
    """Load a dashboard label and target step count from ``run.json``."""

    metadata_path = run_directory / "run.json"
    total_steps: int | None = None
    backend = run_directory.parent.name.removeprefix("ppo-").upper() or "PPO"
    method_variant = "unknown variant"
    if metadata_path.is_file():
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            configuration = payload["configuration"]
            total_steps = int(configuration["ppo"]["total_environment_steps"])
            method_variant = str(configuration["ppo"]["method_variant"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError, UnicodeError):
            LOGGER.warning("could not read complete monitor metadata from %s", metadata_path)
    if total_steps is not None and total_steps <= 0:
        total_steps = None
    return TrainingRunDescriptor(
        run_directory=run_directory,
        total_environment_steps=total_steps,
        run_label=f"{backend} · {method_variant} · {run_directory.name}",
    )


def read_training_metrics(metric_path: Path) -> list[dict[str, str]]:
    """Read every complete metrics row currently visible to another process."""

    if not metric_path.is_file():
        return []
    try:
        with metric_path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except (csv.Error, OSError, UnicodeError):
        LOGGER.debug("training metrics are temporarily unreadable: %s", metric_path)
        return []


def monitor_training_run(
    run_directory: Path,
    *,
    poll_interval_s: float = 0.25,
    parent_pid: int | None = None,
    monitor_factory: Callable[..., _Monitor] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    parent_alive: Callable[[int], bool] | None = None,
) -> None:
    """Display and follow one run's append-only ``metrics.csv`` file."""

    if poll_interval_s <= 0:
        raise ValueError("poll_interval_s must be positive")
    if parent_pid is not None and parent_pid <= 0:
        raise ValueError("parent_pid must be positive")
    descriptor = describe_training_run(run_directory)
    factory = monitor_factory or LiveMonitor
    monitor = factory(
        total_environment_steps=descriptor.total_environment_steps,
        run_label=descriptor.run_label,
    )
    if not monitor.is_enabled:
        raise TrainingMonitorError(
            "training dashboard is unavailable; verify the Matplotlib GUI installation"
        )
    metric_path = run_directory / "metrics.csv"
    summary_path = run_directory / "summary.json"
    processed_rows = 0
    is_parent_alive = parent_alive or _process_exists
    try:
        while monitor.is_enabled:
            rows = read_training_metrics(metric_path)
            if len(rows) < processed_rows:
                processed_rows = 0
            for row in rows[processed_rows:]:
                monitor.update(row)
            processed_rows = len(rows)

            if summary_path.is_file():
                monitor.finish(completed=True)
                monitor.wait_for_close()
                return
            if parent_pid is not None and not is_parent_alive(parent_pid):
                monitor.finish(completed=False)
                monitor.wait_for_close()
                return
            monitor.process_events()
            sleep(poll_interval_s)
    finally:
        monitor.close()


def launch_training_monitor(
    run_directory: Path,
    *,
    poll_interval_s: float = 0.25,
    process_factory: Callable[..., Any] | None = None,
) -> int | None:
    """Start an independent GUI process following one training run."""

    if poll_interval_s <= 0:
        raise ValueError("poll_interval_s must be positive")
    if plt is None:
        LOGGER.warning("training dashboard unavailable; Matplotlib is not installed")
        return None
    executable = Path(sys.executable)
    if sys.platform == "win32":
        pythonw = executable.with_name("pythonw.exe")
        if pythonw.is_file():
            executable = pythonw
    command = [
        str(executable),
        "-m",
        "assetto_corsa_rl",
        "training-monitor",
        "--run",
        str(run_directory.resolve()),
        "--poll-interval",
        str(poll_interval_s),
        "--parent-pid",
        str(os.getpid()),
    ]
    factory = process_factory or subprocess.Popen
    options: dict[str, Any] = {
        "cwd": Path.cwd(),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if sys.platform == "win32":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True
    try:
        process = factory(command, **options)
    except OSError as error:
        LOGGER.warning("could not launch training dashboard: %s", error)
        return None
    pid = int(process.pid)
    LOGGER.info("Opened independent training dashboard for %s (PID %s)", run_directory, pid)
    return pid


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True
