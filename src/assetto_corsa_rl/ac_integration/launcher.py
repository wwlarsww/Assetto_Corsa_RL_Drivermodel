"""Safe local Assetto Corsa session launcher for explicit live-training runs."""

from __future__ import annotations

import configparser
import ctypes
import logging
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Self

from assetto_corsa_rl.ac_integration.shared_memory import (
    AssettoCorsaSharedMemoryReader,
    AssettoCorsaTelemetry,
    AssettoCorsaTelemetryUnavailableError,
)
from assetto_corsa_rl.config import ProjectConfig

LOGGER = logging.getLogger("assetto_corsa_rl")
ASSETTO_CORSA_LIVE_STATUS = 2
ASSETTO_CORSA_STEAM_APP_ID = "244210"
CONFIG_FILE_RETRY_COUNT = 100
CONFIG_FILE_RETRY_DELAY_S = 0.02


class AssettoCorsaLaunchError(RuntimeError):
    """Raised when a requested local session cannot be prepared or verified."""


class _ProcessHandle(Protocol):
    pid: int

    def poll(self) -> int | None:
        """Return an exit code after termination, otherwise ``None``."""

        ...


class _CasePreservingConfigParser(configparser.ConfigParser):
    def optionxform(self, optionstr: str) -> str:
        return optionstr


@dataclass(frozen=True, slots=True)
class AssettoCorsaSessionSpec:
    """Exact offline session identity written to Assetto Corsa's ``race.ini``."""

    car_id: str
    track_id: str
    track: str
    track_configuration: str
    skin: str


def split_track_identifier(track_id: str) -> tuple[str, str]:
    """Split the repository's ``track-layout`` identity for ``race.ini``."""

    track, separator, configuration = track_id.partition("-")
    if not track:
        raise ValueError("track_id must not be empty")
    return track, configuration if separator else ""


def configured_content_paths(config: ProjectConfig) -> tuple[Path, Path]:
    """Return configured car and track-layout paths in the AC installation."""

    installation = Path(config.assetto_corsa.installation_directory)
    track, configuration = split_track_identifier(config.dataset.track_id)
    car_path = installation / "content" / "cars" / config.dataset.car_id
    track_path = installation / "content" / "tracks" / track
    if configuration:
        track_path = track_path / "ui" / configuration
    return car_path, track_path


def configured_track_model_path(config: ProjectConfig) -> Path:
    """Return the native model manifest required by the configured layout."""

    installation = Path(config.assetto_corsa.installation_directory)
    track, configuration = split_track_identifier(config.dataset.track_id)
    filename = f"models_{configuration}.ini" if configuration else "models.ini"
    return installation / "content" / "tracks" / track / filename


def configured_fast_lane_path(config: ProjectConfig) -> Path:
    """Return the AI spline used by the configured track layout."""

    installation = Path(config.assetto_corsa.installation_directory)
    track, configuration = split_track_identifier(config.dataset.track_id)
    track_path = installation / "content" / "tracks" / track
    if configuration:
        track_path /= configuration
    return track_path / "ai" / "fast_lane.ai"


class AssettoCorsaSessionLauncher:
    """Prepare, launch and verify one configured local Hotlap session.

    The user's existing ``race.ini`` is moved to a recovery backup before the
    generated configuration is installed. It is restored exactly when the
    launcher context exits. A surviving backup after a hard process crash is
    never overwritten automatically.
    """

    def __init__(
        self,
        config: ProjectConfig,
        *,
        process_factory: Callable[[Path, Path], _ProcessHandle] | None = None,
        process_close_request: Callable[[_ProcessHandle], bool] | None = None,
        telemetry_probe: Callable[[], AssettoCorsaTelemetry | None] | None = None,
        reuse_existing_session: bool = True,
        controller_index: int | None = None,
        monotonic_clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._installation = Path(config.assetto_corsa.installation_directory)
        self._executable = self._installation / "acs.exe"
        self._steam_app_id_path = self._installation / "steam_appid.txt"
        self._steam_app_id_temporary_path = self._installation / "steam_appid.txt.acrl-tmp"
        self._race_path = config.assetto_corsa.controls_path.with_name("race.ini")
        self._backup_path = self._race_path.with_name(f"{self._race_path.name}.acrl-backup")
        self._temporary_path = self._race_path.with_name(f"{self._race_path.name}.acrl-tmp")
        self._controls_path = config.assetto_corsa.controls_path
        self._controls_backup_path = self._controls_path.with_name(
            f"{self._controls_path.name}.acrl-backup"
        )
        self._controls_temporary_path = self._controls_path.with_name(
            f"{self._controls_path.name}.acrl-tmp"
        )
        self._assists_path = self._controls_path.with_name("assists.ini")
        self._assists_backup_path = self._assists_path.with_name(
            f"{self._assists_path.name}.acrl-backup"
        )
        self._assists_temporary_path = self._assists_path.with_name(
            f"{self._assists_path.name}.acrl-tmp"
        )
        self._process_factory = process_factory or self._start_process
        self._process_close_request = process_close_request or self._request_graceful_close
        self._telemetry_probe = telemetry_probe or self._read_live_telemetry
        self._reuse_existing_session = reuse_existing_session
        if controller_index is not None and not 0 <= controller_index <= 3:
            raise ValueError("controller_index must lie between 0 and 3")
        self._controller_index = controller_index
        self._monotonic_clock = monotonic_clock
        self._sleep = sleep
        self._had_original = False
        self._configuration_installed = False
        self._steam_app_id_created = False
        self._controls_configuration_installed = False
        self._assists_had_original = False
        self._assists_configuration_installed = False
        self._process: _ProcessHandle | None = None
        self._session_spec: AssettoCorsaSessionSpec | None = None
        self._reused_existing_session = False

    def __enter__(self) -> Self:
        self.launch()
        return self

    def __exit__(self, *_args: object) -> None:
        self.restore_user_configuration()

    def launch(self) -> None:
        """Install a temporary Hotlap configuration and wait for live telemetry."""

        current = self._telemetry_probe()
        if current is not None:
            if self._matches_expected_session(current) and self._reuse_existing_session:
                self._reused_existing_session = True
                LOGGER.info("Reusing the already running matching Assetto Corsa session")
                return
            if self._matches_expected_session(current):
                raise AssettoCorsaLaunchError(
                    "Assetto Corsa is already publishing the matching session, but this "
                    "operation requires a newly launched session"
                )
            raise AssettoCorsaLaunchError(
                "Assetto Corsa is already publishing a different or non-live session; "
                "close it before using --launch-game"
            )

        self._session_spec = self._build_session_spec()
        self._install_race_configuration(self._session_spec)
        try:
            self._install_controls_configuration()
            self._install_assists_configuration()
            self._install_steam_app_id_file()
            self._process = self._process_factory(self._executable, self._installation)
            LOGGER.info(
                "Started Assetto Corsa: car=%s track=%s",
                self._session_spec.car_id,
                self._session_spec.track_id,
            )
            self._wait_for_expected_session()
        except BaseException:
            self.restore_user_configuration()
            raise

    @property
    def session_origin(self) -> str:
        """Return whether the verified session was attached or launched here."""

        if self._reused_existing_session:
            return "attached"
        if self._process is not None:
            return "launched"
        return "not_started"

    def stop_owned_session(self, *, timeout_s: float = 15.0) -> bool:
        """Gracefully close only the AC process created by this launcher."""

        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        process = self._process
        if process is None:
            return False
        if process.poll() is not None:
            self._process = None
            return True
        if not self._process_close_request(process):
            raise AssettoCorsaLaunchError(
                f"could not find a closeable Assetto Corsa window for process {process.pid}"
            )
        deadline = self._monotonic_clock() + timeout_s
        while self._monotonic_clock() < deadline:
            if process.poll() is not None:
                self._process = None
                return True
            self._sleep(min(0.05, deadline - self._monotonic_clock()))
        raise AssettoCorsaLaunchError(
            f"Assetto Corsa process {process.pid} did not stop after the graceful close request"
        )

    def restore_user_configuration(self) -> None:
        """Restore the exact pre-launch launcher-file state idempotently."""

        try:
            try:
                if self._configuration_installed:
                    self._unlink_with_retry(self._temporary_path)
                    if self._had_original:
                        if not self._backup_path.is_file():
                            raise AssettoCorsaLaunchError(
                                "cannot restore missing race configuration backup: "
                                f"{self._backup_path}"
                            )
                        self._replace_with_retry(self._backup_path, self._race_path)
                    else:
                        self._unlink_with_retry(self._race_path)
                    self._configuration_installed = False
            finally:
                try:
                    self._restore_controls_configuration()
                finally:
                    self._restore_assists_configuration()
        finally:
            self._restore_steam_app_id_file()

    def _install_controls_configuration(self) -> None:
        """Temporarily point AC's X360 profile at the created ViGEm device."""

        if self._controller_index is None:
            return
        if self._controls_backup_path.exists():
            raise AssettoCorsaLaunchError(
                f"controller recovery backup already exists: {self._controls_backup_path}"
            )
        if not self._controls_path.is_file():
            raise AssettoCorsaLaunchError(
                f"controls configuration is missing: {self._controls_path}"
            )
        parser = _CasePreservingConfigParser(interpolation=None, strict=False)
        try:
            with self._controls_path.open(encoding="utf-8-sig") as handle:
                parser.read_file(handle)
        except (configparser.Error, OSError, UnicodeError) as error:
            raise AssettoCorsaLaunchError(
                f"cannot read controls configuration {self._controls_path}: {error}"
            ) from error
        if not parser.has_section("X360"):
            parser.add_section("X360")
        parser.set("X360", "JOYPAD_INDEX", str(self._controller_index))
        self._replace_with_retry(self._controls_path, self._controls_backup_path)
        try:
            with self._controls_temporary_path.open(
                "w",
                encoding="utf-8",
                newline="\r\n",
            ) as handle:
                parser.write(handle, space_around_delimiters=False)
            self._replace_with_retry(self._controls_temporary_path, self._controls_path)
            self._controls_configuration_installed = True
        except BaseException:
            self._unlink_with_retry(self._controls_temporary_path)
            if self._controls_backup_path.is_file():
                self._replace_with_retry(self._controls_backup_path, self._controls_path)
            raise

    def _restore_controls_configuration(self) -> None:
        self._unlink_with_retry(self._controls_temporary_path)
        if not self._controls_configuration_installed:
            return
        if not self._controls_backup_path.is_file():
            raise AssettoCorsaLaunchError(
                "cannot restore missing controls configuration backup: "
                f"{self._controls_backup_path}"
            )
        self._replace_with_retry(self._controls_backup_path, self._controls_path)
        self._controls_configuration_installed = False

    def _install_assists_configuration(self) -> None:
        """Temporarily enable AC's automatic gearbox for launched sessions."""

        if not self._config.assetto_corsa.automatic_shifting:
            return
        if self._assists_backup_path.exists():
            raise AssettoCorsaLaunchError(
                f"assists recovery backup already exists: {self._assists_backup_path}"
            )
        parser = _CasePreservingConfigParser(interpolation=None, strict=False)
        self._assists_had_original = self._assists_path.is_file()
        if self._assists_had_original:
            try:
                with self._assists_path.open(encoding="utf-8-sig") as handle:
                    parser.read_file(handle)
            except (configparser.Error, OSError, UnicodeError) as error:
                raise AssettoCorsaLaunchError(
                    f"cannot read assists configuration {self._assists_path}: {error}"
                ) from error
        if not parser.has_section("ASSISTS"):
            parser.add_section("ASSISTS")
        parser.set("ASSISTS", "AUTO_SHIFTER", "1")
        parser.set("ASSISTS", "AUTO_CLUTCH", "1")
        self._assists_path.parent.mkdir(parents=True, exist_ok=True)
        if self._assists_had_original:
            self._replace_with_retry(self._assists_path, self._assists_backup_path)
        try:
            with self._assists_temporary_path.open(
                "w",
                encoding="utf-8",
                newline="\r\n",
            ) as handle:
                parser.write(handle, space_around_delimiters=False)
            self._replace_with_retry(self._assists_temporary_path, self._assists_path)
            self._assists_configuration_installed = True
        except BaseException:
            self._unlink_with_retry(self._assists_temporary_path)
            if self._assists_had_original and self._assists_backup_path.is_file():
                self._replace_with_retry(self._assists_backup_path, self._assists_path)
            raise

    def _restore_assists_configuration(self) -> None:
        self._unlink_with_retry(self._assists_temporary_path)
        if not self._assists_configuration_installed:
            return
        if self._assists_had_original:
            if not self._assists_backup_path.is_file():
                raise AssettoCorsaLaunchError(
                    "cannot restore missing assists configuration backup: "
                    f"{self._assists_backup_path}"
                )
            self._replace_with_retry(self._assists_backup_path, self._assists_path)
        else:
            self._unlink_with_retry(self._assists_path)
        self._assists_configuration_installed = False

    def _install_steam_app_id_file(self) -> None:
        """Expose the Steam identity required when starting ``acs.exe`` directly."""

        if self._steam_app_id_temporary_path.exists():
            raise AssettoCorsaLaunchError(
                f"temporary Steam App ID file already exists: {self._steam_app_id_temporary_path}"
            )
        if self._steam_app_id_path.exists():
            if not self._steam_app_id_path.is_file():
                raise AssettoCorsaLaunchError(
                    f"Steam App ID path is not a file: {self._steam_app_id_path}"
                )
            try:
                app_id = self._steam_app_id_path.read_text(encoding="utf-8-sig").strip()
            except (OSError, UnicodeError) as error:
                raise AssettoCorsaLaunchError(
                    f"cannot read existing Steam App ID file {self._steam_app_id_path}: {error}"
                ) from error
            if app_id != ASSETTO_CORSA_STEAM_APP_ID:
                raise AssettoCorsaLaunchError(
                    f"unexpected Steam App ID {app_id!r} in {self._steam_app_id_path}; "
                    f"expected {ASSETTO_CORSA_STEAM_APP_ID}"
                )
            return
        try:
            self._steam_app_id_temporary_path.write_text(
                ASSETTO_CORSA_STEAM_APP_ID,
                encoding="ascii",
            )
            self._replace_with_retry(
                self._steam_app_id_temporary_path,
                self._steam_app_id_path,
            )
            self._steam_app_id_created = True
        except OSError as error:
            self._unlink_with_retry(self._steam_app_id_temporary_path)
            raise AssettoCorsaLaunchError(
                f"cannot install temporary Steam App ID file {self._steam_app_id_path}: {error}"
            ) from error

    def _restore_steam_app_id_file(self) -> None:
        self._unlink_with_retry(self._steam_app_id_temporary_path)
        if not self._steam_app_id_created:
            return
        if self._steam_app_id_path.is_file():
            try:
                app_id = self._steam_app_id_path.read_text(encoding="utf-8-sig").strip()
            except (OSError, UnicodeError) as error:
                raise AssettoCorsaLaunchError(
                    f"cannot verify temporary Steam App ID file {self._steam_app_id_path}: {error}"
                ) from error
            if app_id != ASSETTO_CORSA_STEAM_APP_ID:
                raise AssettoCorsaLaunchError(
                    "temporary Steam App ID file changed while Assetto Corsa was running; "
                    f"leaving it in place: {self._steam_app_id_path}"
                )
            self._unlink_with_retry(self._steam_app_id_path)
        self._steam_app_id_created = False

    def _build_session_spec(self) -> AssettoCorsaSessionSpec:
        if not self._executable.is_file():
            raise AssettoCorsaLaunchError(
                f"Assetto Corsa executable is missing: {self._executable}"
            )
        car_path, track_path = configured_content_paths(self._config)
        track_model_path = configured_track_model_path(self._config)
        fast_lane_path = configured_fast_lane_path(self._config)
        if not car_path.is_dir():
            raise AssettoCorsaLaunchError(f"configured car is not installed: {car_path}")
        if not track_path.is_dir():
            raise AssettoCorsaLaunchError(f"configured track layout is not installed: {track_path}")
        if not track_model_path.is_file():
            raise AssettoCorsaLaunchError(
                "configured track contains only metadata or is incomplete; "
                f"required model manifest is missing: {track_model_path}"
            )
        if not fast_lane_path.is_file():
            raise AssettoCorsaLaunchError(
                f"configured track layout has no AI spline: {fast_lane_path}"
            )
        skins_path = car_path / "skins"
        skins = (
            sorted(path.name for path in skins_path.iterdir() if path.is_dir())
            if skins_path.is_dir()
            else []
        )
        if not skins:
            raise AssettoCorsaLaunchError(f"configured car has no installed skins: {car_path}")
        track, configuration = split_track_identifier(self._config.dataset.track_id)
        return AssettoCorsaSessionSpec(
            car_id=self._config.dataset.car_id,
            track_id=self._config.dataset.track_id,
            track=track,
            track_configuration=configuration,
            skin=skins[0],
        )

    def _install_race_configuration(self, session: AssettoCorsaSessionSpec) -> None:
        if self._backup_path.exists():
            raise AssettoCorsaLaunchError(
                f"recovery backup already exists: {self._backup_path}; restore or remove it first"
            )
        parser = self._read_race_configuration()
        self._configure_hotlap(parser, session)
        self._race_path.parent.mkdir(parents=True, exist_ok=True)
        self._had_original = self._race_path.is_file()
        if self._had_original:
            self._replace_with_retry(self._race_path, self._backup_path)
        try:
            # AC 1.16.4's native INI reader requires Windows line endings here;
            # LF-only files can corrupt the parsed track identifier at startup.
            with self._temporary_path.open("w", encoding="utf-8", newline="\r\n") as handle:
                parser.write(handle, space_around_delimiters=False)
            self._replace_with_retry(self._temporary_path, self._race_path)
            self._configuration_installed = True
        except BaseException:
            self._unlink_with_retry(self._temporary_path)
            if self._had_original and self._backup_path.is_file():
                self._replace_with_retry(self._backup_path, self._race_path)
            raise

    def _replace_with_retry(self, source: Path, destination: Path) -> None:
        """Atomically replace a config file despite short Windows reader locks."""

        for attempt in range(CONFIG_FILE_RETRY_COUNT):
            try:
                source.replace(destination)
                return
            except PermissionError as error:
                if attempt + 1 >= CONFIG_FILE_RETRY_COUNT:
                    raise AssettoCorsaLaunchError(
                        f"cannot replace {destination} with {source}: {error}"
                    ) from error
                self._sleep(CONFIG_FILE_RETRY_DELAY_S)

    def _unlink_with_retry(self, path: Path) -> None:
        """Remove one temporary config file with bounded Windows-lock retries."""

        for attempt in range(CONFIG_FILE_RETRY_COUNT):
            try:
                path.unlink(missing_ok=True)
                return
            except PermissionError as error:
                if attempt + 1 >= CONFIG_FILE_RETRY_COUNT:
                    raise AssettoCorsaLaunchError(f"cannot remove {path}: {error}") from error
                self._sleep(CONFIG_FILE_RETRY_DELAY_S)

    def _read_race_configuration(self) -> configparser.ConfigParser:
        parser = _CasePreservingConfigParser(interpolation=None, strict=False)
        if self._race_path.is_file():
            try:
                with self._race_path.open(encoding="utf-8-sig") as handle:
                    parser.read_file(handle)
            except (configparser.Error, OSError, UnicodeError) as error:
                raise AssettoCorsaLaunchError(
                    f"cannot read existing race configuration {self._race_path}: {error}"
                ) from error
        return parser

    @staticmethod
    def _configure_hotlap(
        parser: configparser.ConfigParser,
        session: AssettoCorsaSessionSpec,
    ) -> None:
        for section in tuple(parser.sections()):
            if (section.startswith("SESSION_") and section != "SESSION_0") or (
                section.startswith("CAR_") and section != "CAR_0"
            ):
                parser.remove_section(section)
        sections = {
            "RACE": {
                "TRACK": session.track,
                "CONFIG_TRACK": session.track_configuration,
                "MODEL": session.car_id,
                "MODEL_CONFIG": "",
                "CARS": "1",
            },
            "SESSION_0": {
                "NAME": "Hotlap",
                "TYPE": "4",
                "DURATION_MINUTES": "0",
                "SPAWN_SET": "HOTLAP_START",
            },
            "CAR_0": {
                "MODEL": "-",
                "MODEL_CONFIG": "",
                "SKIN": session.skin,
            },
            "REMOTE": {"ACTIVE": "0"},
            "REPLAY": {"ACTIVE": "0"},
            "BENCHMARK": {"ACTIVE": "0"},
            "GHOST_CAR": {"RECORDING": "0", "PLAYING": "0", "LOAD": "0"},
        }
        for section, values in sections.items():
            if not parser.has_section(section):
                parser.add_section(section)
            for key, value in values.items():
                parser.set(section, key, value)

    def _wait_for_expected_session(self) -> None:
        deadline = self._monotonic_clock() + self._config.assetto_corsa.launch_timeout_s
        last_identity = "telemetry unavailable"
        while self._monotonic_clock() < deadline:
            if self._process is not None:
                return_code = self._process.poll()
                if return_code is not None:
                    raise AssettoCorsaLaunchError(
                        f"Assetto Corsa exited before the session became live (code {return_code})"
                    )
            telemetry = self._telemetry_probe()
            if telemetry is not None:
                last_identity = self._describe_session(telemetry)
                if self._matches_expected_session(telemetry):
                    LOGGER.info("Assetto Corsa session is live and matches the training config")
                    return
            self._sleep(self._config.assetto_corsa.launch_poll_interval_s)
        raise AssettoCorsaLaunchError(
            f"timed out waiting for the configured Assetto Corsa session ({last_identity})"
        )

    def _matches_expected_session(self, telemetry: AssettoCorsaTelemetry) -> bool:
        static = telemetry.static
        track_id = static.track
        if static.track_configuration:
            track_id += f"-{static.track_configuration}"
        return (
            telemetry.graphics.status == ASSETTO_CORSA_LIVE_STATUS
            and static.car_model == self._config.dataset.car_id
            and track_id == self._config.dataset.track_id
        )

    @staticmethod
    def _describe_session(telemetry: AssettoCorsaTelemetry) -> str:
        static = telemetry.static
        track_id = static.track
        if static.track_configuration:
            track_id += f"-{static.track_configuration}"
        return f"status={telemetry.graphics.status}, car={static.car_model!r}, track={track_id!r}"

    @staticmethod
    def _read_live_telemetry() -> AssettoCorsaTelemetry | None:
        try:
            with AssettoCorsaSharedMemoryReader.connect() as reader:
                return reader.read()
        except (AssettoCorsaTelemetryUnavailableError, OSError, RuntimeError, ValueError):
            return None

    @staticmethod
    def _start_process(executable: Path, working_directory: Path) -> _ProcessHandle:
        try:
            return subprocess.Popen([str(executable)], cwd=working_directory)  # noqa: S603
        except OSError as error:
            raise AssettoCorsaLaunchError(f"cannot start Assetto Corsa: {error}") from error

    @staticmethod
    def _request_graceful_close(process: _ProcessHandle) -> bool:
        """Post ``WM_CLOSE`` to visible top-level windows owned by ``process``."""

        if sys.platform != "win32":
            return False
        ctypes_api: Any = ctypes
        user32: Any = ctypes_api.WinDLL("user32", use_last_error=True)
        callback_type: Any = ctypes_api.WINFUNCTYPE(
            ctypes_api.c_bool,
            ctypes_api.c_void_p,
            ctypes_api.c_void_p,
        )
        close_requested = False

        @callback_type  # type: ignore[untyped-decorator]
        def close_window(window: int, _parameter: int) -> bool:
            nonlocal close_requested
            owner = ctypes_api.c_ulong()
            user32.GetWindowThreadProcessId(window, ctypes_api.byref(owner))
            if (
                owner.value == process.pid
                and user32.IsWindowVisible(window)
                and user32.PostMessageW(window, 0x0010, 0, 0)  # WM_CLOSE
            ):
                close_requested = True
            return True

        if not user32.EnumWindows(close_window, 0):
            error_code = ctypes_api.get_last_error()
            if error_code:
                raise AssettoCorsaLaunchError(
                    f"could not enumerate windows for Assetto Corsa process: {error_code}"
                )
        return close_requested
