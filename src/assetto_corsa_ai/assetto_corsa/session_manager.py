"""Assetto Corsa single-player session management.

This module prepares session files and CSP prerequisites before launching AC.

Responsibilities:
- configure the requested car and track
- configure deterministic driving assists
- write Assetto Corsa's session configuration
- start and stop ``acs.exe``
- verify that the requested session was loaded
- optionally wait until the simulator is actually in the driving state

Runtime vehicle control, teleporting and episode resets belong to
``simulator_controller.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from configparser import ConfigParser
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from types import TracebackType
import os
import math
import subprocess
import time
from typing import Protocol


class SessionManagerError(RuntimeError):
    """Raised when an Assetto Corsa session cannot be prepared or started."""


class AssistMode(IntEnum):
    """Assetto Corsa assist mode."""

    OFF = 0
    FACTORY = 1
    ON = 2


@dataclass(frozen=True)
class SimulatorScenario:
    """Configuration that requires an Assetto Corsa session change."""

    car: str
    track: str
    track_layout: str | None = None
    skin: str | None = None

    abs_mode: AssistMode = AssistMode.FACTORY
    traction_control_mode: AssistMode = AssistMode.FACTORY


class StaticInfoReaderLike(Protocol):
    """Minimal StaticInfoReader interface required by the session manager."""

    def read(self) -> object:
        ...


class AssettoCorsaSessionManager:
    r"""Configure and manage local Assetto Corsa training sessions.

    The session manager owns everything that happens before normal RL runtime
    control begins:

    - select car and track
    - configure assists
    - prepare ``race.ini``
    - start Assetto Corsa
    - wait for the correct session
    - optionally wait until the simulator has entered the actual driving state

    ``drive_ready_check`` is deliberately supplied from outside this class.
    This keeps the session manager independent of the telemetry and CSP
    implementations.

    Example:

        session_manager = AssettoCorsaSessionManager(
            assetto_corsa_directory=Path(
                r"C:\Program Files (x86)\Steam\steamapps\common\assettocorsa"
            ),
            static_info_reader=static_info_reader,
            drive_ready_check=is_driving,
        )

        session_manager.start_session(
            SimulatorScenario(
                car="bmw_z4_gt3",
                track="monza",
            )
        )

    After ``start_session()`` returns, the requested AC session is loaded.
    If a ``drive_ready_check`` was provided, the method only returns once that
    callback confirms that AC is actually in the driving state.
    """

    def __init__(
        self,
        assetto_corsa_directory: Path,
        *,
        documents_directory: Path | None = None,
        static_info_reader: StaticInfoReaderLike | None = None,
        drive_ready_check: Callable[[], bool] | None = None,
        startup_timeout: float = 45.0,
        drive_ready_timeout: float = 15.0,
        poll_interval: float = 0.1,
    ) -> None:
        if not math.isfinite(startup_timeout) or startup_timeout <= 0.0:
            raise ValueError(
                "startup_timeout must be positive."
            )

        if not math.isfinite(drive_ready_timeout) or drive_ready_timeout <= 0.0:
            raise ValueError(
                "drive_ready_timeout must be positive."
            )

        if not math.isfinite(poll_interval) or poll_interval <= 0.0:
            raise ValueError(
                "poll_interval must be positive."
            )

        self._assetto_corsa_directory = Path(
            assetto_corsa_directory
        )

        self._documents_directory = (
            Path(documents_directory)
            if documents_directory is not None
            else Path.home()
            / "Documents"
            / "Assetto Corsa"
        )

        self._race_ini = (
            self._documents_directory
            / "cfg"
            / "race.ini"
        )

        self._assists_ini = (
            self._documents_directory
            / "cfg"
            / "assists.ini"
        )

        self._acs_executable = (
            self._assetto_corsa_directory
            / "acs.exe"
        )

        self._cars_directory = (
            self._assetto_corsa_directory
            / "content"
            / "cars"
        )

        self._tracks_directory = (
            self._assetto_corsa_directory
            / "content"
            / "tracks"
        )

        self._static_info_reader = (
            static_info_reader
        )

        self._drive_ready_check = (
            drive_ready_check
        )

        self._startup_timeout = (
            startup_timeout
        )

        self._drive_ready_timeout = (
            drive_ready_timeout
        )

        self._poll_interval = (
            poll_interval
        )

        self._process: subprocess.Popen[bytes] | None = None
        self._current_scenario: SimulatorScenario | None = None

    @property
    def current_scenario(
        self,
    ) -> SimulatorScenario | None:
        """Return the scenario started by this manager."""

        return self._current_scenario

    @property
    def is_running(
        self,
    ) -> bool:
        """Return True if the AC process started by this manager is alive."""

        return (
            self._process is not None
            and self._process.poll() is None
        )

    def start_session(
        self,
        scenario: SimulatorScenario,
    ) -> None:
        """Configure and start a complete Assetto Corsa training session.

        The method does not return until:

        1. the requested car and track have been loaded, and
        2. if configured, ``drive_ready_check`` confirms that driving is active.

        Raises:
            SessionManagerError:
                If the requested session cannot be started or does not become
                ready before the configured timeout.
        """

        if self.is_running:
            raise SessionManagerError(
                "Assetto Corsa is already running. "
                "Use switch_session() or ensure_session()."
            )

        self._validate_installation()
        self._validate_scenario(
            scenario
        )
        self._prepare_csp_track(
            scenario
        )
        self._prepare_csp_configuration()


        skin = self._resolve_skin(
            car=scenario.car,
            requested_skin=scenario.skin,
        )

        self._write_race_configuration(
            scenario=scenario,
            skin=skin,
        )

        self._write_assist_configuration(
            scenario=scenario,
        )

        self._process = (
            self._launch_assetto_corsa()
        )

        try:
            if self._static_info_reader is not None:
                self._wait_until_loaded(
                    scenario
                )

            if self._drive_ready_check is not None:
                self._wait_until_driving()

        except BaseException:
            self.stop_session()
            raise

        self._current_scenario = (
            scenario
        )

    def switch_session(
        self,
        scenario: SimulatorScenario,
    ) -> None:
        """Stop the current AC session and start another scenario."""

        if (
            self.is_running
            and self._current_scenario == scenario
        ):
            return

        self.stop_session()

        self.start_session(
            scenario
        )

    def ensure_session(
        self,
        scenario: SimulatorScenario,
    ) -> bool:
        """Ensure that the requested scenario is running.

        Returns:
            True:
                Assetto Corsa had to be started or switched.

            False:
                The requested scenario was already running.

        This allows the training environment to avoid restarting AC between
        ordinary episodes that use the same car and track.
        """

        if (
            self.is_running
            and self._current_scenario == scenario
        ):
            return False

        self.switch_session(
            scenario
        )

        return True

    def stop_session(
        self,
        *,
        timeout: float = 5.0,
    ) -> None:
        """Stop the Assetto Corsa process started by this manager."""

        if timeout <= 0.0:
            raise ValueError(
                "timeout must be positive."
            )

        process = self._process

        self._process = None
        self._current_scenario = None

        if process is None:
            return

        if process.poll() is not None:
            return

        process.terminate()

        try:
            process.wait(
                timeout=timeout
            )

        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def _validate_installation(
        self,
    ) -> None:
        """Verify the required Assetto Corsa installation paths."""

        if os.name != "nt":
            raise SessionManagerError(
                "Assetto Corsa session management is currently "
                "implemented for Windows only."
            )

        if not self._assetto_corsa_directory.is_dir():
            raise SessionManagerError(
                "Assetto Corsa directory does not exist: "
                f"{self._assetto_corsa_directory}"
            )

        if not self._acs_executable.is_file():
            raise SessionManagerError(
                "acs.exe was not found: "
                f"{self._acs_executable}"
            )

        if not self._cars_directory.is_dir():
            raise SessionManagerError(
                "Assetto Corsa cars directory was not found: "
                f"{self._cars_directory}"
            )

        if not self._tracks_directory.is_dir():
            raise SessionManagerError(
                "Assetto Corsa tracks directory was not found: "
                f"{self._tracks_directory}"
            )

        self._race_ini.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _validate_scenario(
        self,
        scenario: SimulatorScenario,
    ) -> None:
        """Verify that the requested car, track and layout exist."""

        if not scenario.car.strip():
            raise ValueError(
                "scenario.car must not be empty."
            )

        if not scenario.track.strip():
            raise ValueError(
                "scenario.track must not be empty."
            )

        car_directory = (
            self._cars_directory
            / scenario.car
        )

        if not car_directory.is_dir():
            raise SessionManagerError(
                f"Car '{scenario.car}' was not found at "
                f"{car_directory}."
            )

        track_directory = (
            self._tracks_directory
            / scenario.track
        )

        if not track_directory.is_dir():
            raise SessionManagerError(
                f"Track '{scenario.track}' was not found at "
                f"{track_directory}."
            )

        if scenario.track_layout:
            self._validate_track_layout(
                track_directory=track_directory,
                layout=scenario.track_layout,
            )

    @staticmethod
    def _validate_track_layout(
        *,
        track_directory: Path,
        layout: str,
    ) -> None:
        """Verify that a requested layout exists."""

        layout_directory = (
            track_directory
            / layout
        )

        models_file = (
            track_directory
            / f"models_{layout}.ini"
        )

        if (
            not layout_directory.is_dir()
            and not models_file.is_file()
        ):
            raise SessionManagerError(
                f"Track layout '{layout}' was not found "
                f"for '{track_directory.name}'."
            )

    def _resolve_skin(
        self,
        *,
        car: str,
        requested_skin: str | None,
    ) -> str:
        """Return a valid skin for the requested car."""

        skins_directory = (
            self._cars_directory
            / car
            / "skins"
        )

        if requested_skin is not None:
            skin_directory = (
                skins_directory
                / requested_skin
            )

            if not skin_directory.is_dir():
                raise SessionManagerError(
                    f"Skin '{requested_skin}' was not found "
                    f"for car '{car}'."
                )

            return requested_skin

        if not skins_directory.is_dir():
            return "-"

        available_skins = sorted(
            path.name
            for path in skins_directory.iterdir()
            if path.is_dir()
        )

        return (
            available_skins[0]
            if available_skins
            else "-"
        )

    def _write_race_configuration(
        self,
        *,
        scenario: SimulatorScenario,
        skin: str,
    ) -> None:
        """Configure a single-car Hotlap session.

        Unrelated entries already present in ``race.ini`` are preserved.
        """

        config = self._read_ini(
            self._race_ini
        )

        self._ensure_section(
            config,
            "RACE",
        )

        self._ensure_section(
            config,
            "CAR_0",
        )

        self._ensure_section(
            config,
            "SESSION_0",
        )
        self._ensure_section(config, "REMOTE")
        self._ensure_section(config, "REPLAY")
        config["REMOTE"]["ACTIVE"] = "0"
        config["REPLAY"]["ACTIVE"] = "0"

        race = config["RACE"]

        race["TRACK"] = (
            scenario.track
        )

        race["CONFIG_TRACK"] = (
            scenario.track_layout
            or ""
        )

        race["MODEL"] = (
            scenario.car
        )

        race["MODEL_CONFIG"] = ""
        race["CARS"] = "1"

        # Training should not be interrupted by game penalties.
        race["PENALTIES"] = "0"
        race["JUMP_START_PENALTY"] = "0"

        car = config["CAR_0"]

        car["MODEL"] = (
            scenario.car
        )

        car["MODEL_CONFIG"] = ""
        car["SKIN"] = skin

        # A Hotlap session is a convenient base for RL:
        # one player car, track loaded immediately, no race grid required.
        session = config["SESSION_0"]

        session["NAME"] = "Hotlap"
        session["TYPE"] = "4"  # AC Hotlap; 3 selects a race.
        session["DURATION_MINUTES"] = "0"
        # "HOTLAP" is not a spawn set: AC finds no start point and the car falls.
        session["SPAWN_SET"] = "HOTLAP_START"

        self._write_ini_atomic(
            self._race_ini,
            config,
        )

    def _write_assist_configuration(
        self,
        *,
        scenario: SimulatorScenario,
    ) -> None:
        """Configure deterministic driving assists."""

        config = self._read_ini(
            self._assists_ini
        )

        self._ensure_section(
            config,
            "ASSISTS",
        )

        assists = config["ASSISTS"]

        assists["ABS"] = str(
            int(
                scenario.abs_mode
            )
        )

        assists["TRACTION_CONTROL"] = str(
            int(
                scenario.traction_control_mode
            )
        )

        # Driver aids that should not interfere with the RL policy.
        assists["IDEAL_LINE"] = "0"
        assists["STABILITY_CONTROL"] = "0"
        assists["AUTO_BRAKE"] = "0"

        # CSP Custom AI owns drivetrain automation during training.
        # Keeping AC's normal automatic aids disabled avoids two different
        # systems trying to control the gearbox and clutch at the same time.
        assists["AUTO_SHIFTER"] = "0"
        assists["AUTO_CLUTCH"] = "0"
        assists["AUTO_BLIP"] = "0"

        self._write_ini_atomic(
            self._assists_ini,
            config,
        )

    def _launch_assetto_corsa(
        self,
    ) -> subprocess.Popen[bytes]:
        """Launch Assetto Corsa using the prepared session files."""

        try:
            return subprocess.Popen(
                [
                    str(
                        self._acs_executable
                    )
                ],
                cwd=str(
                    self._assetto_corsa_directory
                ),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        except OSError as error:
            raise SessionManagerError(
                "Failed to start Assetto Corsa."
            ) from error

    def _wait_until_loaded(
        self,
        scenario: SimulatorScenario,
    ) -> None:
        """Wait until shared memory reports the requested car and track."""

        assert (
            self._static_info_reader
            is not None
        )

        deadline = (
            time.monotonic()
            + self._startup_timeout
        )

        last_car: str | None = None
        last_track: str | None = None
        last_layout: str | None = None

        while (
            time.monotonic()
            < deadline
        ):
            self._raise_if_process_exited()

            try:
                static_info = (
                    self._static_info_reader.read()
                )

                last_car = getattr(
                    static_info,
                    "car_model",
                    None,
                )

                last_track = getattr(
                    static_info,
                    "track_name",
                    None,
                )
                last_layout = getattr(static_info, "track_configuration", "")

                if (
                    last_car == scenario.car
                    and last_track == scenario.track
                    and last_layout == (scenario.track_layout or "")
                ):
                    return

            except (
                FileNotFoundError,
                OSError,
                ValueError,
            ):
                # AC shared memory can be unavailable while the game starts.
                pass

            time.sleep(
                self._poll_interval
            )

        raise SessionManagerError(
            "Timed out waiting for Assetto Corsa session. "
            f"Expected car='{scenario.car}', "
            f"track='{scenario.track}', "
            f"last reported car='{last_car}', "
            f"track='{last_track}', layout='{last_layout}'."
        )

    def _wait_until_driving(
        self,
    ) -> None:
        """Wait until the supplied readiness callback confirms driving state.

        This is deliberately separate from ``_wait_until_loaded`` because AC
        can have the correct car and track loaded while still showing its
        pre-drive setup screen.

        The callback should return True only when the simulator is ready for
        RL runtime control.
        """

        assert (
            self._drive_ready_check
            is not None
        )

        deadline = (
            time.monotonic()
            + self._drive_ready_timeout
        )

        while (
            time.monotonic()
            < deadline
        ):
            self._raise_if_process_exited()

            try:
                if self._drive_ready_check():
                    return

            except (
                FileNotFoundError,
                OSError,
                ValueError,
            ):
                # Runtime interfaces can still be initializing.
                pass

            time.sleep(
                self._poll_interval
            )

        raise SessionManagerError(
            "Assetto Corsa loaded the session but did not enter the "
            "driving state before the timeout. If your installation shows "
            "the pre-drive 'Drive / Setup / Exit' screen, configure the "
            "launcher to start the race immediately."
        )

    def _raise_if_process_exited(
        self,
    ) -> None:
        """Raise if the managed AC process exited unexpectedly."""

        if (
            self._process is not None
            and self._process.poll() is not None
        ):
            raise SessionManagerError(
                "Assetto Corsa exited before the session became ready."
            )

    @staticmethod
    def _read_ini(
        path: Path,
    ) -> ConfigParser:
        """Read an INI file while preserving key case."""

        config = ConfigParser(
            interpolation=None,
            strict=False,
        )

        config.optionxform = lambda optionstr: optionstr

        if path.is_file():
            config.read(
                path,
                encoding="utf-8-sig",
            )

        return config

    @staticmethod
    def _ensure_section(
        config: ConfigParser,
        section: str,
    ) -> None:
        """Create an INI section if it does not yet exist."""

        if not config.has_section(
            section
        ):
            config.add_section(
                section
            )

    @staticmethod
    def _write_ini_atomic(
        path: Path,
        config: ConfigParser,
    ) -> None:
        """Atomically replace an Assetto Corsa INI file."""

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        # Keep the original configuration for restoring a manual setup later.
        backup = path.with_suffix(path.suffix + ".before-rl.bak")
        if path.is_file() and not backup.exists():
            with backup.open("xb") as file:
                file.write(path.read_bytes())

        temporary_path = (
            path.with_suffix(
                path.suffix
                + ".tmp"
            )
        )

        with temporary_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as file:
            config.write(
                file,
                space_around_delimiters=False,
            )

        os.replace(
            temporary_path,
            path,
        )


    def _prepare_csp_track(
        self,
        scenario: SimulatorScenario,
    ) -> None:
        """Enable Custom AI manipulation for the selected track."""

        track_directory = self._tracks_directory / scenario.track
        layout_directory = track_directory / scenario.track_layout if scenario.track_layout else track_directory
        surfaces_ini = layout_directory / "data" / "surfaces.ini"
        if not surfaces_ini.is_file() and scenario.track_layout:
            surfaces_ini = track_directory / "data" / "surfaces.ini"
        if not surfaces_ini.is_file():
            raise SessionManagerError(
                f"No unpacked track surfaces.ini found for '{scenario.track}'. "
                "Unpack the track data before enabling CSP Custom AI."
            )

        config = self._read_ini(
            surfaces_ini
        )
        if not config.has_section("SURFACE_0"):
            raise SessionManagerError(f"Missing SURFACE_0 in {surfaces_ini}.")
        pitch = config["SURFACE_0"].get("WAV_PITCH", "0").strip()
        if not pitch.startswith("extended-"):
            config["SURFACE_0"]["WAV_PITCH"] = "extended-" + pitch

        self._ensure_section(
            config,
            "_EXTRA_PERMISSIONS",
        )

        config["_EXTRA_PERMISSIONS"][
            "ALLOW_CUSTOM_AI_MANIPULATION"
        ] = "1"

        self._write_ini_atomic(
            surfaces_ini,
            config,
        )

    def _prepare_csp_configuration(self) -> None:
        """Enable the CSP New Behaviour module and Custom AI before launch."""

        path = self._assetto_corsa_directory / "extension" / "config" / "new_behaviour.ini"
        if not path.is_file():
            raise SessionManagerError(f"CSP configuration not found: {path}")
        config = self._read_ini(path)
        changed = False
        for section in ("BASIC", "CUSTOM_AI"):
            self._ensure_section(config, section)
            value = config[section].get("ENABLED", "0").split(";", 1)[0].strip()
            if value != "1":
                config[section]["ENABLED"] = "1"
                changed = True
        if changed:
            self._write_ini_atomic(path, config)

    def __enter__(
        self,
    ) -> "AssettoCorsaSessionManager":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop_session()
