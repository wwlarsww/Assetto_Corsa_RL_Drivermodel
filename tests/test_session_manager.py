"""Test session preparation in a fake installation without launching AC."""

from configparser import ConfigParser
from itertools import count
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock, create_autospec

import pytest

from assetto_corsa_ai.assetto_corsa import session_manager as module
from assetto_corsa_ai.assetto_corsa.session_manager import (
    AssettoCorsaSessionManager,
    SessionManagerError,
    SimulatorScenario,
)


@pytest.fixture
def installation(tmp_path: Path) -> tuple[Path, Path]:
    game = tmp_path / "game"
    documents = tmp_path / "documents"
    (game / "content/cars/bmw_z4_gt3/skins/default").mkdir(parents=True)
    (game / "content/tracks/monza/data").mkdir(parents=True)
    (game / "extension/config").mkdir(parents=True)
    (documents / "cfg").mkdir(parents=True)
    (game / "acs.exe").touch()
    (game / "content/tracks/monza/data/surfaces.ini").write_text(
        "[SURFACE_0]\nKEY=ROAD\nWAV_PITCH=0\nFRICTION=0.95\n", encoding="utf-8",
    )
    (game / "extension/config/new_behaviour.ini").write_text(
        "[BASIC]\nENABLED=0\n[CUSTOM_AI]\nENABLED=0\n[OTHER]\nKEEP=42\n", encoding="utf-8",
    )
    (documents / "cfg/race.ini").write_text(
        "[RACE]\nTRACK=old\n[REMOTE]\nACTIVE=1\n[REPLAY]\nACTIVE=1\n[OTHER]\nKEEP=42\n",
        encoding="utf-8",
    )
    return game, documents


@pytest.fixture
def processes(monkeypatch: pytest.MonkeyPatch) -> list[Mock]:
    launched: list[Mock] = []
    process_type = subprocess.Popen

    def launch(
        command: list[str], *, cwd: str, stdout: int, stderr: int,
    ) -> subprocess.Popen[bytes]:
        process = create_autospec(process_type, instance=True)
        process.poll.return_value = None
        launched.append(process)
        return process

    def skip_sleep(seconds: float) -> None:
        pass

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    monkeypatch.setattr(module.time, "sleep", skip_sleep)
    return launched


def read_ini(path: Path) -> ConfigParser:
    config = ConfigParser(interpolation=None)
    config.read(path, encoding="utf-8")
    return config


def test_start_prepares_csp_track_and_single_player_configuration(
    installation: tuple[Path, Path], processes: list[Mock],
) -> None:
    game, documents = installation
    surfaces = game / "content/tracks/monza/data/surfaces.ini"
    original = surfaces.read_bytes()
    manager = AssettoCorsaSessionManager(game, documents_directory=documents)
    scenario = SimulatorScenario(car="bmw_z4_gt3", track="monza")
    try:
        manager.start_session(scenario)
        assert manager.is_running
        assert manager.current_scenario == scenario
        config = read_ini(surfaces)
        assert config["SURFACE_0"]["WAV_PITCH"] == "extended-0"
        assert config["SURFACE_0"]["FRICTION"] == "0.95"
        assert config["_EXTRA_PERMISSIONS"]["ALLOW_CUSTOM_AI_MANIPULATION"] == "1"
        assert surfaces.with_suffix(".ini.before-rl.bak").read_bytes() == original
        assert not (game / "content/tracks/monza/surfaces.ini").exists()
        csp = read_ini(game / "extension/config/new_behaviour.ini")
        assert csp["BASIC"]["ENABLED"] == csp["CUSTOM_AI"]["ENABLED"] == "1"
        assert csp["OTHER"]["KEEP"] == "42"
        race = read_ini(documents / "cfg/race.ini")
        assert race["RACE"]["TRACK"] == "monza"
        assert race["RACE"]["MODEL"] == "bmw_z4_gt3"
        assert race["RACE"]["CARS"] == "1"
        assert race["CAR_0"]["SKIN"] == "default"
        assert race["SESSION_0"]["TYPE"] == "4"
        assert race["SESSION_0"]["SPAWN_SET"] == "HOTLAP_START"
        assert race["REMOTE"]["ACTIVE"] == race["REPLAY"]["ACTIVE"] == "0"
        assert race["OTHER"]["KEEP"] == "42"
        assists = read_ini(documents / "cfg/assists.ini")
        assert assists["ASSISTS"]["AUTO_SHIFTER"] == "0"
        assert not manager.ensure_session(scenario)
        assert len(processes) == 1
    finally:
        manager.stop_session()
    processes[0].terminate.assert_called_once()
    assert not manager.is_running


def test_start_replaces_invalid_session_type_and_spawn_set(
    installation: tuple[Path, Path], processes: list[Mock],
) -> None:
    game, documents = installation
    race_ini = documents / "cfg/race.ini"
    with race_ini.open("a", encoding="utf-8") as file:
        file.write("[SESSION_0]\nNAME=Hotlap\nTYPE=3\nSPAWN_SET=HOTLAP\n")

    with AssettoCorsaSessionManager(game, documents_directory=documents) as manager:
        manager.start_session(SimulatorScenario(car="bmw_z4_gt3", track="monza"))
        session = read_ini(race_ini)["SESSION_0"]
        # AC's session/spawn identifiers, independently checked against the
        # Content Manager HotlapProperties implementation, not our constants.
        assert session["NAME"] == "Hotlap"
        assert session["TYPE"] == "4"
        assert session["SPAWN_SET"] == "HOTLAP_START"
        assert session["DURATION_MINUTES"] == "0"


@pytest.mark.parametrize("layout_has_surfaces", [True, False])
def test_layout_data_is_preferred_with_base_data_as_fallback(
    installation: tuple[Path, Path], processes: list[Mock], layout_has_surfaces: bool,
) -> None:
    game, documents = installation
    layout = game / "content/tracks/monza/short/data"
    layout.mkdir(parents=True)
    base = game / "content/tracks/monza/data/surfaces.ini"
    original_base = base.read_bytes()
    if layout_has_surfaces:
        (layout / "surfaces.ini").write_text("[SURFACE_0]\nWAV_PITCH=extended-1\n", encoding="utf-8")
    with AssettoCorsaSessionManager(game, documents_directory=documents) as manager:
        manager.start_session(SimulatorScenario(car="bmw_z4_gt3", track="monza", track_layout="short"))
        target = layout / "surfaces.ini" if layout_has_surfaces else base
        config = read_ini(target)
        assert config["_EXTRA_PERMISSIONS"]["ALLOW_CUSTOM_AI_MANIPULATION"] == "1"
        if layout_has_surfaces:
            assert config["SURFACE_0"]["WAV_PITCH"] == "extended-1"
            assert base.read_bytes() == original_base


def test_missing_unpacked_surfaces_fails_before_launch(
    installation: tuple[Path, Path], processes: list[Mock],
) -> None:
    game, documents = installation
    (game / "content/tracks/monza/data/surfaces.ini").unlink()
    manager = AssettoCorsaSessionManager(game, documents_directory=documents)
    with pytest.raises(SessionManagerError, match="Unpack the track data"):
        manager.start_session(SimulatorScenario(car="bmw_z4_gt3", track="monza"))
    assert not processes
    assert not (game / "content/tracks/monza/surfaces.ini").exists()


@pytest.mark.parametrize("failure", ["wrong_layout", "interrupted"])
def test_startup_failure_stops_owned_process(
    installation: tuple[Path, Path],
    processes: list[Mock],
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    class Reader:
        def read(self) -> object:
            if failure == "interrupted":
                raise KeyboardInterrupt
            return SimpleNamespace(car_model="bmw_z4_gt3", track_name="monza", track_configuration="wrong")

    clock = count(0.0, 0.01)
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
    game, documents = installation
    manager = AssettoCorsaSessionManager(
        game, documents_directory=documents, static_info_reader=Reader(), startup_timeout=0.1,
    )
    error = KeyboardInterrupt if failure == "interrupted" else SessionManagerError
    with pytest.raises(error):
        manager.start_session(SimulatorScenario(car="bmw_z4_gt3", track="monza"))
    processes[0].terminate.assert_called_once()
    assert not manager.is_running


def test_start_waits_for_matching_static_info_and_driving_callback(
    installation: tuple[Path, Path], processes: list[Mock],
) -> None:
    class Reader:
        def read(self) -> object:
            return SimpleNamespace(car_model="bmw_z4_gt3", track_name="monza", track_configuration="")

    checks = 0

    def is_driving() -> bool:
        nonlocal checks
        checks += 1
        return checks == 3

    game, documents = installation
    with AssettoCorsaSessionManager(
        game, documents_directory=documents, static_info_reader=Reader(), drive_ready_check=is_driving,
    ) as manager:
        manager.start_session(SimulatorScenario(car="bmw_z4_gt3", track="monza"))
        assert checks == 3
