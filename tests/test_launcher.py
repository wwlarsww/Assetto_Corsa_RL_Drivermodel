from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from assetto_corsa_rl.ac_integration.launcher import (
    AssettoCorsaLaunchError,
    AssettoCorsaSessionLauncher,
    split_track_identifier,
)
from assetto_corsa_rl.ac_integration.shared_memory import AssettoCorsaTelemetry
from assetto_corsa_rl.config import (
    LINEAR_BC_X360_CONTROL_PROFILE,
    AssettoCorsaRuntimeConfig,
    ProjectConfig,
)


class _RunningProcess:
    pid = 1234

    def poll(self) -> None:
        return None


def _launcher_config(tmp_path: Path) -> tuple[ProjectConfig, Path]:
    installation = tmp_path / "assettocorsa"
    (installation / "content/cars/bmw_z4_gt3/skins/test_skin").mkdir(parents=True)
    (installation / "content/tracks/monza/ui").mkdir(parents=True)
    (installation / "content/tracks/monza/models.ini").write_text(
        "[MODEL_0]\nFILE=monza.kn5\n",
        encoding="ascii",
    )
    fast_lane = installation / "content/tracks/monza/ai/fast_lane.ai"
    fast_lane.parent.mkdir(parents=True)
    fast_lane.write_bytes(b"synthetic launcher fixture")
    (installation / "acs.exe").write_bytes(b"test")
    race_path = tmp_path / "documents/cfg/race.ini"
    race_path.parent.mkdir(parents=True)
    race_path.write_bytes(b"[RACE]\nTRACK=magione\nMODEL=abarth500\n")
    race_path.with_name("assists.ini").write_bytes(
        b"[ASSISTS]\r\nAUTO_SHIFTER=0\r\nAUTO_CLUTCH=0\r\nAUTO_BLIP=1\r\n"
    )
    base = ProjectConfig()
    return (
        replace(
            base,
            assetto_corsa=AssettoCorsaRuntimeConfig(
                installation_directory=str(installation),
                controls_configuration_path=str(race_path.with_name("controls.ini")),
                launch_timeout_s=2.0,
                launch_poll_interval_s=0.01,
            ),
        ),
        race_path,
    )


def _matching_telemetry(packet_id: int = 1) -> AssettoCorsaTelemetry:
    return cast(
        AssettoCorsaTelemetry,
        SimpleNamespace(
            physics=SimpleNamespace(packet_id=packet_id),
            graphics=SimpleNamespace(status=2, packet_id=packet_id),
            static=SimpleNamespace(
                car_model="bmw_z4_gt3",
                track="monza",
                track_configuration="",
            ),
        ),
    )


def test_launcher_writes_hotlap_and_restores_exact_user_configuration(tmp_path: Path) -> None:
    config, race_path = _launcher_config(tmp_path)
    original = race_path.read_bytes()
    assists_path = race_path.with_name("assists.ini")
    original_assists = assists_path.read_bytes()
    probes = iter((None, _matching_telemetry(1), _matching_telemetry(2)))
    starts: list[tuple[Path, Path]] = []

    def start(executable: Path, working_directory: Path) -> _RunningProcess:
        assert (working_directory / "steam_appid.txt").read_text(encoding="ascii") == "244210"
        generated_assists = assists_path.read_text(encoding="utf-8")
        assert "AUTO_SHIFTER=1" in generated_assists
        assert "AUTO_CLUTCH=1" in generated_assists
        assert assists_path.with_name("assists.ini.acrl-backup").is_file()
        starts.append((executable, working_directory))
        return _RunningProcess()

    with AssettoCorsaSessionLauncher(
        config,
        process_factory=start,
        telemetry_probe=lambda: next(probes),
    ):
        generated_bytes = race_path.read_bytes()
        assert b"\r\n" in generated_bytes
        assert b"\n" not in generated_bytes.replace(b"\r\n", b"")
        generated = generated_bytes.decode("utf-8")
        assert "TRACK=monza" in generated
        assert "CONFIG_TRACK=" in generated
        assert "MODEL=bmw_z4_gt3" in generated
        assert "TYPE=4" in generated
        assert race_path.with_name("race.ini.acrl-backup").is_file()
        assert (Path(config.assetto_corsa.installation_directory) / "steam_appid.txt").read_text(
            encoding="ascii"
        ) == "244210"

    assert race_path.read_bytes() == original
    assert assists_path.read_bytes() == original_assists
    assert not race_path.with_name("race.ini.acrl-backup").exists()
    assert not assists_path.with_name("assists.ini.acrl-backup").exists()
    assert not (Path(config.assetto_corsa.installation_directory) / "steam_appid.txt").exists()
    assert starts == [
        (
            Path(config.assetto_corsa.installation_directory) / "acs.exe",
            Path(config.assetto_corsa.installation_directory),
        )
    ]


def test_launcher_refuses_to_overwrite_recovery_backup(tmp_path: Path) -> None:
    config, race_path = _launcher_config(tmp_path)
    backup = race_path.with_name("race.ini.acrl-backup")
    backup.write_bytes(b"original")
    launcher = AssettoCorsaSessionLauncher(config, telemetry_probe=lambda: None)
    with pytest.raises(AssettoCorsaLaunchError, match="recovery backup already exists"):
        launcher.launch()
    assert backup.read_bytes() == b"original"


def test_launcher_rejects_metadata_without_track_models(tmp_path: Path) -> None:
    config, _ = _launcher_config(tmp_path)
    model_manifest = (
        Path(config.assetto_corsa.installation_directory) / "content/tracks/monza/models.ini"
    )
    model_manifest.unlink()

    launcher = AssettoCorsaSessionLauncher(config, telemetry_probe=lambda: None)
    with pytest.raises(AssettoCorsaLaunchError, match="required model manifest is missing"):
        launcher.launch()


def test_launcher_preserves_existing_assetto_corsa_steam_app_id(tmp_path: Path) -> None:
    config, _ = _launcher_config(tmp_path)
    app_id_path = Path(config.assetto_corsa.installation_directory) / "steam_appid.txt"
    app_id_path.write_bytes(b"244210\r\n")
    probes = iter((None, _matching_telemetry(1), _matching_telemetry(2)))

    with AssettoCorsaSessionLauncher(
        config,
        process_factory=lambda _executable, _working_directory: _RunningProcess(),
        telemetry_probe=lambda: next(probes),
    ):
        assert app_id_path.read_bytes() == b"244210\r\n"

    assert app_id_path.read_bytes() == b"244210\r\n"


def test_launcher_rejects_unexpected_steam_app_id_and_restores_race_ini(
    tmp_path: Path,
) -> None:
    config, race_path = _launcher_config(tmp_path)
    original = race_path.read_bytes()
    app_id_path = Path(config.assetto_corsa.installation_directory) / "steam_appid.txt"
    app_id_path.write_text("unexpected", encoding="ascii")

    launcher = AssettoCorsaSessionLauncher(config, telemetry_probe=lambda: None)
    with pytest.raises(AssettoCorsaLaunchError, match="unexpected Steam App ID"):
        launcher.launch()

    assert race_path.read_bytes() == original
    assert app_id_path.read_text(encoding="ascii") == "unexpected"


@pytest.mark.parametrize(
    ("track_id", "expected"),
    [
        ("ks_nurburgring-layout_gp_a", ("ks_nurburgring", "layout_gp_a")),
        ("monza", ("monza", "")),
    ],
)
def test_split_track_identifier(track_id: str, expected: tuple[str, str]) -> None:
    assert split_track_identifier(track_id) == expected


def test_launcher_gracefully_stops_only_its_owned_process(tmp_path: Path) -> None:
    config, _race_path = _launcher_config(tmp_path)

    class _StoppableProcess:
        pid = 4321

        def __init__(self) -> None:
            self.running = True

        def poll(self) -> int | None:
            return None if self.running else 0

    process = _StoppableProcess()
    probes = iter((None, _matching_telemetry(1), _matching_telemetry(2)))

    def request_close(selected: object) -> bool:
        assert selected is process
        process.running = False
        return True

    launcher = AssettoCorsaSessionLauncher(
        config,
        process_factory=lambda _executable, _working_directory: process,
        process_close_request=request_close,
        telemetry_probe=lambda: next(probes),
    )
    launcher.launch()
    assert launcher.session_origin == "launched"
    assert launcher.stop_owned_session(timeout_s=1.0)
    assert launcher.session_origin == "not_started"
    launcher.restore_user_configuration()


def test_launcher_waits_for_and_adopts_delayed_steam_process(tmp_path: Path) -> None:
    config, _race_path = _launcher_config(tmp_path)

    class _BootstrapProcess:
        pid = 2001

        def poll(self) -> int:
            return 0

    class _SteamProcess:
        pid = 2002

        def __init__(self) -> None:
            self.running = True
            self.released = False

        def poll(self) -> int | None:
            return None if self.running else 0

        def close(self) -> None:
            self.released = True

    steam_process = _SteamProcess()
    locator_calls = 0
    telemetry_packet = 0

    def locate(_executable: Path) -> tuple[_SteamProcess, ...]:
        nonlocal locator_calls
        locator_calls += 1
        return () if locator_calls < 3 else (steam_process,)

    def probe() -> AssettoCorsaTelemetry | None:
        nonlocal telemetry_packet
        if locator_calls < 3:
            return None
        telemetry_packet += 1
        return _matching_telemetry(telemetry_packet)

    def request_close(process: object) -> bool:
        assert process is steam_process
        steam_process.running = False
        return True

    launcher = AssettoCorsaSessionLauncher(
        config,
        process_factory=lambda _executable, _working_directory: _BootstrapProcess(),
        process_close_request=request_close,
        existing_process_locator=locate,
        telemetry_probe=probe,
        sleep=lambda _seconds: None,
    )

    launcher.launch()

    assert launcher.session_origin == "launched"
    assert locator_calls == 3
    assert launcher.stop_owned_session(timeout_s=1.0)
    assert steam_process.released
    launcher.restore_user_configuration()


def test_launcher_waits_until_timeout_when_steam_does_not_start_game(tmp_path: Path) -> None:
    config, _race_path = _launcher_config(tmp_path)

    class _BootstrapProcess:
        pid = 3001

        def poll(self) -> int:
            return 0

    clock_s = 0.0

    def monotonic_clock() -> float:
        return clock_s

    def sleep(duration_s: float) -> None:
        nonlocal clock_s
        clock_s += duration_s

    launcher = AssettoCorsaSessionLauncher(
        config,
        process_factory=lambda _executable, _working_directory: _BootstrapProcess(),
        existing_process_locator=lambda _executable: (),
        telemetry_probe=lambda: None,
        monotonic_clock=monotonic_clock,
        sleep=sleep,
    )

    with pytest.raises(AssettoCorsaLaunchError, match="timed out waiting for Steam"):
        launcher.launch()

    assert clock_s >= config.assetto_corsa.launch_timeout_s


def test_launcher_rejects_ambiguous_steam_handoff_processes(tmp_path: Path) -> None:
    config, _race_path = _launcher_config(tmp_path)

    class _BootstrapProcess:
        pid = 4001

        def poll(self) -> int:
            return 0

    class _SteamProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid
            self.released = False

        def poll(self) -> None:
            return None

        def close(self) -> None:
            self.released = True

    first = _SteamProcess(4002)
    second = _SteamProcess(4003)
    launcher = AssettoCorsaSessionLauncher(
        config,
        process_factory=lambda _executable, _working_directory: _BootstrapProcess(),
        existing_process_locator=lambda _executable: (first, second),
        telemetry_probe=lambda: None,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(AssettoCorsaLaunchError, match="multiple Assetto Corsa processes"):
        launcher.launch()

    assert first.released
    assert second.released


def test_launcher_can_require_a_new_session(tmp_path: Path) -> None:
    config, _race_path = _launcher_config(tmp_path)
    launcher = AssettoCorsaSessionLauncher(
        config,
        telemetry_probe=_matching_telemetry,
        reuse_existing_session=False,
    )
    with pytest.raises(AssettoCorsaLaunchError, match="requires a newly launched session"):
        launcher.launch()


def test_launcher_reuses_only_an_advancing_matching_session(tmp_path: Path) -> None:
    config, _race_path = _launcher_config(tmp_path)
    probes = iter((_matching_telemetry(10), _matching_telemetry(11)))
    launcher = AssettoCorsaSessionLauncher(
        config,
        telemetry_probe=lambda: next(probes),
        sleep=lambda _seconds: None,
    )

    launcher.launch()

    assert launcher.session_origin == "attached"


def test_launcher_rejects_stale_matching_shared_memory(tmp_path: Path) -> None:
    config, _race_path = _launcher_config(tmp_path)
    frozen = _matching_telemetry(10)
    launcher = AssettoCorsaSessionLauncher(
        config,
        existing_process_locator=lambda _executable: (),
        telemetry_probe=lambda: frozen,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(AssettoCorsaLaunchError, match="requires exactly one process"):
        launcher.launch()

    assert launcher.session_origin == "not_started"


def test_launcher_recovers_one_exact_stale_process_and_launches_fresh_session(
    tmp_path: Path,
) -> None:
    config, _race_path = _launcher_config(tmp_path)

    class _StaleProcess:
        pid = 4321

        def __init__(self) -> None:
            self.running = True

        def poll(self) -> int | None:
            return None if self.running else 0

    stale = _StaleProcess()
    starts: list[Path] = []
    fresh_packet_id = 10

    def telemetry_probe() -> AssettoCorsaTelemetry:
        nonlocal fresh_packet_id
        if stale.running:
            return _matching_telemetry(10)
        fresh_packet_id += 1
        return _matching_telemetry(fresh_packet_id)

    def request_close(process: object) -> bool:
        assert process is stale
        stale.running = False
        return True

    def start(executable: Path, _working_directory: Path) -> _RunningProcess:
        starts.append(executable)
        return _RunningProcess()

    with AssettoCorsaSessionLauncher(
        config,
        process_factory=start,
        process_close_request=request_close,
        existing_process_locator=lambda _executable: (stale,),
        telemetry_probe=telemetry_probe,
        sleep=lambda _seconds: None,
    ) as launcher:
        assert launcher.session_origin == "launched"

    assert not stale.running
    assert starts == [Path(config.assetto_corsa.installation_directory) / "acs.exe"]


def test_launcher_does_not_close_ambiguous_stale_processes(tmp_path: Path) -> None:
    config, _race_path = _launcher_config(tmp_path)
    first = _RunningProcess()
    second = _RunningProcess()
    close_requests: list[object] = []

    def request_close(process: object) -> bool:
        close_requests.append(process)
        return True

    launcher = AssettoCorsaSessionLauncher(
        config,
        process_close_request=request_close,
        existing_process_locator=lambda _executable: (first, second),
        telemetry_probe=lambda: _matching_telemetry(10),
        sleep=lambda _seconds: None,
    )

    with pytest.raises(AssettoCorsaLaunchError, match="found 2"):
        launcher.launch()

    assert close_requests == []


def test_launcher_force_terminates_one_exact_stale_process_without_window(
    tmp_path: Path,
) -> None:
    config, _race_path = _launcher_config(tmp_path)

    class _ForceableStaleProcess:
        pid = 9876

        def __init__(self) -> None:
            self.running = True
            self.terminate_requests = 0

        def poll(self) -> int | None:
            return None if self.running else 1

        def terminate(self) -> None:
            self.terminate_requests += 1
            self.running = False

    stale = _ForceableStaleProcess()
    starts: list[Path] = []
    fresh_packet_id = 20

    def telemetry_probe() -> AssettoCorsaTelemetry:
        nonlocal fresh_packet_id
        if stale.running:
            return _matching_telemetry(20)
        fresh_packet_id += 1
        return _matching_telemetry(fresh_packet_id)

    def start(executable: Path, _working_directory: Path) -> _RunningProcess:
        starts.append(executable)
        return _RunningProcess()

    with AssettoCorsaSessionLauncher(
        config,
        process_factory=start,
        process_close_request=lambda _process: False,
        existing_process_locator=lambda _executable: (stale,),
        telemetry_probe=telemetry_probe,
        sleep=lambda _seconds: None,
    ) as launcher:
        assert launcher.session_origin == "launched"

    assert not stale.running
    assert stale.terminate_requests == 1
    assert starts == [Path(config.assetto_corsa.installation_directory) / "acs.exe"]


def test_launcher_temporarily_selects_virtual_xinput_index(tmp_path: Path) -> None:
    config, _race_path = _launcher_config(tmp_path)
    controls_path = config.assetto_corsa.controls_path
    original = b"[HEADER]\r\nINPUT_METHOD=X360\r\n[X360]\r\nJOYPAD_INDEX=0\r\n"
    controls_path.write_bytes(original)
    probes = iter((None, _matching_telemetry(1), _matching_telemetry(2)))

    def start(_executable: Path, _working_directory: Path) -> _RunningProcess:
        generated = controls_path.read_text(encoding="utf-8")
        assert "JOYPAD_INDEX=2" in generated
        assert controls_path.with_name("controls.ini.acrl-backup").is_file()
        return _RunningProcess()

    with AssettoCorsaSessionLauncher(
        config,
        process_factory=start,
        telemetry_probe=lambda: next(probes),
        controller_index=2,
    ):
        pass

    assert controls_path.read_bytes() == original
    assert not controls_path.with_name("controls.ini.acrl-backup").exists()


def test_launcher_temporarily_linearizes_x360_steering(tmp_path: Path) -> None:
    config, _race_path = _launcher_config(tmp_path)
    config = replace(
        config,
        assetto_corsa=replace(
            config.assetto_corsa,
            x360_control_profile=LINEAR_BC_X360_CONTROL_PROFILE,
        ),
    )
    controls_path = config.assetto_corsa.controls_path
    original = (
        b"[HEADER]\r\nINPUT_METHOD=X360\r\n[X360]\r\nJOYPAD_INDEX=0\r\n"
        b"STEER_GAMMA=2\r\nSTEER_FILTER=0.7\r\nSPEED_SENSITIVITY=0.2\r\n"
        b"STEER_DEADZONE=0.1\r\nSTEER_SPEED=0.2\r\n"
    )
    controls_path.write_bytes(original)
    probes = iter((None, _matching_telemetry(1), _matching_telemetry(2)))

    def start(_executable: Path, _working_directory: Path) -> _RunningProcess:
        generated = controls_path.read_text(encoding="utf-8")
        assert "JOYPAD_INDEX=1" in generated
        assert "STEER_GAMMA=1" in generated
        assert "STEER_FILTER=0.7" in generated
        assert "SPEED_SENSITIVITY=0" in generated
        assert "STEER_DEADZONE=0" in generated
        assert "STEER_SPEED=1" in generated
        return _RunningProcess()

    with AssettoCorsaSessionLauncher(
        config,
        process_factory=start,
        telemetry_probe=lambda: next(probes),
        controller_index=1,
    ):
        pass

    assert controls_path.read_bytes() == original
    assert not controls_path.with_name("controls.ini.acrl-backup").exists()
