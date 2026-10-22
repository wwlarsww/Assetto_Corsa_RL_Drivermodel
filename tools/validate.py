"""Choose a focused validator; arguments following its name go to that tool."""

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from typing import cast

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@dataclass(frozen=True)
class ValidationTool:
    module: str
    description: str
    accepts_arguments: bool = False


TOOLS = {
    "track": ValidationTool("validate_track_alignment", "Live spline, track edges, projection and observation plot"),
    "reward": ValidationTool("validate_reward", "Live reward and termination plot"),
    "damage": ValidationTool("validate_damage", "Read raw damage while driving manually"),
    "dynamics": ValidationTool("validate_vehicle_dynamics", "Apply CSP controls to test vehicle dynamics"),
    "controls": ValidationTool("validate_simulator_control", "Interactive CSP controls and session launch", True),
    "environment": ValidationTool("validate_racing_environment", "Gymnasium/SB3 checks with live resets and steps", True),
    "predictor": ValidationTool("validate_predictor", "Offline predictor and future reward validation", True),
}


def select_tool() -> str | None:
    names = tuple(TOOLS)
    print("Assetto Corsa validation\n")
    for index, name in enumerate(names, 1):
        print(f"  {index}. {name}: {TOOLS[name].description}")
    print("  q. Quit")
    while True:
        try:
            answer = input("\nSelect a tool: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if answer in ("q", "quit", "exit"):
            return None
        if answer in TOOLS:
            return answer
        if answer.isdecimal() and 1 <= int(answer) <= len(names):
            return names[int(answer) - 1]
        print("Choose a number, tool name or q.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, epilog="Omit the tool name to open the selection menu.")
    parser.add_argument("tool", nargs="?", choices=tuple(TOOLS))
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help="Options passed unchanged to the selected validator")
    args = parser.parse_args(argv)
    name: str | None = args.tool or select_tool()
    if name is None:
        return
    tool = TOOLS[name]
    arguments: list[str] = args.arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    if not tool.accepts_arguments:
        # These existing validators have no parser. Handle help before any
        # imports or AC access, and reject unsupported options explicitly.
        if arguments in (["-h"], ["--help"]):
            print(f"{name}: {tool.description}\nUsage: python tools/validate.py {name}")
            return
        if arguments:
            parser.error(f"{name} does not accept additional arguments")
    module = importlib.import_module(f"tools.validation.{tool.module}")
    entry = getattr(module, "main")
    if not callable(entry):
        raise TypeError(f"Validator {name} has no callable main()")
    previous_argv = sys.argv
    try:
        sys.argv = [f"tools/validate.py {name}", *arguments]
        cast(Callable[[], None], entry)()
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
