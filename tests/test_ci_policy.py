from pathlib import Path

_FORBIDDEN_WORKFLOW_COMMANDS = (
    "bc-train",
    "rl-train",
    "download-data",
    "telemetry-diagnose",
    "slip-diagnose",
)


def test_github_workflows_do_not_launch_research_or_live_commands() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    workflow_directory = repository_root / ".github" / "workflows"

    for workflow in workflow_directory.glob("*.yml"):
        content = workflow.read_text(encoding="utf-8")
        for command in _FORBIDDEN_WORKFLOW_COMMANDS:
            assert command not in content, (
                f"{workflow.name} must not invoke local-only command {command!r}"
            )
