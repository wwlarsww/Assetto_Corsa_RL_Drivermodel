"""Generate API pages and resolve repository links for the MkDocs build.

Runs through mkdocs-gen-files. Generated Markdown lives in its virtual
filesystem; tracked docs and application modules are never rewritten or imported.
"""

from pathlib import Path, PurePosixPath
import posixpath
import re
from urllib.parse import unquote, urlsplit

import mkdocs_gen_files


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
nav = mkdocs_gen_files.Nav()
nav["Overview"] = "index.md"
module_pages: dict[Path, PurePosixPath] = {}
CONCEPTS = {
    "RacingEnvironment": "racing_environment.md",
    "action": "control.md",
    "action_wrapper": "control.md",
    "agent": "agent_training.md",
    "episode_start": "racing_environment.md",
    "reset_preroll": "racing_environment.md",
    "observation": "observation.md",
    "observation_builder": "observation.md",
    "tyre_dynamics": "observation.md",
    "reward": "reward.md",
    "future_reward": "reward.md",
    "reward_wrapper": "reward.md",
    "predictor": "predictor.md",
    "racing_policy": "predictor.md",
    "racing_ppo": "predictor.md",
    "termination": "termination.md",
    "training_config": "configuration.md",
    "training_run": "agent_training.md",
    "training_callbacks": "agent_training.md",
    "training_snapshots": "diagnostics.md",
    "telemetry_reader": "input.md",
    "static_info_reader": "input.md",
    "track_reader": "input.md",
    "shared_memory": "input.md",
    "simulator_controller": "control.md",
    "session_manager": "session_manager.md",
}

for source_root, group in ((ROOT / "src", "Core"), (ROOT / "tools", "Tools")):
    for source in sorted(source_root.rglob("*.py")):
        if source.name in {"__init__.py", "__main__.py"}:
            continue
        relative = source.relative_to(ROOT / "src" if group == "Core" else ROOT)
        module = ".".join(relative.with_suffix("").parts)
        destination = PurePosixPath("reference", relative.with_suffix(".md").as_posix())
        module_pages[source.resolve()] = destination
        nav[(group, *relative.with_suffix("").parts)] = destination.relative_to("reference").as_posix()
        with mkdocs_gen_files.open(destination, "w") as stream:
            stream.write(f"# {module}\n\n")
            stream.write(f"Generated from `{source.relative_to(ROOT).as_posix()}`.\n\n")
            concept = CONCEPTS.get(source.stem, "architecture.md")
            if group == "Tools" and "diagnostics" in relative.parts:
                concept = "diagnostics.md"
            concept_link = posixpath.relpath(concept, destination.parent.as_posix())
            stream.write(f"For context, see the [project guide]({concept_link}).\n\n")
            stream.write(f"::: {module}\n")

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as stream:
    stream.writelines(nav.build_literate_nav())


def published_target(target: Path) -> PurePosixPath:
    """Use an API page for modules and a generated source view for other files."""
    if target in module_pages:
        return module_pages[target]
    if target.is_dir():
        return PurePosixPath("reference/index.md")
    relative = target.relative_to(ROOT)
    if relative.as_posix() == "README.md":
        return PurePosixPath("index.md")
    destination = PurePosixPath("reference/source", relative.as_posix() + ".md")
    language = {".py": "python", ".toml": "toml"}.get(target.suffix, "text")
    with mkdocs_gen_files.open(destination, "w") as stream:
        stream.write(f"# {relative.as_posix()}\n\n")
        stream.write("This view is generated from the repository file at build time.\n\n")
        stream.write(f"````{language}\n{target.read_text(encoding='utf-8')}\n````\n")
    return destination


def resolve_link(match: re.Match[str], page: Path) -> str:
    url = urlsplit(match[1])
    if url.scheme or url.netloc or not url.path:
        return match[0]
    target = (page.parent / unquote(url.path)).resolve()
    if target.is_relative_to(DOCS) or not target.is_relative_to(ROOT):
        return match[0]
    if not target.exists():
        raise FileNotFoundError(f"Broken repository link in {page.name}: {match[1]}")
    destination = published_target(target)
    relative_url = posixpath.relpath(destination.as_posix(), page.relative_to(DOCS).parent.as_posix())
    if url.fragment:
        relative_url += "#" + url.fragment
    return f"]({relative_url})"


# Keep source links useful both in repository Markdown and in the built site.
for page in sorted(DOCS.rglob("*.md")):
    text = page.read_text(encoding="utf-8")
    resolved = re.sub(r"\]\(([^\s)]+)\)", lambda match: resolve_link(match, page), text)
    if resolved != text:
        with mkdocs_gen_files.open(page.relative_to(DOCS).as_posix(), "w") as stream:
            stream.write(resolved)
