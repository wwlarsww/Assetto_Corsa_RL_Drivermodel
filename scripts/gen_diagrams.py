"""Publish automatically analyzed class diagrams through mkdocs-gen-files."""

from collections import defaultdict
from pathlib import Path, PurePosixPath

import mkdocs_gen_files

from scripts.class_diagrams import analyze, class_link, mermaid, relationship_table, relative_link


ROOT = Path(__file__).resolve().parents[1]
graph = analyze(ROOT)
if not graph.classes:
    raise RuntimeError("Class analysis produced no project classes.")

module_pages = {}
for name, item in graph.classes.items():
    module_pages.setdefault(item.module, PurePosixPath("diagrams/modules", item.module + ".md"))


def write_view(page, title, focus, introduction):
    visible, edges = graph.neighbors(focus)
    with mkdocs_gen_files.open(page, "w") as stream:
        stream.write(f"# {title}\n\n{introduction}\n\n")
        stream.write(f"[Diagram guide and class index]({relative_link(PurePosixPath('diagrams/index.md'), page)}).\n\n")
        stream.write('<div data-class-diagram></div>\n\n')
        stream.write("Only direct relationships to the selected classes are shown. "
                     "Highlighted classes belong to the selected view. Related classes link to their own neighborhood views below.\n\n")
        stream.write("```classdiagram\n" + mermaid(graph, visible, edges, focus) + "```\n\n")
        stream.write("## Classes in this view\n\n")
        stream.write("| Class | Module | API |\n| --- | --- | --- |\n")
        for name in sorted(visible):
            item = graph.classes[name]
            stream.write(f"| {class_link(item, page)} | `{item.module}` | {class_link(item, page, api=True)} |\n")
        stream.write("\n## Direct relationships\n\n")
        stream.write(relationship_table(graph, edges, page) if edges else "No relationships were detected for this view.\n")


for name, item in graph.classes.items():
    write_view(item.diagram_page, item.name, {name}, f"Direct dependencies and dependents of `{name}`.")

for module, page in sorted(module_pages.items()):
    focus = {name for name, item in graph.classes.items() if item.module == module}
    write_view(page, module, focus, "Classes declared in this module, together with their direct neighbors.")

index_page = PurePosixPath("diagrams/index.md")
intro = (ROOT / "docs/diagrams/index.md").read_text(encoding="utf-8")
with mkdocs_gen_files.open(index_page, "w") as stream:
    stream.write(intro)
    stream.write(f"\n## Generated class index\n\n{len(graph.classes)} public project classes "
                 f"and {len(graph.relationships)} detected relationships.\n\n")
    groups = defaultdict(list)
    for name, item in graph.classes.items():
        groups[item.module].append(item)
    for module, items in sorted(groups.items()):
        stream.write(f"### {module}\n\n")
        url = relative_link(module_pages[module], index_page)
        stream.write(f"[Module diagram]({url})\n\n")
        stream.write("| Class | Diagram | API reference |\n| --- | --- | --- |\n")
        for item in sorted(items, key=lambda value: value.name):
            stream.write(f"| `{item.name}` | {class_link(item, index_page)} | {class_link(item, index_page, api=True)} |\n")
        stream.write("\n")
