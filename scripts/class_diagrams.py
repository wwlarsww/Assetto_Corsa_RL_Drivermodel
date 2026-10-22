"""Analyze project classes with Pyreverse and render focused Mermaid diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path, PurePosixPath
import posixpath
import sys

import astroid
from astroid import nodes
from pylint.checkers.utils import safe_infer
from pylint.pyreverse.diadefslib import DiadefsHandler
from pylint.pyreverse.diagrams import ClassDiagram, PackageDiagram
from pylint.pyreverse.inspector import Linker, Project
from pylint.pyreverse.main import Run


@dataclass(frozen=True)
class ProjectClass:
    qualified_name: str
    module: str
    name: str

    @property
    def diagram_page(self) -> PurePosixPath:
        return PurePosixPath("diagrams/classes", self.qualified_name + ".md")

    @property
    def api_page(self) -> PurePosixPath:
        return PurePosixPath("reference", *self.module.split(".")).with_suffix(".md")


@dataclass(frozen=True, order=True)
class Relationship:
    """Source depends on target; member records the inferred attribute, if any."""

    source: str
    target: str
    kind: str
    member: str = ""


@dataclass
class ClassGraph:
    classes: dict[str, ProjectClass]
    relationships: tuple[Relationship, ...]

    def neighbors(self, names: set[str]) -> tuple[set[str], tuple[Relationship, ...]]:
        """Return exactly one hop, including incoming and outgoing relationships."""
        edges = tuple(edge for edge in self.relationships if edge.source in names or edge.target in names)
        visible = names | {name for edge in edges for name in (edge.source, edge.target)}
        return visible, edges


def _in_annotation(node: nodes.NodeNG) -> bool:
    """Distinguish forward annotations from docstrings and ordinary string values."""
    current = node
    while current.parent is not None:
        parent = current.parent
        if isinstance(parent, nodes.AnnAssign):
            return parent.annotation is current
        if isinstance(parent, nodes.FunctionDef):
            return parent.returns is current
        if isinstance(parent, nodes.Arguments):
            annotations = [
                *parent.annotations, *parent.posonlyargs_annotations,
                *parent.kwonlyargs_annotations, parent.varargannotation, parent.kwargannotation,
            ]
            return current in annotations
        if not isinstance(parent, (nodes.Subscript, nodes.BinOp, nodes.Tuple, nodes.List)):
            return False
        current = parent
    return False


def analyze(root: Path) -> ClassGraph:
    """Read source statically; never import the simulator, learning stack or Qt.

    Explicit canonical module names prevent the namespace package "src" from
    splitting definitions and imported references into different identities.
    Pyreverse supplies inferred structural relationships. Direct references to
    project class symbols supplement it, including constructor type annotations.
    """
    astroid.MANAGER.clear_cache()
    project = Project("Assetto Corsa RL")
    project.path = str(root)
    original_path = sys.path[:]
    sys.path[:0] = [str(root / "src"), str(root)]
    try:
        for base in (root / "src/assetto_corsa_ai", root / "tools"):
            for path in sorted(base.rglob("*.py")):
                relative = path.relative_to(root / "src" if base.name == "assetto_corsa_ai" else root)
                parts = relative.with_suffix("").parts
                if parts[-1] == "__main__":
                    continue
                if parts[-1] == "__init__":
                    parts = parts[:-1]
                module = ".".join(parts)
                project.add_module(astroid.MANAGER.ast_from_file(str(path), modname=module))

        config = Run(["--output", "mmd", "--filter-mode", "ALL", "--only-classnames"]).config
        diagrams = DiadefsHandler(config, []).get_diadefs(project, Linker(project, tag=True))
        diagram = next(item for item in diagrams if isinstance(item, ClassDiagram) and not isinstance(item, PackageDiagram))
        definitions = {
            entity.node.qname(): entity.node
            for entity in diagram.classes()
            if isinstance(entity.node.parent, nodes.Module) and not entity.node.name.startswith("_")
        }
        classes = {
            name: ProjectClass(name, node.root().name, node.name)
            for name, node in sorted(definitions.items())
        }
        edges: set[Relationship] = set()
        for kind in ("specialization", "association", "aggregation", "composition"):
            for relation in diagram.get_relationships(kind):
                source = relation.from_object.node.qname()
                target = relation.to_object.node.qname()
                if kind != "specialization":
                    source, target = target, source
                if source in classes and target in classes and source != target:
                    edges.add(Relationship(source, target, kind, relation.name or ""))

        class_names = {item.name for item in classes.values()}
        class_names.update(
            alias
            for module in project.modules
            for statement in module.nodes_of_class(nodes.ImportFrom)
            for original, alias in statement.names
            if alias and original in class_names
        )
        for name, definition in definitions.items():
            for expression in definition.nodes_of_class((nodes.Name, nodes.Attribute)):
                symbol = expression.name if isinstance(expression, nodes.Name) else expression.attrname
                if symbol not in class_names:
                    continue
                target = safe_infer(expression)
                if isinstance(target, nodes.ClassDef) and target.qname() in classes and target.qname() != name:
                    edges.add(Relationship(name, target.qname(), "uses"))

            # Quoted forward annotations have no Name node in the original AST.
            for annotation in definition.nodes_of_class(nodes.Const):
                if not isinstance(annotation.value, str):
                    continue
                if not annotation.value.strip() or not _in_annotation(annotation):
                    continue
                try:
                    expression = astroid.extract_node(annotation.value)
                except (astroid.AstroidError, SyntaxError, ValueError):
                    continue
                if not isinstance(expression, nodes.NodeNG):
                    continue
                expression.parent = annotation.parent
                for reference in expression.nodes_of_class((nodes.Name, nodes.Attribute)):
                    symbol = reference.name if isinstance(reference, nodes.Name) else reference.attrname
                    if symbol not in class_names:
                        continue
                    target = safe_infer(reference)
                    if isinstance(target, nodes.ClassDef) and target.qname() in classes and target.qname() != name:
                        edges.add(Relationship(name, target.qname(), "uses"))

        structural = {(edge.source, edge.target) for edge in edges if edge.kind != "uses"}
        edges = {edge for edge in edges if edge.kind != "uses" or (edge.source, edge.target) not in structural}
        return ClassGraph(classes, tuple(sorted(edges)))
    finally:
        sys.path[:] = original_path


def relative_link(target: PurePosixPath, page: PurePosixPath) -> str:
    return posixpath.relpath(target.as_posix(), page.parent.as_posix())


def class_link(item: ProjectClass, page: PurePosixPath, *, api: bool = False) -> str:
    target = item.api_page if api else item.diagram_page
    url = relative_link(target, page)
    if api:
        url += "#" + item.qualified_name
    return f"[{item.name}]({url})"


def node_id(name: str) -> str:
    # Stable IDs preserve distinct classes with the same short name.
    return "C" + sha1(name.encode("utf-8")).hexdigest()[:16]


ARROWS = {
    "specialization": "--|>",
    "association": "-->",
    "aggregation": "o--",
    "composition": "*--",
    "uses": "..>",
}
LABELS = {
    "specialization": "inherits",
    "association": "associated field",
    "aggregation": "received object",
    "composition": "created object",
    "uses": "references",
}


def mermaid(graph: ClassGraph, visible: set[str], edges: tuple[Relationship, ...], focus: set[str] | None = None) -> str:
    lines = ["classDiagram", "    direction LR"]
    for name in sorted(visible):
        lines.append(f'    class {node_id(name)}["{graph.classes[name].name}"]')
    grouped: dict[tuple[str, str, str], set[str]] = {}
    for edge in edges:
        grouped.setdefault((edge.source, edge.target, edge.kind), set()).add(edge.member)
    for (source, target, kind), members in sorted(grouped.items()):
        label = ", ".join(sorted(member for member in members if member)) or LABELS[kind]
        lines.append(f"    {node_id(source)} {ARROWS[kind]} {node_id(target)} : {label}")
    for name in sorted(focus or set()):
        lines.append(f"    style {node_id(name)} fill:#fff4cc,stroke:#9a6700,stroke-width:3px")
    return "\n".join(lines) + "\n"


def relationship_table(graph: ClassGraph, edges: tuple[Relationship, ...], page: PurePosixPath) -> str:
    lines = [
        "| Dependent class | Related class | Relationship | Field |",
        "| --- | --- | --- | --- |",
    ]
    for edge in edges:
        source = class_link(graph.classes[edge.source], page)
        target = class_link(graph.classes[edge.target], page)
        member = f"`{edge.member}`" if edge.member else "—"
        lines.append(f"| {source} | {target} | {LABELS[edge.kind]} | {member} |")
    return "\n".join(lines) + "\n"
