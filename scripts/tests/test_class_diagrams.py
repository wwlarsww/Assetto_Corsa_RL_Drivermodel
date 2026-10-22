"""Check dependency meaning, source-only analysis and navigable diagram views."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.class_diagrams import ClassGraph, ProjectClass, Relationship, analyze, mermaid


class ClassDiagramTests(unittest.TestCase):
    def test_analysis_resolves_aliases_forward_annotations_and_injected_types(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "src/assetto_corsa_ai"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text(
                'raise RuntimeError("Documentation must never import this package")\n'
            )
            (package / "types.py").write_text(
                "class Base: pass\n"
                "class Engine: pass\n"
                "class Spare: pass\n"
                "class Unused: pass\n"
                "class _Internal: pass\n"
            )
            (package / "driver.py").write_text(
                "from assetto_corsa_ai.types import Base, Engine as Motor, Spare as Reserve, Unused\n"
                "class Driver(Base):\n"
                "    def __init__(self, *, engine: Motor):\n"
                "        self.engine = engine\n"
                "    def spare(self) -> 'Reserve | None':\n"
                "        return self.engine\n"
                "    def note(self, text='Unused'):\n"
                "        return {'': text, 'Unused': text}['']\n"
            )
            graph = analyze(root)
        driver = "assetto_corsa_ai.driver.Driver"
        engine = "assetto_corsa_ai.types.Engine"
        base = "assetto_corsa_ai.types.Base"
        self.assertIn(driver, graph.classes)
        self.assertIn(Relationship(driver, base, "specialization"), graph.relationships)
        self.assertTrue(any(edge.source == driver and edge.target == engine for edge in graph.relationships))
        self.assertTrue(any(edge.source == driver and edge.target.endswith(".Spare") for edge in graph.relationships))
        self.assertFalse(any(edge.target.endswith(".Unused") for edge in graph.relationships))
        self.assertFalse(any(name.endswith("._Internal") for name in graph.classes))
        self.assertFalse(any(name.startswith("src.") for name in graph.classes))

    def test_neighborhood_includes_users_but_does_not_expand_transitively(self):
        graph = ClassGraph(
            {},
            (
                Relationship("A", "B", "uses"),
                Relationship("C", "A", "uses"),
                Relationship("B", "D", "uses"),
            ),
        )
        names, edges = graph.neighbors({"A"})
        self.assertEqual(names, {"A", "B", "C"})
        self.assertEqual(len(edges), 2)
        self.assertNotIn("D", names)

    def test_rendering_keeps_same_named_classes_distinct_and_owner_on_diamond_side(self):
        first = ProjectClass("app.first.Node", "app.first", "Node")
        second = ProjectClass("app.second.Node", "app.second", "Node")
        graph = ClassGraph(
            {first.qualified_name: first, second.qualified_name: second},
            (Relationship(first.qualified_name, second.qualified_name, "composition", "child"),),
        )
        output = mermaid(graph, set(graph.classes), graph.relationships)
        declarations = [line for line in output.splitlines() if "class C" in line]
        self.assertEqual(len(declarations), 2)
        self.assertNotEqual(declarations[0], declarations[1])
        self.assertIn('*--', output)
        self.assertIn(': child', output)


if __name__ == "__main__":
    unittest.main()
