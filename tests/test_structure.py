"""Guard project entry points and references after directory reorganization."""
import ast
import importlib.util
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


class StructureTests(unittest.TestCase):
    def test_all_project_python_sources_parse(self):
        for directory in ("code", "benchmarks", "tests", "dashboard"):
            for path in (ROOT / directory).rglob("*.py"):
                with self.subTest(path=path):
                    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_root_readme_local_links_exist(self):
        for target in re.findall(r"\]\(([^)]+)\)", (ROOT / "readme.md").read_text(encoding="utf-8")):
            if not target.startswith("http"):
                self.assertTrue((ROOT / target).is_file(), target)

    def test_notebook_python_entrypoints_exist(self):
        for path in (ROOT / "notebooks").glob("*.ipynb"):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for cell in notebook["cells"]:
                source = "".join(cell.get("source", []))
                for script in re.findall(r"['\"](code/[^'\"]+\.py)['\"]", source):
                    self.assertTrue((ROOT / script).is_file(), (path.name, script))

    def test_dashboard_discovers_moved_docs_and_scripts(self):
        spec = importlib.util.spec_from_file_location("dashboard_structure_check", ROOT / "dashboard/server.py")
        server = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(server)
        docs = {row["path"] for row in server.docs()}
        self.assertIn("docs/setup/local.md", docs)
        self.assertIn("benchmarks/README.md", docs)
        for task in server.TASKS.values():
            for label, script in task.get("variants", {}).values():
                self.assertTrue((ROOT / script).is_file(), script)


if __name__ == "__main__":
    unittest.main()
