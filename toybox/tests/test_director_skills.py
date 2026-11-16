"""Exercise the real prompt/skill functions without importing GPU/media dependencies."""
import ast
from pathlib import Path
import re
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
NAMES = {"SKILL_NAMES", "SYSTEM_REF", "SYSTEM_BASE", "LABEL", "CHINESE_NAME"}
FUNCTIONS = {"load_skills", "skill_text", "rewrite_message"}
tree = ast.parse((ROOT / "server.py").read_text(encoding="utf-8"))
nodes = [node for node in tree.body if (
    isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS
    or isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id in NAMES for target in node.targets
    )
)]
scope = {"Path": Path, "re": re, "BUILTIN_SKILLS": ROOT / "directional_skills"}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / "server.py"), "exec"), scope)


class DirectorSkillsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.external = Path(self.tmp.name)
        self.cfg = {"skills_dir": str(self.external)}

    def add_external(self, name, text):
        directory = self.external / name
        directory.mkdir()
        (directory / "SKILL.md").write_text(text, encoding="utf-8")

    def test_builtin_available_without_external_install(self):
        self.cfg["skills_dir"] = str(self.external / "missing")
        entries = scope["load_skills"](self.cfg)
        self.assertIn("product_ad", [entry["id"] for entry in entries])
        self.assertIn("產品廣告", next(e["title"] for e in entries if e["id"] == "product_ad"))

    def test_external_directors_still_load(self):
        self.add_external("existing", "Existing direction")
        self.assertIn("existing", [e["id"] for e in scope["load_skills"](self.cfg)])
        self.assertEqual(scope["skill_text"](self.cfg, "existing"), "Existing direction")

    def test_builtin_collision_is_unique_and_uses_bundled_text(self):
        self.add_external("product_ad", "Obsolete external direction")
        entries = scope["load_skills"](self.cfg)
        self.assertEqual(sum(e["id"] == "product_ad" for e in entries), 1)
        text = scope["skill_text"](self.cfg, "product_ad")
        self.assertIn("# 产品广告导演", text)
        self.assertNotIn("description:", text)
        self.assertNotIn("Obsolete external direction", text)

    def test_empty_missing_and_traversal(self):
        self.assertEqual(scope["skill_text"](self.cfg, ""), "")
        for name in ("missing", "../product_ad", "/tmp/product_ad"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                scope["skill_text"](self.cfg, name)

    def test_selected_direction_reaches_ref_and_base_rewrite_messages(self):
        guide = self.external / "guide.md"
        guide.write_text("Official guide sentinel", encoding="utf-8")
        self.cfg.update(rewriter={"image_side": 768}, guides={"ref": str(guide), "base": str(guide)})
        chosen = {"image": [], "video": [], "audio": []}
        for task in ("Ref2VA", "T2VA"):
            with self.subTest(task=task):
                system, parts = scope["rewrite_message"](
                    self.cfg, "展示收纳盒", chosen, task, 10, "product_ad", None
                )
                text = "".join(value for kind, value in parts if kind == "text")
                self.assertIn("Official guide sentinel", system)
                self.assertIn("# 产品广告导演", text)
                self.assertIn("10.000000 seconds", text)
                self.assertIn("Original prompt:\n展示收纳盒", text)
                _, no_skill = scope["rewrite_message"](
                    self.cfg, "展示收纳盒", chosen, task, 10, "", None
                )
                self.assertNotIn("# 产品广告导演", str(no_skill))


if __name__ == "__main__":
    unittest.main()
