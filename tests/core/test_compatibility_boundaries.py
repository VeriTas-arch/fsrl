import ast
import unittest
from pathlib import Path

from fsrl.paths import REPO_ROOT

ACTIVE_SOURCE_ROOT = REPO_ROOT / "fsrl"
COMPATIBILITY_FILES = {
    Path("fsrl/core/plastic_rnn.py"),
    Path("fsrl/tasks/meta_tasks.py"),
    Path("fsrl/tasks/registered_protocol.py"),
}


class CompatibilityBoundaryTests(unittest.TestCase):
    def test_active_code_does_not_call_legacy_model_or_task_interfaces(self):
        violations = []
        legacy_tokens = (
            "initialZeroState",
            "initialZeroET",
            "initialZeroPlasticWeights",
            ".GG",
            "fsrl.tasks.meta_tasks",
            "fsrl.tasks.registered_protocol",
            "exclude_liu_graph",
        )
        roots = (
            ACTIVE_SOURCE_ROOT,
            REPO_ROOT / "reproductions" / "relational_learning_2024",
        )
        for root in roots:
            for source in root.rglob("*.py"):
                relative = source.relative_to(REPO_ROOT)
                if relative in COMPATIBILITY_FILES or "upstream" in source.parts:
                    continue
                text = source.read_text(encoding="utf-8")
                for token in legacy_tokens:
                    if token in text:
                        violations.append(f"{relative}: {token}")
        self.assertEqual(violations, [])

    def test_maintained_code_never_writes_a_dat_checkpoint(self):
        violations = []
        roots = (
            ACTIVE_SOURCE_ROOT,
            REPO_ROOT / "reproductions" / "relational_learning_2024",
        )
        for root in roots:
            for source in root.rglob("*.py"):
                if "upstream" in source.parts:
                    continue
                tree = ast.parse(
                    source.read_text(encoding="utf-8"), filename=str(source)
                )
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call):
                        continue
                    if not (
                        isinstance(node.func, ast.Attribute)
                        and node.func.attr == "save"
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "torch"
                    ):
                        continue
                    expression = ast.unparse(node)
                    if ".dat" in expression:
                        violations.append(
                            f"{source.relative_to(REPO_ROOT)}: {expression}"
                        )
        self.assertEqual(violations, [])

    def test_human_dataset_has_one_metadata_authority(self):
        benchmark = (
            REPO_ROOT / "fsrl" / "experiments" / "human" / "benchmark.py"
        ).read_text(encoding="utf-8")
        self.assertIn("LIU_DATASET_FILES", benchmark)
        self.assertNotIn("SOURCE_FILES =", benchmark)
