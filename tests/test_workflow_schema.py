import copy
import unittest
from pathlib import Path

from fsrl.workflows import check_rendered_readme, load_workflow, validate_workflow

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "workflows" / "relational_model" / "workflow.toml"


class WorkflowSchemaTests(unittest.TestCase):
    def test_workflow_and_human_view_are_in_sync(self):
        result = validate_workflow(load_workflow(WORKFLOW))
        self.assertTrue(result["passed"])
        self.assertEqual(result["workflow_id"], "relational_model")
        self.assertGreaterEqual(result["stages"], 6)
        self.assertTrue(check_rendered_readme(WORKFLOW)["passed"])

    def test_dependency_must_point_to_preceding_stage(self):
        workflow = copy.deepcopy(load_workflow(WORKFLOW))
        workflow["stages"][0]["depends_on"] = [workflow["stages"][-1]["id"]]
        with self.assertRaisesRegex(ValueError, "preceding stages"):
            validate_workflow(workflow)

    def test_missing_implementation_path_is_rejected(self):
        workflow = copy.deepcopy(load_workflow(WORKFLOW))
        workflow["stages"][0]["implementation"] = ["fsrl/does_not_exist.py"]
        with self.assertRaises(FileNotFoundError):
            validate_workflow(workflow)


if __name__ == "__main__":
    unittest.main()
