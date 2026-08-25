import copy
import unittest

from fsrl.paths import REPO_ROOT
from fsrl.workflows import check_rendered_readme, load_workflow, validate_workflow

ROOT = REPO_ROOT
WORKFLOW = ROOT / "workflows" / "relational_model" / "workflow.toml"


class WorkflowSchemaTests(unittest.TestCase):
    def test_workflow_and_human_view_are_in_sync(self):
        result = validate_workflow(load_workflow(WORKFLOW))
        self.assertTrue(result["passed"])
        self.assertEqual(result["workflow_id"], "relational_model")
        self.assertGreaterEqual(result["stages"], 6)
        self.assertGreaterEqual(result["evidence"], result["studies"])
        self.assertGreaterEqual(result["test_paths"], result["stages"])
        self.assertEqual(result["figures"], 4)
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

    def test_evidence_pointer_must_resolve(self):
        workflow = copy.deepcopy(load_workflow(WORKFLOW))
        workflow["stages"][1]["evidence"][0]["json_pointer"] = "/not/a/field"
        with self.assertRaisesRegex(ValueError, "pointer does not resolve"):
            validate_workflow(workflow)

    def test_every_declared_study_requires_exact_evidence(self):
        workflow = copy.deepcopy(load_workflow(WORKFLOW))
        workflow["stages"][0]["studies"].append("development_qualification")
        with self.assertRaisesRegex(ValueError, "studies without exact evidence"):
            validate_workflow(workflow)

    def test_figure_id_must_exist_in_its_specification(self):
        workflow = copy.deepcopy(load_workflow(WORKFLOW))
        workflow["stages"][-1]["figures"][0]["figure"] = "figure_missing"
        with self.assertRaisesRegex(ValueError, "figure is absent"):
            validate_workflow(workflow)
