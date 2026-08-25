import ast
import unittest

import torch

from fsrl.core import ConjunctiveLocalTrace, RelationalInputLayout, RetroModulRNN
from fsrl.core.config import TrainConfig
from fsrl.core.local_trace import ConjunctiveLocalTrace as CanonicalTrace
from fsrl.core.plastic_rnn import RetroModulRNN as CanonicalRNN
from fsrl.evaluation import FrozenFastWeightEvaluator
from fsrl.evaluation.frozen_fast_weight import (
    FrozenFastWeightEvaluator as CanonicalEvaluator,
)
from fsrl.paths import REPO_ROOT
from fsrl.tasks import RankingProtocol
from fsrl.tasks.protocol import RankingProtocol as CanonicalProtocol

ROOT = REPO_ROOT


class CoreArchitectureTests(unittest.TestCase):
    def test_public_packages_export_canonical_objects(self):
        self.assertIs(RetroModulRNN, CanonicalRNN)
        self.assertIs(ConjunctiveLocalTrace, CanonicalTrace)
        self.assertIs(FrozenFastWeightEvaluator, CanonicalEvaluator)
        self.assertIs(RankingProtocol, CanonicalProtocol)

    def test_named_input_layout_preserves_checkpoint_abi(self):
        config = TrainConfig(bs=2, hs=5, cs=6)
        layout = RelationalInputLayout(config.cs)
        self.assertEqual(layout.stimulus_width, config.nbstimbits)
        self.assertEqual(layout.bias_index, config.nbstimbits)
        self.assertEqual(layout.time_index, config.nbstimbits + 1)
        self.assertEqual(layout.reward_index, config.nbstimbits + 2)
        self.assertEqual(layout.evidence_index, config.nbstimbits + 3)
        self.assertEqual(layout.input_size, config.inputsize)
        layout.validate_width(config.inputsize)

    def test_plastic_rnn_matches_independent_v1_equations(self):
        torch.manual_seed(19)
        config = TrainConfig(bs=3, hs=5, cs=4).to_model_dict()
        net = RetroModulRNN(config, device="cpu")
        inputs = torch.randn(config["bs"], config["inputsize"])
        hidden = torch.randn(config["bs"], config["hs"])
        eligibility = torch.randn(config["bs"], config["hs"], config["hs"])
        fast_weights = torch.randn(config["bs"], config["hs"], config["hs"])

        hidden_expected = torch.tanh(
            net.i2h(inputs).view(config["bs"], config["hs"], 1)
            + torch.matmul(
                net.w + net.alpha * fast_weights,
                hidden.view(config["bs"], config["hs"], 1),
            )
        ).view(config["bs"], config["hs"])
        logits_expected = net.h2o(hidden_expected)
        value_expected = net.h2v(hidden_expected)
        da_pair = torch.tanh(net.h2DA(hidden_expected))
        da_expected = net.DAmult * (da_pair[:, 0] - da_pair[:, 1])[:, None]
        weights_expected = torch.clamp(
            fast_weights + da_expected.view(config["bs"], 1, 1) * eligibility,
            min=-50.0,
            max=50.0,
        )
        delta = torch.tanh(
            torch.bmm(
                hidden_expected.view(config["bs"], config["hs"], 1),
                hidden.view(config["bs"], 1, config["hs"]),
            )
        )
        eligibility_expected = (1 - net.etaet) * eligibility + net.etaet * delta

        observed = net(inputs, hidden, eligibility, fast_weights)
        expected = (
            logits_expected,
            value_expected,
            da_expected,
            hidden_expected,
            eligibility_expected,
            weights_expected,
        )
        for observed_tensor, expected_tensor in zip(observed, expected, strict=True):
            torch.testing.assert_close(observed_tensor, expected_tensor)

    def test_state_dict_keys_remain_checkpoint_compatible(self):
        config = TrainConfig(bs=1, hs=3, cs=2).to_model_dict()
        net = RetroModulRNN(config, device="cpu")
        self.assertEqual(
            set(net.state_dict()),
            {
                "DAmult",
                "alpha",
                "etaet",
                "h2DA.bias",
                "h2DA.weight",
                "h2o.bias",
                "h2o.weight",
                "h2v.bias",
                "h2v.weight",
                "i2h.bias",
                "i2h.weight",
                "w",
            },
        )

    def test_package_dependencies_follow_the_explicit_layer_graph(self):
        allowed = {
            "analysis": {"analysis", "evaluation", "infra", "tasks"},
            "core": {"core", "infra"},
            "evaluation": {
                "core",
                "evaluation",
                "infra",
                "tasks",
                "training",
            },
            "experiments": {
                "analysis",
                "core",
                "evaluation",
                "experiments",
                "infra",
                "paths",
                "tasks",
                "training",
            },
            "infra": {"infra", "paths"},
            "tasks": {"infra", "tasks"},
            "training": {"core", "infra", "tasks", "training"},
            "workflows": {
                "analysis",
                "evaluation",
                "experiments",
                "infra",
                "paths",
                "tasks",
            },
        }
        violations = []
        for owner, dependencies in allowed.items():
            for source in (ROOT / "fsrl" / owner).rglob("*.py"):
                tree = ast.parse(
                    source.read_text(encoding="utf-8"), filename=str(source)
                )
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                    elif isinstance(node, ast.Import):
                        modules = [alias.name for alias in node.names]
                    else:
                        continue
                    if isinstance(node, ast.ImportFrom):
                        modules = [module]
                    for module in modules:
                        if not module.startswith("fsrl."):
                            continue
                        dependency = module.split(".")[1]
                        if dependency not in dependencies:
                            violations.append(f"{source.relative_to(ROOT)} -> {module}")
        self.assertEqual(violations, [])

    def test_source_and_test_roots_are_not_flat_module_collections(self):
        source_files = {path.name for path in (ROOT / "fsrl").glob("*.py")}
        root_tests = list((ROOT / "tests").glob("test_*.py"))
        self.assertEqual(source_files, {"__init__.py", "paths.py"})
        self.assertEqual(root_tests, [])

    def test_scoped_agent_guides_cover_distinct_repository_boundaries(self):
        expected = {
            "fsrl/AGENTS.md",
            "fsrl/experiments/AGENTS.md",
            "fsrl/infra/AGENTS.md",
            "fsrl/tasks/AGENTS.md",
            "fsrl/workflows/AGENTS.md",
            "reproductions/AGENTS.md",
            "studies/AGENTS.md",
            "synthesis/AGENTS.md",
            "synthesis/figures/AGENTS.md",
            "synthesis/snapshots/AGENTS.md",
            "tests/AGENTS.md",
            "tools/AGENTS.md",
            "workflows/AGENTS.md",
        }
        missing = sorted(path for path in expected if not (ROOT / path).is_file())
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
