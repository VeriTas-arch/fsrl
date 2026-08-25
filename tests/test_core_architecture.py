import ast
import unittest
from pathlib import Path

import torch

from fsrl.config import TrainConfig
from fsrl.conjunctive_local_trace import ConjunctiveLocalTrace as CompatibilityTrace
from fsrl.core import ConjunctiveLocalTrace, RelationalInputLayout, RetroModulRNN
from fsrl.evaluation.frozen_fast_weight import (
    FrozenFastWeightEvaluator as CanonicalEvaluator,
)
from fsrl.liu_eval import FrozenFastWeightEvaluator as CompatibilityEvaluator
from fsrl.model import RetroModulRNN as CompatibilityRNN
from fsrl.ranking_protocol import RankingProtocol as CompatibilityProtocol
from fsrl.tasks.protocol import RankingProtocol

ROOT = Path(__file__).resolve().parents[1]


class CoreArchitectureTests(unittest.TestCase):
    def test_compatibility_imports_resolve_to_canonical_objects(self):
        self.assertIs(CompatibilityRNN, RetroModulRNN)
        self.assertIs(CompatibilityTrace, ConjunctiveLocalTrace)
        self.assertIs(CompatibilityEvaluator, CanonicalEvaluator)
        self.assertIs(CompatibilityProtocol, RankingProtocol)

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

    def test_stable_packages_do_not_import_diagnostic_runners(self):
        forbidden_tokens = {
            "_audit",
            "_confirmation",
            "_pilot",
            "_replication",
            "_transport",
            "formal_runtime",
            "liu_mainline",
        }
        stable_roots = (
            ROOT / "fsrl" / "analysis",
            ROOT / "fsrl" / "core",
            ROOT / "fsrl" / "evaluation",
            ROOT / "fsrl" / "tasks",
            ROOT / "fsrl" / "training",
            ROOT / "fsrl" / "workflows",
        )
        violations = []
        for stable_root in stable_roots:
            for source in stable_root.rglob("*.py"):
                tree = ast.parse(
                    source.read_text(encoding="utf-8"), filename=str(source)
                )
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                    elif isinstance(node, ast.Import):
                        module = ",".join(alias.name for alias in node.names)
                    else:
                        continue
                    if any(token in module for token in forbidden_tokens):
                        violations.append(f"{source.relative_to(ROOT)} -> {module}")
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
