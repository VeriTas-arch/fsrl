import unittest

import torch

from fsrl.core.local_trace import PackedConjunctiveLocalTrace
from fsrl.core.model_config import RetroModelConfig
from fsrl.core.plastic_rnn import RetroModulRNN
from fsrl.experiments.pl_direct_training.model import map_shadow_model
from fsrl.experiments.pl_direct_training.optimization import (
    clip_effective_backbone_gradients,
    folded_parameters,
    make_optimizer,
)
from fsrl.experiments.pl_direct_training.protocol import load_specification


class DirectOptimizationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3001)
        shadow = RetroModulRNN(RetroModelConfig(37, 7, 2, 2), device="cpu")
        self.backbone = map_shadow_model(shadow, "time_retained_control")
        self.local = PackedConjunctiveLocalTrace(15, device="cpu")
        self.optimization = load_specification()["optimization"]

    def test_optimizer_uses_doubled_folded_learning_rate(self):
        optimizer = make_optimizer(self.backbone, self.local, self.optimization)
        groups = {group["name"]: group for group in optimizer.param_groups}
        self.assertEqual(groups["backbone"]["lr"], 1e-4)
        self.assertEqual(groups["folded_backbone"]["lr"], 2e-4)
        self.assertEqual(groups["local"]["lr"], 0.01)
        self.assertEqual(
            {id(value) for value in groups["folded_backbone"]["params"]},
            {id(value) for value in folded_parameters(self.backbone)},
        )

    def test_gradient_norm_counts_folded_coordinates_twice(self):
        expected_square = 0.0
        folded_ids = {id(value) for value in folded_parameters(self.backbone)}
        for parameter in self.backbone.parameters():
            parameter.grad = torch.ones_like(parameter)
            expected_square += parameter.numel() * (
                2.0 if id(parameter) in folded_ids else 1.0
            )
        observed = clip_effective_backbone_gradients(self.backbone, 1e9)
        self.assertAlmostEqual(float(observed), expected_square**0.5, places=4)


if __name__ == "__main__":
    unittest.main()
