import unittest

import torch

from fsrl.core import PlasticRNNState, RetroModelConfig, RetroModulRNN
from fsrl.core.config import TrainConfig


class ModelBoundaryTests(unittest.TestCase):
    def test_typed_and_legacy_configs_initialize_identical_models(self):
        legacy = TrainConfig(bs=2, hs=5, cs=3).to_model_dict()
        typed = RetroModelConfig.from_legacy_mapping(legacy)
        torch.manual_seed(73)
        legacy_net = RetroModulRNN(legacy, device="cpu")
        torch.manual_seed(73)
        typed_net = RetroModulRNN(typed, device="cpu")
        self.assertEqual(typed_net.model_config, typed)
        self.assertEqual(typed.to_legacy_mapping()["hs"], legacy["hs"])
        for name, value in legacy_net.state_dict().items():
            self.assertTrue(torch.equal(value, typed_net.state_dict()[name]))

    def test_typed_state_is_exactly_equivalent_to_legacy_zero_methods(self):
        config = RetroModelConfig(
            input_size=11, hidden_size=4, output_size=2, batch_size=3
        )
        net = RetroModulRNN(config, device="cpu")
        state = net.initial_state(3)
        self.assertIsInstance(state, PlasticRNNState)
        self.assertTrue(torch.equal(state.hidden, net.initialZeroState(3)))
        self.assertTrue(torch.equal(state.eligibility, net.initialZeroET(3)))
        self.assertTrue(
            torch.equal(state.fast_weights, net.initialZeroPlasticWeights(3))
        )

    def test_typed_config_rejects_invalid_dimensions(self):
        with self.assertRaisesRegex(ValueError, "hidden_size"):
            RetroModelConfig(input_size=11, hidden_size=0, output_size=2, batch_size=3)
        with self.assertRaisesRegex(KeyError, "missing keys"):
            RetroModelConfig.from_legacy_mapping({"hs": 4})

    def test_typed_refactor_preserves_forward_equations_and_state_keys(self):
        config = RetroModelConfig(
            input_size=11, hidden_size=4, output_size=2, batch_size=2
        )
        torch.manual_seed(79)
        net = RetroModulRNN(config, device="cpu")
        state = net.initial_state(2)
        output = net(
            torch.randn(2, config.input_size),
            state.hidden,
            state.eligibility,
            state.fast_weights,
        )
        self.assertEqual(output[0].shape, (2, 2))
        self.assertEqual(output[3].shape, (2, 4))
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
