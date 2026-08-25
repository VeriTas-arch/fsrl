import unittest

import numpy as np
import torch

from fsrl import functional_fast_weight_latent_sufficiency as audit


class FunctionalFastWeightLatentSufficiencyTests(unittest.TestCase):
    def test_centered_basis_is_orthonormal_and_spans_centered_vectors(self):
        basis = audit.centered_basis()
        np.testing.assert_allclose(basis.T @ basis, np.eye(7), atol=1e-12)
        np.testing.assert_allclose(np.ones(8) @ basis, 0.0, atol=1e-12)
        rng = np.random.default_rng(1)
        values = rng.normal(size=(11, 8))
        values -= np.mean(values, axis=1, keepdims=True)
        np.testing.assert_allclose(values @ basis @ basis.T, values, atol=1e-12)

    def test_relative_primal_ridge_recovers_low_noise_mapping(self):
        rng = np.random.default_rng(2)
        features = rng.normal(size=(500, 5)).astype(np.float32)
        coefficients = rng.normal(size=(5, 3)).astype(np.float32)
        targets = features @ coefficients
        fitted, penalty = audit._primal_ridge(
            torch.as_tensor(features), torch.as_tensor(targets)
        )
        self.assertGreater(penalty, 0.0)
        relative_error = np.linalg.norm(fitted.numpy() - coefficients) / np.linalg.norm(
            coefficients
        )
        self.assertLess(relative_error, 1e-3)

    def test_dual_ridge_uses_stable_system_and_returns_input_dtype(self):
        rng = np.random.default_rng(21)
        base = rng.normal(size=(40, 6)).astype(np.float32)
        features = np.concatenate((base, base + 1e-7), axis=1)
        coefficients = rng.normal(size=(12, 3)).astype(np.float32)
        targets = features @ coefficients
        fitted, penalty = audit._dual_ridge(
            torch.as_tensor(features), torch.as_tensor(targets)
        )
        self.assertEqual(fitted.dtype, torch.float32)
        self.assertGreater(penalty, 0.0)
        prediction_error = np.mean((features @ fitted.numpy() - targets) ** 2)
        self.assertLess(prediction_error, 1e-5)

    def test_rank_seven_projection_reconstructs_full_centered_contribution(self):
        rng = np.random.default_rng(3)
        contribution = rng.normal(size=(200, 7))
        _, _, right = np.linalg.svd(contribution, full_matrices=False)
        directions = right.T
        reconstructed = (contribution @ directions) @ directions.T
        np.testing.assert_allclose(reconstructed, contribution, atol=1e-12)

    def test_prediction_interface_rank_seven_equals_full_oracle(self):
        rng = np.random.default_rng(31)
        potentials = rng.normal(size=(4, 8))
        potentials -= np.mean(potentials, axis=1, keepdims=True)
        episode = audit.FunctionalEpisode(
            potentials=potentials,
            fields=np.zeros((4, 28)),
            evidence=rng.normal(size=(3, 8)),
            functional_state=rng.normal(size=(4, 6)).astype(np.float32),
            loo_potentials=potentials.copy(),
            loo_fields=np.zeros((4, 28)),
            loo_evidence=rng.normal(size=(3, 8)),
            loo_functional_state=rng.normal(size=(4, 6)).astype(np.float32),
            loo_relation=(0, 1),
        )
        output_directions, _ = np.linalg.qr(rng.normal(size=(7, 7)))
        predictor = audit.FunctionalPredictor(
            potential_to_state=rng.normal(size=(8, 6)).astype(np.float32),
            baseline=rng.normal(size=(16, 7)).astype(np.float32),
            full_state=rng.normal(size=(6, 7)).astype(np.float32),
            output_directions=output_directions.astype(np.float32),
            centered_basis=audit.centered_basis().astype(np.float32),
        )
        predictions, _, _ = audit.predict_updates(
            [episode], predictor, loo=False, device="cpu"
        )
        np.testing.assert_allclose(
            predictions["rank_7"], predictions["full_P"], atol=2e-5
        )

    def test_episode_error_keeps_episode_as_resampling_unit(self):
        target = np.zeros((5, 8))
        prediction = np.zeros_like(target)
        prediction[:2] = 1.0
        prediction[2:] = 2.0
        episode_index = np.asarray([0, 0, 1, 1, 1])
        np.testing.assert_allclose(
            audit._episode_errors(prediction, target, episode_index), [1.0, 4.0]
        )


if __name__ == "__main__":
    unittest.main()
