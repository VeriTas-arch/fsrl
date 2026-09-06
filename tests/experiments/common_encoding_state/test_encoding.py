import unittest

import numpy as np

from fsrl.experiments.common_encoding_state.encoding import (
    copula_uniforms,
    draw_streams,
    encode_common,
)
from fsrl.experiments.common_encoding_state.recovery import (
    choices,
    likelihoods,
)
from fsrl.experiments.minimal_learner.data import ModelBatch


def fixture(subjects=512):
    cues = np.asarray(
        [
            [[1, -1, -1, 1]],
            [[-1, 1, 1, -1]],
            [[1, 1, -1, -1]],
            [[1, -1, -1, 1]],
        ],
        dtype=np.float64,
    )
    cues = np.broadcast_to(cues, (4, subjects, 4)).copy()
    signed = np.broadcast_to(
        np.asarray([[-0.5], [0.5], [-0.5], [-0.5]]), (4, subjects)
    ).copy()
    return ModelBatch(
        {
            "support_cues": cues,
            "signed": signed,
            "retention": np.ones_like(signed),
            "local_evidence": np.zeros_like(signed),
        }
    )


class CommonEncodingTests(unittest.TestCase):
    def test_marginals_stay_uniform_and_relation_latent_binds_by_cue(self):
        batch = fixture(20_000)
        streams = draw_streams(batch, np.random.default_rng(31))
        np.testing.assert_array_equal(streams["relation"][0], streams["relation"][3])
        for condition in ("independent", "relation_common", "episode_common"):
            uniforms = copula_uniforms(batch, condition, 0.5, streams)
            self.assertLess(abs(float(np.mean(uniforms < 0.4)) - 0.4), 0.01)

    def test_primary_control_separates_cross_relation_dependence(self):
        batch = fixture(40_000)
        streams = draw_streams(batch, np.random.default_rng(37))
        relation = copula_uniforms(batch, "relation_common", 0.5, streams)
        episode = copula_uniforms(batch, "episode_common", 0.5, streams)
        relation_cross = np.corrcoef(relation[0] < 0.5, relation[2] < 0.5)[0, 1]
        episode_cross = np.corrcoef(episode[0] < 0.5, episode[2] < 0.5)[0, 1]
        self.assertLess(abs(relation_cross), 0.02)
        self.assertGreater(episode_cross, 0.25)

    def test_rho_zero_is_exactly_the_independent_stream(self):
        batch = fixture(8)
        streams = draw_streams(batch, np.random.default_rng(41))
        expected = copula_uniforms(batch, "independent", 0.0, streams)
        for condition in ("relation_common", "episode_common"):
            np.testing.assert_array_equal(
                copula_uniforms(batch, condition, 0.0, streams), expected
            )

    def test_encoder_preserves_codec_and_likelihood_prefers_generator(self):
        batch = fixture(1000)
        streams = draw_streams(batch, np.random.default_rng(43))
        encoded, witness = encode_common(batch, "episode_common", 0.5, streams)
        observed, probability, keys = choices(
            batch, "episode_common", 0.5, streams, (-1, -1 / 3, 1 / 3, 1)
        )
        scores = likelihoods(observed, probability, keys, 0.5)
        self.assertEqual(max(scores, key=scores.__getitem__), "episode_common")
        np.testing.assert_array_equal(
            encoded.arrays["signed"], witness["internal_signed"]
        )

    def test_invalid_or_unbound_streams_fail_closed(self):
        batch = fixture(4)
        streams = draw_streams(batch, np.random.default_rng(47))
        streams["relation"][3, 0] += 1
        with self.assertRaisesRegex(ValueError, "not stable"):
            copula_uniforms(batch, "relation_common", 0.5, streams)


if __name__ == "__main__":
    unittest.main()
