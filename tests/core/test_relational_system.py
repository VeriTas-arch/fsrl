import unittest

import numpy as np
import torch

from fsrl.core import (
    ConjunctiveLocalTrace,
    GlobalLocalRelationalSystem,
    RelationalIntervention,
    RetroModulRNN,
)
from fsrl.core.config import TrainConfig
from fsrl.evaluation import FrozenFastWeightEvaluator
from fsrl.tasks.evidence import broader_local_admission
from fsrl.tasks.registered_protocol import load_ranking_protocol


class RelationalSystemTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.config = TrainConfig(
            bs=2,
            hs=5,
            cs=6,
            triallen=4,
            nbtraintrials=32,
            nbtesttrials=28,
            nbcues_min=8,
            nbcues_max=8,
        )
        self.backbone = RetroModulRNN(self.config.to_model_dict(), device="cpu")
        self.protocol = load_ranking_protocol()
        self.evaluator = FrozenFastWeightEvaluator(
            self.backbone,
            self.config,
            self.protocol,
            cue_seed=5,
            support_seed=11,
            subject_encoding_mode="none",
        )
        self.local = ConjunctiveLocalTrace(
            self.config.cs, initial_gain=0.2, device="cpu"
        )
        self.system = GlobalLocalRelationalSystem(self.backbone, self.local)

    def _support_inputs(self, trial_index):
        trials = [
            schedule[trial_index] for schedule in self.evaluator.support_schedules
        ]
        left = np.asarray([trial.left_item for trial in trials], dtype=np.int64)
        right = np.asarray([trial.right_item for trial in trials], dtype=np.int64)
        signed = np.asarray(
            [trial.signed_magnitude for trial in trials], dtype=np.float32
        )
        trial_time = (
            trial_index
            / max(1, self.protocol.support_trials - 1)
            * self.evaluator.test_time_value
        )
        sequence = torch.stack(
            [
                self.evaluator._step_inputs(
                    left,
                    right,
                    signed,
                    numstep=step,
                    time_value=trial_time,
                    support_trial=True,
                )
                for step in range(self.config.triallen)
            ]
        )
        return sequence, sequence[0, :, : 2 * self.config.cs], torch.from_numpy(signed)

    def _query_inputs(self, left, right):
        left = np.asarray(left, dtype=np.int64)
        right = np.asarray(right, dtype=np.int64)
        signed = np.zeros(self.config.bs, dtype=np.float32)
        sequence = torch.stack(
            [
                self.evaluator._step_inputs(
                    left,
                    right,
                    signed,
                    numstep=step,
                    time_value=self.evaluator.test_time_value,
                    support_trial=False,
                )
                for step in range(self.config.triallen)
            ]
        )
        return sequence, sequence[0, :, : 2 * self.config.cs]

    def test_global_rollout_and_query_match_frozen_evaluator(self):
        with torch.no_grad():
            state = self.system.initialize_episode(self.config.bs)
            expected_weights = self.evaluator.initialize_fast_weights()
            torch.testing.assert_close(
                state.global_fast_weights, expected_weights, rtol=0.0, atol=0.0
            )
            for trial_index in range(self.protocol.support_trials):
                sequence, pair_cues, signed = self._support_inputs(trial_index)
                state = self.system.support_trial(
                    state,
                    sequence,
                    pair_cues=pair_cues,
                    local_signed_value=signed,
                )
                expected_weights = self.evaluator.advance_support_trial(
                    expected_weights, trial_index
                )
                torch.testing.assert_close(
                    state.global_fast_weights,
                    expected_weights,
                    rtol=0.0,
                    atol=0.0,
                )

            left = (0, 2)
            right = (1, 5)
            sequence, pair_cues = self._query_inputs(left, right)
            readout = self.system.query(state, sequence, pair_cues=pair_cues)
            schedules = tuple(((left[i], right[i]),) for i in range(self.config.bs))
            expected = self.evaluator.readout_logits(expected_weights, schedules)
            expected_margins = torch.tensor(
                [expected[i][(left[i], right[i])] for i in range(self.config.bs)]
            )
            observed_margins = readout.global_logits[:, 1] - readout.global_logits[:, 0]
            torch.testing.assert_close(observed_margins, expected_margins)

            local_off = self.system.query(
                state,
                sequence,
                pair_cues=pair_cues,
                intervention=RelationalIntervention.LOCAL_OFF,
            )
            torch.testing.assert_close(local_off.logits, readout.global_logits)
            torch.testing.assert_close(
                readout.logits[:, 1] - readout.logits[:, 0],
                observed_margins + readout.local_correction[:, 0],
            )

            global_off = self.system.query(
                state,
                sequence,
                pair_cues=pair_cues,
                intervention=RelationalIntervention.GLOBAL_OFF,
            )
            zero_expected = self.evaluator.readout_logits(
                torch.zeros_like(expected_weights), schedules
            )
            zero_margins = torch.tensor(
                [zero_expected[i][(left[i], right[i])] for i in range(self.config.bs)]
            )
            torch.testing.assert_close(
                global_off.global_logits[:, 1] - global_off.global_logits[:, 0],
                zero_margins,
            )

    def test_differential_admission_preserves_registered_equation(self):
        global_access = np.asarray([0.0, 1.0, 0.25])
        reliability = np.asarray([0.4, 0.2, 0.8])
        observed = broader_local_admission(global_access, reliability)
        np.testing.assert_array_equal(
            observed,
            global_access + (1.0 - global_access) * reliability,
        )

    def test_write_off_keeps_global_state_and_retains_local_state(self):
        state = self.system.initialize_episode(self.config.bs)
        sequence, pair_cues, signed = self._support_inputs(0)
        updated = self.system.support_trial(
            state,
            sequence,
            pair_cues=pair_cues,
            local_signed_value=signed,
            global_write=False,
        )
        torch.testing.assert_close(
            updated.global_fast_weights, state.global_fast_weights
        )
        self.assertGreater(float(torch.linalg.vector_norm(updated.local_trace)), 0.0)
