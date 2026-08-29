import inspect
import unittest

from fsrl.evaluation.causal_suite import run_causal_suite
from fsrl.evaluation.cli import parse_args
from fsrl.evaluation.frozen_fast_weight import FrozenEvaluationBackend


class CurrentExecutionDefaultTests(unittest.TestCase):
    def test_high_level_causal_suite_defaults_to_current_batched_schema(self):
        default = (
            inspect.signature(run_causal_suite).parameters["evaluation_backend"].default
        )
        self.assertEqual(default, FrozenEvaluationBackend.BATCHED_SEQUENCE)

    def test_cli_defaults_to_current_batched_schema(self):
        parsed = parse_args(
            ["--checkpoint", "/tmp/model.pth", "--output", "/tmp/result.json"]
        )
        self.assertEqual(
            parsed.evaluation_backend,
            FrozenEvaluationBackend.BATCHED_SEQUENCE.value,
        )

    def test_historical_backend_remains_an_explicit_option(self):
        parsed = parse_args(
            [
                "--checkpoint",
                "/tmp/model.dat",
                "--output",
                "/tmp/result.json",
                "--evaluation-backend",
                FrozenEvaluationBackend.LEGACY_STEPWISE.value,
            ]
        )
        self.assertEqual(
            parsed.evaluation_backend,
            FrozenEvaluationBackend.LEGACY_STEPWISE.value,
        )
