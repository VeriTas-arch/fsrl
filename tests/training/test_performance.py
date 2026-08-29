import unittest

from fsrl.training.performance import benchmark_training_hot_path


class TrainingPerformanceTests(unittest.TestCase):
    def test_cpu_benchmark_reports_hot_path_contract(self):
        result = benchmark_training_hot_path(
            batch_size=2,
            hidden_size=8,
            cue_size=8,
            min_edges=7,
            max_edges=7,
            support_blocks=1,
            warmups=1,
            repeats=2,
            seed=29,
            device="cpu",
            compile_model=False,
        )

        self.assertEqual(result["benchmark_schema_version"], 1)
        self.assertEqual(result["scope"], "engineering_only_not_scientific_evidence")
        self.assertEqual(len(result["measurement"]["seconds"]["samples"]), 2)
        self.assertEqual(result["measurement"]["sampled_edge_counts"], [7, 7])
        self.assertGreater(result["throughput"]["optimizer_steps_per_second"], 0)
        self.assertEqual(
            result["host_synchronization"][
                "pre_backward_metric_materializations_per_step"
            ],
            0,
        )
        self.assertEqual(
            result["host_synchronization"][
                "post_optimizer_metric_materializations_per_step"
            ],
            1,
        )
        self.assertFalse(result["memory"]["available"])
        self.assertIsNone(result["memory"]["peak_allocated_bytes"])

    def test_benchmark_requires_positive_measurement_counts(self):
        with self.assertRaisesRegex(ValueError, "warmups and repeats"):
            benchmark_training_hot_path(warmups=0, repeats=1, device="cpu")


if __name__ == "__main__":
    unittest.main()
