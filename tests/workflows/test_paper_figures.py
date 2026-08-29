import unittest

import numpy as np

from fsrl.analysis.geometry import rank_positions
from fsrl.workflows import (
    paper_figure_contract,
    paper_figure_data,
    paper_figure_replay,
    paper_figures,
)
from fsrl.workflows.paper_figures import (
    DATASET_ORDER,
    REPLAY_CSV_PATH,
    REPLAY_MANIFEST_PATH,
    SUITE_ROOT,
    check_suite,
    file_sha256,
    load_datasets,
    load_json,
    select_exemplar,
    validate_specification,
)


class PaperFigureAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validation = validate_specification()
        cls.specification = cls.validation["specification"]

    def test_static_contract_locks_scope_and_sources(self):
        self.assertTrue(self.validation["passed"], self.validation)
        self.assertTrue(all(check["passed"] for check in self.validation["checks"]))
        self.assertEqual(
            [figure["id"] for figure in self.specification["figures"]],
            [
                "figure_01_group_behavior",
                "figure_02_pair_structure",
                "figure_02h_error_fingerprints",
                "figure_03_global_rankings",
            ],
        )
        self.assertEqual(self.specification["model"]["network_pooling"], "forbidden")
        self.assertIn(
            "paper_q_learning_controls", self.specification["excluded_panels"]
        )
        self.assertIn(
            "meg_measurements_not_generated_by_model",
            self.specification["excluded_panels"],
        )

    def test_public_entrypoint_reexports_split_contracts(self):
        self.assertIs(paper_figures.Dataset, paper_figure_contract.Dataset)
        self.assertIs(paper_figures.load_datasets, paper_figure_data.load_datasets)
        self.assertIs(
            paper_figures.replay_model_subject_pairs,
            paper_figure_replay.replay_model_subject_pairs,
        )

    def test_subject_pair_replay_matches_the_frozen_results(self):
        replay = load_json(REPLAY_MANIFEST_PATH)
        self.assertEqual(replay["output"]["rows"], 2 * 77 * 28)
        self.assertEqual(file_sha256(REPLAY_CSV_PATH), replay["output"]["sha256"])
        self.assertEqual(replay["network_pooling"], "not_performed")
        for seed in (2104, 2105):
            check = replay["seeds"][str(seed)]
            self.assertTrue(check["stored_behavior_exact_match"])
            self.assertLess(check["pair_mean_max_abs_error"], 1e-12)

        protocol, datasets, _ = load_datasets()
        self.assertEqual(tuple(datasets), DATASET_ORDER)
        for dataset in datasets.values():
            self.assertEqual(dataset.pair_accuracy.shape, (77, 28))
            self.assertTrue(np.all(np.isfinite(dataset.pair_accuracy)))

        true_positions = rank_positions(list(protocol.true_order_high_to_low))
        displayed_ids = []
        for dataset in datasets.values():
            index = select_exemplar(dataset, true_positions)
            subject = dataset.subjects[index]
            displayed_ids.append(
                int(subject.get("combined_id", subject.get("subject", index) + 1))
            )
        self.assertEqual(displayed_ids, [3, 7, 8])

    def test_rendered_manifest_accounts_for_every_output(self):
        manifest = load_json(SUITE_ROOT / "manifest.json")
        self.assertEqual(manifest["suite_id"], self.specification["suite_id"])
        self.assertEqual(manifest["network_pooling"], "not_performed")
        self.assertEqual(len(manifest["figures"]), 4)
        for figure in manifest["figures"]:
            self.assertGreater(figure["source_rows"], 0)
            self.assertEqual(
                {entry["path"].rsplit(".", 1)[-1] for entry in figure["files"]},
                {"svg", "pdf", "png", "csv"},
            )
            for entry in figure["files"]:
                path = SUITE_ROOT / entry["path"]
                self.assertEqual(path.stat().st_size, entry["bytes"])
                self.assertEqual(file_sha256(path), entry["sha256"])

    def test_committed_suite_rebuilds_byte_for_byte(self):
        check = check_suite()
        self.assertTrue(check["passed"], check["mismatches"])
        self.assertEqual(check["checked_files"], 17)
