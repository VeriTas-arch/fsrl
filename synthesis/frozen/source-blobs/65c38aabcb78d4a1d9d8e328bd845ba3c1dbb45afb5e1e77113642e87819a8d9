import copy
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import fsrl.global_policy_field_fingerprint_replication as fingerprint
from fsrl.curvature_gate_pilot import load_json
from fsrl.global_policy_field_fingerprint_replication import (
    artifact_lock_document,
    backbone_training_config,
    cross_seed_decision,
    mandatory_seeds,
    validate_artifacts,
    validate_sources,
)


class GlobalPolicyFieldFingerprintReplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specification = load_json(
            "benchmarks/global_policy_field_fingerprint_replication_v1.json"
        )

    def test_mandatory_seed_contract_is_exact(self):
        self.assertEqual(mandatory_seeds(self.specification), (2106, 2107))

        replacements = (
            [2106],
            [2106, 2108],
            [2104, 2107],
            [2106, 2107, 2108],
        )
        for seeds in replacements:
            with self.subTest(seeds=seeds):
                changed = copy.deepcopy(self.specification)
                changed["network_seed_contract"]["mandatory_seeds"] = seeds
                with self.assertRaisesRegex(RuntimeError, "2106 and 2107"):
                    mandatory_seeds(changed)

    def test_backbone_training_config_strips_documentation_fields(self):
        observed = backbone_training_config(self.specification, 2106)
        self.assertEqual(observed.seed, 2106)
        self.assertEqual(observed.outer_steps, 1000)
        self.assertEqual(observed.save_every, 1000)
        self.assertEqual(observed.subject_encoding_mode, "stable_omission")

        config = asdict(observed)
        for documentation_field in (
            "seeds",
            "held_out_graph",
            "architecture",
            "checkpoint_selection",
            "local_gain_adaptation",
        ):
            self.assertNotIn(documentation_field, config)

        with self.assertRaisesRegex(ValueError, "not registered"):
            backbone_training_config(self.specification, 2104)

    def test_formal_paths_are_canonical_except_tmp_result_replay(self):
        parsed = fingerprint.parse_args(["train-artifacts"])
        fingerprint._canonical_paths(parsed)

        replay = fingerprint.parse_args(
            ["evaluate", "--result", "/tmp/fingerprint-replay.json"]
        )
        fingerprint._canonical_paths(replay)

        train_replay = copy.copy(parsed)
        train_replay.result = Path("/tmp/fingerprint-replay.json")
        with self.assertRaisesRegex(RuntimeError, "only for evaluate"):
            fingerprint._canonical_paths(train_replay)

        changed_output = copy.copy(parsed)
        changed_output.output_root = Path("/tmp/alternate-output")
        with self.assertRaisesRegex(RuntimeError, "canonical output_root"):
            fingerprint._canonical_paths(changed_output)

        changed_result = fingerprint.parse_args(
            ["evaluate", "--result", "/var/tmp/alternate-result.json"]
        )
        with self.assertRaisesRegex(RuntimeError, "only under /tmp"):
            fingerprint._canonical_paths(changed_result)

        tmp_directory = fingerprint.parse_args(["evaluate", "--result", "/tmp"])
        with self.assertRaisesRegex(RuntimeError, "file below /tmp"):
            fingerprint._canonical_paths(tmp_directory)

    def test_pushed_freeze_requires_clean_shared_dev_head(self):
        freeze_path = fingerprint.DEFAULT_SPECIFICATION_PATH

        def passing_git(*arguments):
            lookup = {
                ("branch", "--show-current"): "dev",
                ("rev-parse", "HEAD"): "head-sha",
                ("rev-parse", "origin/dev"): "head-sha",
                ("status", "--porcelain", "--untracked-files=all"): "",
            }
            return lookup.get(arguments, "")

        with patch.object(fingerprint, "_git", side_effect=passing_git):
            result = fingerprint.require_pushed_freeze((freeze_path,))
        self.assertTrue(result["worktree_clean"])
        self.assertEqual(result["head"], result["origin_dev"])

        def stale_git(*arguments):
            if arguments == ("rev-parse", "origin/dev"):
                return "stale-origin"
            return passing_git(*arguments)

        with (
            patch.object(fingerprint, "_git", side_effect=stale_git),
            self.assertRaisesRegex(RuntimeError, "HEAD equal to origin/dev"),
        ):
            fingerprint.require_pushed_freeze((freeze_path,))

    @staticmethod
    def _seed_result(
        *,
        a="material_positive",
        q_shape="material_positive",
        interaction="material_negative",
        r="unresolved",
        c_a="material_positive",
        c_shape="material_positive",
        d_lower95=0.02,
        competence=True,
        integrity=True,
    ):
        return {
            "qualification": {"passed": competence},
            "integrity": {"passed": integrity},
            "statistics": {
                "summaries": {
                    "D": {"bootstrap": {"lower95": d_lower95}},
                },
                "statuses": {
                    "A": a,
                    "Q_shape": q_shape,
                    "I": interaction,
                    "R": r,
                    "C_A": c_a,
                    "C_shape": c_shape,
                },
            },
        }

    def test_cross_seed_decision_requires_all_three_links_in_both_networks(self):
        seeds = {
            str(seed): self._seed_result()
            for seed in mandatory_seeds(self.specification)
        }
        decision = cross_seed_decision(self.specification, seeds)

        self.assertEqual(decision["outcome"], "replicated_field_fingerprint")
        self.assertEqual(decision["links"]["A"]["status"], "replicated")
        self.assertEqual(decision["links"]["Q_shape"]["status"], "replicated")
        self.assertEqual(decision["links"]["I"]["status"], "replicated")
        self.assertEqual(decision["network_population_inference"], "not_performed")

    def test_both_unresolved_is_heterogeneous_or_unresolved(self):
        seeds = {
            str(seed): self._seed_result(a="unresolved")
            for seed in mandatory_seeds(self.specification)
        }
        decision = cross_seed_decision(self.specification, seeds)

        self.assertEqual(
            decision["links"]["A"]["status"], "heterogeneous_or_unresolved"
        )
        self.assertEqual(decision["outcome"], "heterogeneous_or_unresolved_fingerprint")

    def test_same_resolved_nonexpected_status_is_not_replicated(self):
        seeds = {
            str(seed): self._seed_result(a="equivalent")
            for seed in mandatory_seeds(self.specification)
        }
        decision = cross_seed_decision(self.specification, seeds)

        self.assertEqual(decision["links"]["A"]["status"], "not_replicated")
        self.assertEqual(decision["outcome"], "field_fingerprint_not_replicated")

    def test_competence_integrity_and_anchor_precede_fingerprint(self):
        passing = {
            str(seed): self._seed_result()
            for seed in mandatory_seeds(self.specification)
        }

        competence_failure = copy.deepcopy(passing)
        competence_failure["2106"]["qualification"]["passed"] = False
        competence_failure["2106"]["statistics"]["summaries"]["D"]["bootstrap"][
            "lower95"
        ] = -1.0
        self.assertEqual(
            cross_seed_decision(self.specification, competence_failure)["outcome"],
            "noninterpretable_competence_or_integrity_failure",
        )

        integrity_failure = copy.deepcopy(passing)
        integrity_failure["2107"]["integrity"]["passed"] = False
        self.assertEqual(
            cross_seed_decision(self.specification, integrity_failure)["outcome"],
            "noninterpretable_competence_or_integrity_failure",
        )

        premise_failure = copy.deepcopy(passing)
        premise_failure["2107"]["statistics"]["summaries"]["D"]["bootstrap"][
            "lower95"
        ] = 0.0
        self.assertEqual(
            cross_seed_decision(self.specification, premise_failure)["outcome"],
            "premise_not_confirmed",
        )

    def test_secondary_boundaries_neither_gate_nor_rescue_primary_outcome(self):
        secondary_disagreement = {
            "2106": self._seed_result(
                r="material_positive", c_a="equivalent", c_shape="material_negative"
            ),
            "2107": self._seed_result(
                r="material_negative", c_a="unresolved", c_shape="equivalent"
            ),
        }
        self.assertEqual(
            cross_seed_decision(self.specification, secondary_disagreement)["outcome"],
            "replicated_field_fingerprint",
        )

        failed_primary = {
            "2106": self._seed_result(a="equivalent"),
            "2107": self._seed_result(a="equivalent"),
        }
        self.assertEqual(
            cross_seed_decision(self.specification, failed_primary)["outcome"],
            "field_fingerprint_not_replicated",
        )

    def test_registered_integrity_failure_does_not_skip_second_seed(self):
        artifact_validation = {
            "lock": {
                "artifacts": {
                    str(seed): {
                        "checkpoint": {
                            "path": f"output/seed-{seed}/net.dat",
                            "sha256": f"sha-{seed}",
                        }
                    }
                    for seed in mandatory_seeds(self.specification)
                }
            }
        }
        calls = []

        def fake_analyze(specification, seed, artifacts):
            self.assertIs(specification, self.specification)
            self.assertIs(artifacts, artifact_validation)
            calls.append(seed)
            if seed == 2106:
                raise fingerprint.NonInterpretableEstimate("registered zero norm")
            return self._seed_result()

        with patch.object(fingerprint, "analyze_seed", side_effect=fake_analyze):
            result = fingerprint.evaluate_replication(
                self.specification, {"passed": True}, artifact_validation, {}
            )

        self.assertEqual(calls, [2106, 2107])
        self.assertEqual(
            result["seeds"]["2106"]["integrity"]["failure_type"],
            "registered_noninterpretable_estimate",
        )
        self.assertEqual(
            result["decision"]["outcome"],
            "noninterpretable_competence_or_integrity_failure",
        )

    @staticmethod
    def _write_dummy_backbones(output_root):
        for seed in (2106, 2107):
            backbone = output_root / f"seed-{seed}" / "backbone"
            backbone.mkdir(parents=True)
            (backbone / "net.dat").write_bytes(f"checkpoint-{seed}".encode())
            (backbone / "config.json").write_text("{}\n", encoding="utf-8")
            (backbone / "train_log.jsonl").write_text("{}\n", encoding="utf-8")
            (backbone / "replication_manifest.json").write_text(
                "{}\n", encoding="utf-8"
            )

    def _artifact_fixture(self, root):
        specification_path = root / "specification.json"
        implementation_lock_path = root / "implementation.lock.json"
        artifact_lock_path = root / "artifact.lock.json"
        output_root = root / "output"
        specification_path.write_text("{}\n", encoding="utf-8")
        implementation_lock_path.write_text("{}\n", encoding="utf-8")
        self._write_dummy_backbones(output_root)
        with (
            patch.object(fingerprint, "ROOT", root),
            patch.object(fingerprint, "_validate_complete_backbone"),
        ):
            document = artifact_lock_document(
                self.specification,
                specification_path,
                implementation_lock_path,
                output_root,
            )
        artifact_lock_path.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return (
            specification_path,
            implementation_lock_path,
            artifact_lock_path,
            output_root,
            document,
        )

    def test_artifact_lock_has_exact_seeds_and_backbone_only_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            *_, document = self._artifact_fixture(root)

        self.assertEqual(set(document["artifacts"]), {"2106", "2107"})
        expected = {
            "checkpoint",
            "backbone_config",
            "backbone_log",
            "backbone_manifest",
        }
        for seed in ("2106", "2107"):
            self.assertEqual(set(document["artifacts"][seed]), expected)
            self.assertNotIn("gain", document["artifacts"][seed])
            self.assertNotIn("local_log", document["artifacts"][seed])

    def test_artifact_validation_rejects_wrong_seed_or_artifact_keys(self):
        mutations = (
            lambda document: document["artifacts"].update(
                {"2104": copy.deepcopy(document["artifacts"]["2106"])}
            ),
            lambda document: document["artifacts"].pop("2107"),
            lambda document: document["artifacts"]["2106"].update(
                {"gain": copy.deepcopy(document["artifacts"]["2106"]["checkpoint"])}
            ),
        )
        for mutate in mutations:
            with (
                self.subTest(mutation=mutate),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                (
                    specification_path,
                    implementation_lock_path,
                    artifact_lock_path,
                    output_root,
                    document,
                ) = self._artifact_fixture(root)
                mutate(document)
                artifact_lock_path.write_text(
                    json.dumps(document, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                with (
                    patch.object(fingerprint, "ROOT", root),
                    patch.object(fingerprint, "_validate_complete_backbone"),
                    self.assertRaisesRegex(RuntimeError, "artifact.*(seed|keys)"),
                ):
                    validate_artifacts(
                        self.specification,
                        specification_path,
                        implementation_lock_path,
                        artifact_lock_path,
                        output_root,
                    )

    def test_source_validation_hashes_but_does_not_load_discovery_artifacts(self):
        specification_path = Path(
            "benchmarks/global_policy_field_fingerprint_replication_v1.json"
        ).resolve()
        implementation_lock_path = Path("/tmp/fingerprint-implementation-lock.json")
        implementation_sources = {
            name: {"path": path, "sha256": f"implementation-{name}"}
            for name, path in fingerprint.REQUIRED_IMPLEMENTATION_SOURCE_PATHS.items()
        }
        reused_sources = {
            name: {"path": path, "sha256": f"reused-{name}"}
            for name, path in fingerprint.REQUIRED_REUSED_SOURCE_PATHS.items()
        }
        implementation_lock = {
            "replication_specification_sha256": "specification-sha256",
            "implementation_sources": implementation_sources,
            "reused_frozen_sources": reused_sources,
        }
        expected_hashes = {
            (fingerprint.ROOT / registration["path"]).resolve(): registration["sha256"]
            for registration in self.specification["registered_sources"].values()
        }
        expected_hashes[specification_path] = "specification-sha256"
        for registration in (
            *implementation_sources.values(),
            *reused_sources.values(),
        ):
            expected_hashes[(fingerprint.ROOT / registration["path"]).resolve()] = (
                registration["sha256"]
            )

        def fake_load(path):
            resolved = Path(path).resolve()
            if resolved == specification_path:
                return self.specification
            if resolved == implementation_lock_path:
                return implementation_lock
            raise AssertionError(f"unexpected JSON load: {resolved}")

        def fake_sha256(path):
            resolved = Path(path).resolve()
            try:
                return expected_hashes[resolved]
            except KeyError as error:
                raise AssertionError(f"unexpected source hash: {resolved}") from error

        with (
            patch.object(fingerprint, "load_json", side_effect=fake_load) as loader,
            patch.object(fingerprint, "file_sha256", side_effect=fake_sha256),
        ):
            result = validate_sources(specification_path, implementation_lock_path)

        self.assertTrue(result["passed"])
        self.assertEqual(loader.call_count, 2)

    def test_source_validation_rejects_missing_extra_or_rebound_sources(self):
        specification_path = Path(
            "benchmarks/global_policy_field_fingerprint_replication_v1.json"
        ).resolve()
        implementation_lock_path = Path("/tmp/fingerprint-invalid-lock.json")
        base = {
            "replication_specification_sha256": "specification-sha256",
            "implementation_sources": {
                name: {"path": path, "sha256": name}
                for name, path in fingerprint.REQUIRED_IMPLEMENTATION_SOURCE_PATHS.items()
            },
            "reused_frozen_sources": {
                name: {"path": path, "sha256": name}
                for name, path in fingerprint.REQUIRED_REUSED_SOURCE_PATHS.items()
            },
        }
        mutations = (
            lambda lock: lock["implementation_sources"].pop("replication_runner"),
            lambda lock: lock["reused_frozen_sources"].update(
                {"extra": {"path": "fsrl/model.py", "sha256": "extra"}}
            ),
            lambda lock: lock["implementation_sources"]["formal_runtime"].update(
                {"path": "fsrl/model.py"}
            ),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                lock = copy.deepcopy(base)
                mutate(lock)

                def fake_load(path, current_lock=lock):
                    if Path(path).resolve() == specification_path:
                        return self.specification
                    return current_lock

                with (
                    patch.object(fingerprint, "load_json", side_effect=fake_load),
                    self.assertRaisesRegex(RuntimeError, "source (lock|path)"),
                ):
                    validate_sources(specification_path, implementation_lock_path)


if __name__ == "__main__":
    unittest.main()
