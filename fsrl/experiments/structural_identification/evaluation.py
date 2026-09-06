"""Locked generic, Liu and support-manipulation evaluation."""

import numpy as np
import torch

from fsrl.experiments.adaptive_plasticity.data import model_tensors, relation_slots
from fsrl.experiments.minimal_learner.data import ModelBatch
from fsrl.experiments.minimal_learner.locks import validate_complete
from fsrl.experiments.minimal_learner.training import compiled, runtime
from fsrl.experiments.quantized_learner.encoding import canonical_addresses
from fsrl.infra.provenance import load_json, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun
from fsrl.tasks.protocol import ordered_pairs
from fsrl.tasks.protocol_catalog import load_registered_protocol

from .execution import fitted_parameters
from .inputs import load_inputs, read_batch, save_arrays
from .measurement import REFERENCE
from .model import encode, make_model
from .observation import observe, record, replicated_cycles, sample_choices
from .protocol import RUN_ROOT, cohort_specification, specification


def identity(row):
    return f"{row['seed']}-{row['structure']}-{row['schedule']}"


def load_runner(row):
    model = make_model(row["schedule"], "cuda")
    model.load_state_dict(
        {
            key: torch.tensor(value, device="cuda", dtype=torch.float32)
            for key, value in row["raw_parameters"].items()
        }
    )
    return compiled(model.requires_grad_(False).eval())


def predict(runner, batch):
    with torch.no_grad():
        margins, state, efficacy = runner(*model_tensors(batch, "cuda"))
    if efficacy.numel() != 0:
        raise RuntimeError("unregistered relation efficacy state")
    return margins.cpu().numpy(), state.cpu().numpy()


def mean_interval(values, seed=7900001) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or not len(values):
        return {"mean": None, "lower": None, "upper": None}
    counts = np.random.default_rng(seed).multinomial(
        len(values), np.full(len(values), 1 / len(values)), size=10000
    )
    lower, upper = np.quantile(counts @ values / len(values), [0.025, 0.975])
    return {"mean": float(values.mean()), "lower": float(lower), "upper": float(upper)}


def routing_control(batch, seed):
    """One cue-address permutation of admitted teaching streams per subject."""
    a = dict(batch.arrays)
    signed = np.array(a["signed"], copy=True)
    slots = relation_slots(a["support_cues"])
    _, orientation = canonical_addresses(a["support_cues"])
    canonical = orientation * signed
    rng = np.random.default_rng(seed)
    for s in range(signed.shape[1]):
        admitted = np.unique(slots[a["retention"][:, s] == 1, s])
        order = rng.permutation(admitted)
        for target, source in zip(admitted, order, strict=True):
            target_rows, source_rows = slots[:, s] == target, slots[:, s] == source
            signed[target_rows, s] = (
                canonical[source_rows, s] * orientation[target_rows, s]
            )
    return ModelBatch({**a, "signed": signed})


def evaluate_generic() -> dict:
    runtime()
    fits, inputs = fitted_parameters(), load_inputs()
    results = {}
    for row in fits:
        name = identity(row)
        directory = RUN_ROOT / "generic" / name
        if directory.exists():
            validate_complete(directory)
            results[name] = load_json(directory / "result.json")
            continue
        runner = load_runner(row)
        with ProspectiveRun.start(
            directory,
            workflow_id="structural_identification_v1",
            execution_id=f"generic-{name}",
            producer={"module": __name__},
            resolved_config=row,
        ):
            margins, shuffled, signs, learned = [], [], [], []
            for index, source in enumerate(inputs["generic"]):
                base, extra = read_batch(source)
                batch = encode(base, row["structure"], extra["uniforms"])
                margins.append(predict(runner, batch)[0])
                shuffled.append(
                    predict(runner, routing_control(batch, 1330011 + index))[0]
                )
                signs.append(2 * base.arrays["targets"] - 1)
                learned.append(base.arrays["learned"])
            intact, shuffled, signs, learned = map(
                np.concatenate, (margins, shuffled, signs, learned)
            )
            correct = intact * signs > 0
            result: dict = {
                key: mean_interval(
                    (correct * mask).sum(axis=1) / mask.sum(axis=1),
                    7900001 + row["seed"],
                )
                for key, mask in (("learned", learned), ("nonlearned", ~learned))
            }
            result["binding"] = mean_interval(
                correct.mean(axis=1) - (shuffled * signs > 0).mean(axis=1),
                7900001 + row["seed"],
            )
            result["competence"] = all(
                result[key]["lower"] > 0.5 for key in ("learned", "nonlearned")
            )
            result["binding_pass"] = result["binding"]["lower"] > 0
            save_arrays(
                directory / "outputs.npz",
                margins=intact,
                shuffled=shuffled,
                signs=signs,
                learned=learned,
            )
            write_json_exclusive(directory / "result.json", result)
            results[name] = result
        print("generic", name, result, flush=True)
    return results


def scalar_identity(margins):
    matrix = np.zeros((len(margins), 8, 8), dtype=np.float64)
    for i, pair in enumerate(ordered_pairs(8)):
        matrix[:, pair[0], pair[1]] = margins[:, i]
    residual = (
        matrix[:, :, :, None]
        + matrix[:, None, :, :]
        + matrix.transpose(0, 2, 1)[:, :, None, :]
    )
    maximum = float(np.max(np.abs(residual)))
    if maximum > 1e-4:
        raise RuntimeError("scalar margin closure failed")
    return maximum


def evaluate_liu() -> dict:
    runtime()
    fits, inputs = fitted_parameters(), load_inputs()
    human = load_json(REFERENCE)
    protocol = load_registered_protocol("liu_v2")
    runners = {identity(row): load_runner(row) for row in fits}
    for index, source in enumerate(inputs["liu"]):
        directory = RUN_ROOT / "liu" / f"cohort-{index:03}"
        if directory.exists():
            validate_complete(directory)
            continue
        with ProspectiveRun.start(
            directory,
            workflow_id="structural_identification_v1",
            execution_id=f"liu-{index}",
            producer={"module": __name__, "input": source},
            resolved_config=cohort_specification(index),
        ):
            base, extra = read_batch(source)
            encoded = {
                structure: encode(base, structure, extra["uniforms"])
                for structure in specification()["design"]["structures"]
            }
            results, arrays = {}, {}
            choice_seed = cohort_specification(index)["evaluation"]["liu"][
                "choice_seed"
            ]
            for row in fits:
                name = identity(row)
                margins, weights = predict(runners[name], encoded[row["structure"]])
                residual = scalar_identity(margins)
                choices, canonical = sample_choices(margins, protocol, choice_seed)
                results[name] = {
                    "common": record(choices, protocol, human["references"]),
                    "legacy": record(
                        choices,
                        protocol,
                        human["legacy_reference"],
                        legacy_margins=canonical,
                    ),
                    "replicated_cycle_prevalence": float(
                        np.mean(replicated_cycles(choices) > 0)
                    ),
                    "scalar_closure_max": residual,
                }
                arrays.update(
                    {
                        f"{name}__choices": np.packbits(choices, axis=-1),
                        f"{name}__margins": margins,
                        f"{name}__w": weights,
                    }
                )
            save_arrays(directory / "outputs.npz", **arrays)
            write_json_exclusive(
                directory / "result.json", {"cohort": index, "fits": results}
            )
        if index % 10 == 0:
            print("Liu cohort", index, "complete", flush=True)
    return {"fits": len(fits), "cohorts": len(inputs["liu"])}


def manipulation_record(margins, batch, choice_seed):
    from itertools import combinations

    from scipy.special import expit

    targets = batch.arrays["targets"]
    signs = 2 * targets - 1
    correct = margins * signs > 0
    pairs = batch.arrays["query_pairs"]
    # Chain graph labels identify the bridge only to the evaluator.
    support = batch.arrays["support_pairs"]
    classes, cycles = [], []
    rng = np.random.default_rng(choice_seed)
    chosen_left = rng.random((len(margins), 10, 28)) < expit(margins[:, None] / 0.25)
    for s in range(len(margins)):
        canonical = np.empty((1, 10, 28), dtype=bool)
        lookup = {pair: i for i, pair in enumerate(combinations(range(8), 2))}
        for j, pair in enumerate(pairs[s]):
            canonical[0, :, lookup[tuple(sorted(pair))]] = (
                chosen_left[s, :, j] if pair[0] < pair[1] else ~chosen_left[s, :, j]
            )
        true_winners = np.where(targets[s], pairs[s, :, 0], pairs[s, :, 1])
        wins = np.bincount(true_winners, minlength=8)
        order = tuple(np.argsort(-wins).tolist())
        learned = tuple(dict.fromkeys(tuple(sorted(pair)) for pair in support[s]))
        from fsrl.tasks.protocol import RankingProtocol

        protocol = RankingProtocol(
            "structure-manipulation",
            tuple(map(str, range(8))),
            order,
            learned,
            4,
            10,
            {},
        )
        observed = observe(canonical, protocol)
        classes.append(observed["subjects"][0]["ranking_class"])
        cycles.append(replicated_cycles(canonical)[0])
    bridge_mask = np.zeros_like(correct)
    cross_mask = np.zeros_like(correct)
    for s in range(len(margins)):
        true_winners = np.where(targets[s], pairs[s, :, 0], pairs[s, :, 1])
        order = np.argsort(-np.bincount(true_winners, minlength=8))
        first_half = set(order[:4].tolist())
        for j, pair in enumerate(pairs[s]):
            bridge_mask[s, j] = set(pair) == set(order[3:5])
            cross_mask[s, j] = (pair[0] in first_half) != (pair[1] in first_half)
    sampled_correct = chosen_left == targets[:, None].astype(bool)
    return {
        "deterministic_correct": correct,
        "sampled_correct": sampled_correct.mean(axis=1),
        "bridge": bridge_mask,
        "cross": cross_mask,
        "classes": np.asarray(classes),
        "replicated_cycles": np.asarray(cycles),
    }


def evaluate_manipulations() -> dict:
    runtime()
    fits, inputs = fitted_parameters(), load_inputs()
    for row in fits:
        name = identity(row)
        directory = RUN_ROOT / "manipulations" / name
        if directory.exists():
            validate_complete(directory)
            continue
        runner = load_runner(row)
        with ProspectiveRun.start(
            directory,
            workflow_id="structural_identification_v1",
            execution_id=f"manipulations-{name}",
            producer={"module": __name__},
            resolved_config=row,
        ):
            arrays = {}
            for condition, source in inputs["manipulations"].items():
                base, extra = read_batch(source)
                margins, state = predict(
                    runner, encode(base, row["structure"], extra["uniforms"])
                )
                values = manipulation_record(margins, base, 1350011)
                arrays.update(
                    {
                        f"{condition}__{key}": value
                        for key, value in {
                            **values,
                            "margins": margins,
                            "w": state,
                        }.items()
                    }
                )
            save_arrays(directory / "outputs.npz", **arrays)
        print("manipulations", name, "complete", flush=True)
    return {"fits": len(fits), "conditions": len(inputs["manipulations"])}
