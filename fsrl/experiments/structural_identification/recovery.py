"""Observable joint-choice recovery with independent nuisance integration."""

from itertools import product

import numpy as np
import torch
from scipy.special import expit, logsumexp

from fsrl.experiments.cohort_diagnostic.statistics import wilson
from fsrl.experiments.minimal_learner.locks import validate_complete
from fsrl.experiments.minimal_learner.training import runtime
from fsrl.experiments.quantized_learner.recovery_inputs import (
    ObservedDesign,
    nuisance_pool,
)
from fsrl.infra.provenance import load_json, write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .execution import validate_source
from .inputs import load_inputs, read_batch, save_arrays
from .model import encode, reference
from .protocol import RUN_ROOT, STRUCTURES, specification


def generating_counts(design, rng, settings):
    grid = list(product(settings["eta_grid"], settings["gain_grid"]))
    repeats = settings["datasets_per_setting"]
    size = len(STRUCTURES) * len(grid) * repeats
    pool, witness = nuisance_pool(design, rng, size)
    margins = np.empty((size, len(design.query_cues)))
    for family, structure in enumerate(STRUCTURES):
        encoded = encode(pool, structure, witness["encoding_uniforms"])
        for parameter, (eta, gain) in enumerate(grid):
            sl = slice(
                (family * len(grid) + parameter) * repeats,
                (family * len(grid) + parameter + 1) * repeats,
            )
            arrays = {
                key: value[sl] if key == "query_cues" else value[:, sl]
                for key, value in encoded.arrays.items()
            }
            from fsrl.experiments.minimal_learner.data import ModelBatch

            margins[sl] = reference(ModelBatch(arrays), eta, gain, "decay")[0]
    choices = rng.random(
        (size, settings["choice_repetitions"], margins.shape[1])
    ) < expit(margins[:, None] / 0.25)
    return choices.sum(axis=1).astype(np.uint8)


def mixture_scores(margins, counts, budgets, repetitions):
    """Multiply all observations for a subject, then integrate over its state."""
    logits = torch.as_tensor(margins / 0.25, dtype=torch.float64, device="cuda")
    left = torch.as_tensor(counts.T, dtype=torch.float64, device="cuda")
    conditional = -torch.logaddexp(torch.zeros_like(logits), -logits) @ left
    conditional -= torch.logaddexp(torch.zeros_like(logits), logits) @ (
        repetitions - left
    )
    return np.stack(
        [
            (torch.logsumexp(conditional[:budget], dim=0) - np.log(budget))
            .cpu()
            .numpy()
            for budget in budgets
        ]
    )


def decode_episode(design, counts, rng, settings):
    """Decoder has no generating latent, identity, or ground-truth rank argument."""
    budgets = settings["nuisance_draws"]
    pool, witness = nuisance_pool(design, rng, max(budgets))
    grid = list(product(settings["eta_grid"], settings["gain_grid"]))
    scores = np.empty((len(budgets), len(counts), len(STRUCTURES), len(grid)))
    for family, structure in enumerate(STRUCTURES):
        encoded = encode(pool, structure, witness["encoding_uniforms"])
        for index, eta in enumerate(settings["eta_grid"]):
            margins = reference(encoded, eta, 1, "decay")[0]
            for g, gain in enumerate(settings["gain_grid"]):
                scores[:, :, family, index * len(settings["gain_grid"]) + g] = (
                    mixture_scores(
                        gain * margins, counts, budgets, settings["choice_repetitions"]
                    )
                )
    return scores


def recovery_summary(scores, settings):
    # Parameters are shared across all episodes in one simulated dataset.
    integrated = logsumexp(scores.sum(axis=0), axis=-1) - np.log(scores.shape[-1])
    selected = integrated.argmax(axis=-1)
    per_family = (
        len(settings["eta_grid"])
        * len(settings["gain_grid"])
        * settings["datasets_per_setting"]
    )
    truth = np.repeat(np.arange(4), per_family)
    confusion = np.zeros((len(selected), 4, 4), dtype=int)
    for budget in range(len(selected)):
        np.add.at(confusion[budget], (truth, selected[budget]), 1)
    agreement = float(np.mean(selected[0] == selected[1]))
    pairwise = {}
    for a, b in product(range(4), repeat=2):
        if a >= b:
            continue
        directional = []
        for generator, opponent in ((a, b), (b, a)):
            correct = (
                integrated[-1, truth == generator, generator]
                > integrated[-1, truth == generator, opponent]
            )
            directional.append(wilson(correct))
        pairwise[f"{STRUCTURES[a]}-{STRUCTURES[b]}"] = {
            "directions": directional,
            "distinguishable": all(
                d["rate"] >= 0.8 and d["lower"] > 0.5 for d in directional
            )
            and agreement >= 0.95,
        }
    return {
        "confusion_counts": confusion.tolist(),
        "integration_winner_agreement": agreement,
        "pairwise": pairwise,
        "per_parameter_correct": (selected[-1] == truth)
        .reshape(4, -1, settings["datasets_per_setting"])
        .mean(axis=-1)
        .tolist(),
        "scope": "Finite synthetic screen; no human fitting and no structural winner selected.",
    }


def run_recovery() -> dict:
    runtime()
    validate_source()
    inputs = load_inputs()
    settings = specification()["recovery"]
    results = {}
    for design_index, (name, source) in enumerate(inputs["recovery"].items()):
        directory = RUN_ROOT / "recovery" / name
        if directory.exists():
            validate_complete(directory)
            results[name] = load_json(directory / "result.json")
            continue
        with ProspectiveRun.start(
            directory,
            workflow_id="structural_identification_v1",
            execution_id=f"recovery-{name}",
            producer={"module": __name__, "input": source},
            resolved_config=settings,
        ):
            batch, _ = read_batch(source)
            scores, observed = [], []
            for subject in range(settings["episodes_per_dataset"]):
                design = ObservedDesign.from_batch(batch, subject)
                # Across schedule conditions, generating/prior RNG channels are paired.
                counts = generating_counts(
                    design,
                    np.random.default_rng(settings["seed"] + 100 + subject),
                    settings,
                )
                scores.append(
                    decode_episode(
                        design,
                        counts,
                        np.random.default_rng(settings["seed"] + 1000 + subject),
                        settings,
                    )
                )
                observed.append(counts)
                print(
                    "recovery",
                    name,
                    subject + 1,
                    "of",
                    settings["episodes_per_dataset"],
                    flush=True,
                )
            scores = np.stack(scores)
            result = recovery_summary(scores, settings)
            save_arrays(
                directory / "outputs.npz",
                counts=np.stack(observed),
                integrated_episode_scores=scores,
            )
            write_json_exclusive(directory / "result.json", result)
            results[name] = result
    return results
