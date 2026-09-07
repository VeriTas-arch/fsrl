"""Matched base task and observation streams, with no local optimizer group."""

import hashlib
import json
import time

import numpy as np

from fsrl.experiments.local_memory_removal.model import update
from fsrl.experiments.memory_structure.inputs import generator, prepare_shared
from fsrl.experiments.memory_structure.model import optimizer_for
from fsrl.experiments.observation_uncertainty.inputs import encode
from fsrl.experiments.training_strategy.batches import sample_episodes

from .inputs import remove_hint


def train_steps(backbone, local, seqs, spec, seed, arm, states, coefficient, directory):
    optimizer = optimizer_for(backbone, local, spec)
    task = generator(spec)
    rng = np.random.default_rng(spec["seeds"]["training_rng_offset"] + seed)
    digest, started = hashlib.sha256(), time.perf_counter()
    with (directory / "train_log.jsonl").open("x") as log:
        for step in range(spec["optimization"]["total_steps"]):
            cpu = prepare_shared(
                sample_episodes(task, rng, spec["optimization"]["batch_size"])
            )
            fingerprint = cpu.fingerprint()
            digest.update(bytes.fromhex(fingerprint))
            draw = 400000000 + seed * 10000 + step
            observed = encode(cpu, arm, spec["observation"]["sigma"], seed=draw)
            candidate = remove_hint(observed)
            row: dict = update(
                backbone,
                local,
                seqs,
                candidate.to("cuda"),
                optimizer,
                spec,
                states,
                coefficient,
                draw,
            )
            row.update(
                step=step,
                observation_seed=draw,
                observed_fingerprint=observed.fingerprint(),
                candidate_fingerprint=candidate.fingerprint(),
                q_sha256=hashlib.sha256(
                    candidate.arrays["local_evidence"].tobytes()
                ).hexdigest(),
                batch_fingerprint=fingerprint,
                stream_fingerprint=digest.hexdigest(),
            )
            log.write(json.dumps(row, allow_nan=False) + "\n")
            if step % 100 == 0 or step == spec["optimization"]["total_steps"] - 1:
                log.flush()
                print(
                    json.dumps({"seed": seed, "arm": arm, **row}),
                    flush=True,
                )
    return {
        "training_seconds": time.perf_counter() - started,
        "stream_fingerprint": digest.hexdigest(),
        "optimizer_steps": {
            f"{prefix}.{name}": int(optimizer.state.get(p, {}).get("step", 0))
            for prefix, module in [("backbone", backbone)]
            for name, p in module.named_parameters()
        },
    }
