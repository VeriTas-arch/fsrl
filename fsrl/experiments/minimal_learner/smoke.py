"""Non-Liu CUDA parity of the complete candidate and all scalar updates."""

import copy

import numpy as np
import torch
import torch.nn.functional as F

from fsrl.infra.provenance import write_json_exclusive
from fsrl.infra.run_manifest import ProspectiveRun

from .locks import sources
from .model import make_model
from .protocol import RUN_ROOT, specification
from .training import compiled, runtime


def smoke(attempt: int) -> dict:
    if attempt < 1:
        raise ValueError("smoke attempt must be positive")
    spec = specification()
    execution = runtime()
    rng = np.random.default_rng(910101)
    support = torch.tensor(
        rng.normal(size=(4, 3, 30)), device="cuda", dtype=torch.float32
    )
    signed = torch.tensor(rng.normal(size=(4, 3)), device="cuda", dtype=torch.float32)
    retained = torch.tensor(
        rng.integers(0, 2, size=(4, 3)), device="cuda", dtype=torch.float32
    )
    local = signed * (retained + (1 - retained) * 0.4)
    query = torch.tensor(
        rng.normal(size=(3, 5, 30)), device="cuda", dtype=torch.float32
    )
    signs = torch.tensor(rng.integers(0, 2, size=(3, 5)) * 2 - 1, device="cuda")
    arguments = (support, signed, retained, local, query)
    checks = {}

    def check(name, first, second):
        checks[name] = {
            "passed": bool(torch.allclose(first, second, atol=1e-5, rtol=1e-4)),
            "max_abs_error": float((first - second).abs().max()),
        }

    directory = RUN_ROOT / "smoke" / f"attempt-{attempt}"
    with ProspectiveRun.start(
        directory,
        workflow_id="minimal_relational_learner_v1",
        execution_id=f"smoke-{attempt}",
        producer={"module": __name__},
        resolved_config={"runtime": execution, "seed": 910101},
    ):
        for condition in spec["seeds"]["conditions"]:
            eager = make_model(condition, spec, "cuda")
            model = copy.deepcopy(eager)
            runner = compiled(model)
            first, second = eager(*arguments), runner(*arguments)
            for index in range(4):
                check(f"{condition}/output_{index}", first[index], second[index])
            losses = [
                F.softplus(-signs * outputs[0]).mean() for outputs in (first, second)
            ]
            check(f"{condition}/loss", *losses)
            for loss in losses:
                loss.backward()
            for (name, p), q in zip(
                eager.named_parameters(), model.parameters(), strict=True
            ):
                assert p.grad is not None and q.grad is not None
                check(f"{condition}/gradient_{name}", p.grad, q.grad)
                checks[f"{condition}/nonzero_gradient_{name}"] = {
                    "passed": bool(p.grad.abs().max() > 1e-12)
                }
            for current in (eager, model):
                optimizer = torch.optim.Adam(
                    current.parameters(), lr=spec["optimization"]["learning_rate"]
                )
                torch.nn.utils.clip_grad_norm_(
                    current.parameters(), spec["optimization"]["gradient_clip"]
                )
                optimizer.step()
            for (name, p), q in zip(
                eager.named_parameters(), model.parameters(), strict=True
            ):
                check(f"{condition}/updated_{name}", p, q)
            with torch.no_grad():
                batch = runner(*arguments)[0]
                for index in range(query.shape[1]):
                    single = eager(*arguments[:4], query[:, index : index + 1])[0][:, 0]
                    check(f"{condition}/query_{index}", batch[:, index], single)
        result = {
            "passed": all(row["passed"] for row in checks.values()),
            "seed": 910101,
            "liu_evaluated": False,
            "runtime": execution,
            "sources": sources(),
            "checks": checks,
        }
        write_json_exclusive(directory / "smoke.json", result)
    return {"passed": result["passed"], "checks": checks, "directory": str(directory)}
