from __future__ import annotations

import ast

from fsrl.experiments.single_p_time_role_propagated.protocol import (
    PROTOCOL,
    PROTOCOL_SHA256,
    specification,
)
from fsrl.infra.provenance import file_sha256


def test_frozen_protocol_identity_and_two_phase_gate():
    spec = specification()
    assert file_sha256(PROTOCOL) == PROTOCOL_SHA256
    assert spec["design"]["network_seeds"] == [3011, 3012, 3013]
    assert spec["design"]["no_training"].endswith("are prohibited.")
    assert spec["numerical_contract"]["margin_gate"]["formula"].startswith(
        "For every finite reference margin"
    )
    assert spec["batch_shape_contract"]["stage_3_probe"].startswith(
        "For every episode and prefix"
    )
    assert spec["baseline_freeze"]["authorization_gate"].startswith(
        "Only baseline_compatible"
    )


def test_direct_internal_module_has_no_adapter_import():
    direct = (
        PROTOCOL.parents[4] / "fsrl/experiments/single_p_time_role_propagated/direct.py"
    )
    tree = ast.parse(direct.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not any("adapter" in name for name in imports)
