from __future__ import annotations

import ast

from fsrl.experiments.single_p_time_role_direct.protocol import (
    PROTOCOL,
    PROTOCOL_SHA256,
    specification,
)
from fsrl.infra.provenance import file_sha256


def test_frozen_protocol_identity_and_two_phase_gate():
    spec = specification()
    assert file_sha256(PROTOCOL) == PROTOCOL_SHA256
    assert spec["design"]["network_seeds"] == [3011, 3012, 3013]
    assert spec["design"]["no_training"].startswith("No optimizer")
    assert spec["phase_1_direct_authority_baseline"][
        "self_reproducibility_gate"
    ].endswith("No numerical fallback tolerance is allowed.")
    assert spec["baseline_freeze"]["authorization_gate"].startswith(
        "Only baseline_compatible"
    )


def test_direct_internal_module_has_no_adapter_import():
    direct = (
        PROTOCOL.parents[4] / "fsrl/experiments/single_p_time_role_direct/direct.py"
    )
    tree = ast.parse(direct.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not any("adapter" in name for name in imports)
