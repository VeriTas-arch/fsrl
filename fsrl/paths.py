"""Repository paths shared by the installed package and research workflows."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
DATA_ROOT = REPO_ROOT / "data"
STUDIES_ROOT = REPO_ROOT / "studies"
SYNTHESIS_ROOT = REPO_ROOT / "synthesis"
WORKFLOWS_ROOT = REPO_ROOT / "workflows"

__all__ = [
    "ARTIFACTS_ROOT",
    "DATA_ROOT",
    "PACKAGE_ROOT",
    "REPO_ROOT",
    "STUDIES_ROOT",
    "SYNTHESIS_ROOT",
    "WORKFLOWS_ROOT",
]
