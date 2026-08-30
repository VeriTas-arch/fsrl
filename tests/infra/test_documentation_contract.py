import re
import unittest
from itertools import pairwise
from pathlib import Path
from urllib.parse import unquote

from fsrl.infra.markdown_rendering import wrap_markdown
from fsrl.infra.study_registry import GENERATED_PATHS, render_navigation
from fsrl.paths import REPO_ROOT

ROOT = REPO_ROOT
FROZEN_SNAPSHOT = ROOT / "synthesis" / "snapshots" / "reporting_v1"
UPSTREAM_SOURCE = ROOT / "reproductions" / "relational_learning_2024" / "upstream"
WORKFLOW_README = Path("workflows/relational_model/README.md")
GENERATED_MARKER = re.compile(
    r"<!-- fsrl-doc role=generated-navigation source=([^ ]+) -->"
)
LOCAL_LINK = re.compile(r"(?<!!)\[[^]]*\]\(([^)]+)\)")
EXPLICIT_ANCHOR = re.compile(r'<a\s+id="([^"]+)"\s*></a>')


def _is_active_document(path: Path) -> bool:
    return FROZEN_SNAPSHOT not in path.parents and UPSTREAM_SOURCE not in path.parents


def _active_navigation_documents() -> list[Path]:
    paths = {*ROOT.rglob("AGENTS.md"), *ROOT.rglob("README.md")}
    return sorted(path for path in paths if _is_active_document(path))


def _headings(content: str) -> list[tuple[int, str]]:
    headings = []
    in_fence = False
    for line in content.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = re.fullmatch(r"(#{1,6})\s+(.+)", line)
        if match is not None:
            headings.append((len(match.group(1)), match.group(2).strip()))
    return headings


def _heading_slug(title: str) -> str:
    title = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", title)
    title = title.replace("`", "").lower()
    title = re.sub(r"[^\w\- ]", "", title)
    return title.replace(" ", "-")


def _document_anchors(path: Path) -> set[str]:
    content = path.read_text(encoding="utf-8")
    anchors = set(EXPLICIT_ANCHOR.findall(content))
    occurrences: dict[str, int] = {}
    for _, title in _headings(content):
        base = _heading_slug(title)
        occurrence = occurrences.get(base, 0)
        occurrences[base] = occurrence + 1
        anchors.add(base if occurrence == 0 else f"{base}-{occurrence}")
    return anchors


def _heading_structure_failures(path: Path) -> list[str]:
    relative = path.relative_to(ROOT)
    headings = _headings(path.read_text(encoding="utf-8"))
    failures = []
    if not headings or headings[0][0] != 1:
        failures.append(f"{relative}: first heading is not H1")
    if sum(level == 1 for level, _ in headings) != 1:
        failures.append(f"{relative}: expected exactly one H1")
    for previous, current in pairwise(headings):
        if current[0] > previous[0] + 1:
            failures.append(
                f"{relative}: heading level jumps {previous[0]}->{current[0]}"
            )
    return failures


def _fence_structure_failures(path: Path) -> list[str]:
    relative = path.relative_to(ROOT)
    content = path.read_text(encoding="utf-8")
    failures = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith("~~~"):
            failures.append(f"{relative}:{line_number}: tilde code fence")
        if not stripped.startswith("```"):
            continue
        if not in_fence and stripped == "```":
            failures.append(f"{relative}:{line_number}: opening fence has no language")
        in_fence = not in_fence
    if in_fence:
        failures.append(f"{relative}: unclosed code fence")
    return failures


class DocumentationContractTests(unittest.TestCase):
    def test_generated_prose_wrapping_preserves_markdown_atoms(self):
        link = "[source paper](<records/docs/source paper.pdf>)"
        inline_code = "`direnv exec . python -m example`"
        wrapped = "\n".join(wrap_markdown(f"- {link} — rebuild with {inline_code}"))
        self.assertIn(link, wrapped)
        self.assertIn(inline_code, wrapped)

    def test_active_navigation_markdown_has_a_stable_structure(self):
        failures = []
        for path in _active_navigation_documents():
            failures.extend(_heading_structure_failures(path))
            failures.extend(_fence_structure_failures(path))
        self.assertEqual(failures, [])

    def test_generated_navigation_declares_source_and_rebuild_command(self):
        registry_generated = set(render_navigation())
        portal_paths = {path for path in registry_generated if len(path.parts) == 2}
        self.assertEqual(set(GENERATED_PATHS), portal_paths)
        generated = registry_generated | {WORKFLOW_README}
        for relative in sorted(generated):
            content = (ROOT / relative).read_text(encoding="utf-8")
            marker = GENERATED_MARKER.search(content)
            self.assertIsNotNone(marker, relative)
            source = Path(marker.group(1))
            self.assertFalse(source.is_absolute(), relative)
            self.assertTrue((ROOT / source).is_file(), f"{relative}: {source}")
            self.assertIn("do not edit this README directly", content, relative)
            if relative == WORKFLOW_README:
                self.assertIn("python -m fsrl.workflows render", content)
            else:
                self.assertIn("python -m fsrl.infra.study_registry build", content)

        figure_readme = ROOT / "synthesis" / "figures" / "README.md"
        self.assertNotIn(figure_readme.relative_to(ROOT), generated)
        self.assertIsNone(GENERATED_MARKER.search(figure_readme.read_text()))

    def test_local_markdown_fragments_resolve(self):
        anchor_cache: dict[Path, set[str]] = {}
        failures = []
        for source in _active_navigation_documents():
            content = source.read_text(encoding="utf-8")
            for raw_target in LOCAL_LINK.findall(content):
                target = raw_target.strip("<>")
                if "#" not in target or "://" in target:
                    continue
                path_text, fragment = target.split("#", 1)
                if not fragment:
                    continue
                target_path = (
                    source if not path_text else source.parent / unquote(path_text)
                )
                target_path = target_path.resolve()
                if not target_path.is_file() or target_path.suffix != ".md":
                    continue
                anchors = anchor_cache.setdefault(
                    target_path, _document_anchors(target_path)
                )
                if unquote(fragment) not in anchors:
                    failures.append(f"{source.relative_to(ROOT)} -> {raw_target}")
        self.assertEqual(failures, [])

    def test_agent_scope_tree_and_human_entrypoints_are_navigable(self):
        guides = sorted(ROOT.rglob("AGENTS.md"))
        failures = []
        for guide in guides:
            content = guide.read_text(encoding="utf-8")
            lines = content.splitlines()
            if len(lines) < 3 or not lines[2].startswith("This file applies"):
                failures.append(f"{guide.relative_to(ROOT)}: missing scope sentence")
            if guide == ROOT / "AGENTS.md":
                continue
            if "Navigation:" not in content:
                failures.append(f"{guide.relative_to(ROOT)}: missing Navigation")

            parent = guide.parent.parent
            while parent != ROOT and not (parent / "AGENTS.md").is_file():
                parent = parent.parent
            parent_guide = parent / "AGENTS.md"
            relative = guide.relative_to(parent_guide.parent).as_posix()
            if f"]({relative})" not in parent_guide.read_text(encoding="utf-8"):
                failures.append(
                    f"{parent_guide.relative_to(ROOT)}: does not link {relative}"
                )

            if (guide.parent / "README.md").is_file() and "](README.md" not in content:
                failures.append(
                    f"{guide.relative_to(ROOT)}: does not link its local README"
                )
        self.assertEqual(failures, [])

    def test_root_guides_cover_every_public_top_level_directory(self):
        directories = {
            path.name
            for path in ROOT.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        }
        for document in (ROOT / "AGENTS.md", ROOT / "README.md"):
            content = document.read_text(encoding="utf-8")
            missing = sorted(name for name in directories if f"{name}/" not in content)
            self.assertEqual(missing, [], document.name)


if __name__ == "__main__":
    unittest.main()
