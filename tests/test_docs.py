# img2svg - structural tests for the docs/ directory and MkDocs configuration.
# Copyright (c) 2026, CloudBSD
# SPDX-License-Identifier: BSD-3-Clause
"""Structural tests for the docs/ directory and MkDocs configuration."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"

REQUIRED_DOCS: tuple[str, ...] = (
    "index.md",
    "installation.md",
    "usage.md",
    "api.md",
    "modes.md",
    "gpu.md",
    "configuration.md",
    "troubleshooting.md",
    "development.md",
    "architecture.md",
    "changelog.md",
)
REQUIRED_CONFIG: str = "mkdocs.yml"

INDEX_TARGETS: tuple[str, ...] = tuple(name for name in REQUIRED_DOCS if name != "index.md")


@pytest.mark.parametrize("filename", REQUIRED_DOCS)
def test_required_doc_file_exists(filename: str) -> None:
    assert (DOCS_DIR / filename).is_file(), f"missing doc file: docs/{filename}"


def test_mkdocs_yml_is_valid_yaml() -> None:
    yaml = pytest.importorskip("yaml")
    path = DOCS_DIR / REQUIRED_CONFIG
    assert path.is_file(), f"missing config: {path}"
    with path.open("rb") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), "mkdocs.yml must be a YAML mapping"
    assert data.get("site_name") == "img2svg", "site_name must be 'img2svg'"
    theme = data.get("theme")
    assert isinstance(theme, dict), "theme must be a mapping"
    assert theme.get("name") == "material", "theme.name must be 'material'"
    nav = data.get("nav")
    assert isinstance(nav, list), "nav must be a list"
    assert len(nav) == len(REQUIRED_DOCS), (
        f"nav must list all {len(REQUIRED_DOCS)} markdown docs "
        f"(including index.md as Home). Got {len(nav)} entries."
    )
    plugins = data.get("plugins")
    assert plugins is not None, "plugins must be set"
    assert isinstance(plugins, list), "plugins must be a list"
    assert len(plugins) >= 1, "plugins must list at least one plugin"
    plugin_names = _plugin_names(plugins)
    assert any(name in {"search", "mkdocstrings"} for name in plugin_names), (
        f"plugins must include 'search' or 'mkdocstrings'. Got: {plugin_names}"
    )


def _plugin_names(plugins: list[object]) -> list[str]:
    """Extract plugin names; entries can be bare strings or single-key mappings."""
    names: list[str] = []
    for p in plugins:
        if isinstance(p, str):
            names.append(p)
        elif isinstance(p, dict):
            names.extend(p.keys())
    return names


def test_mkdocs_nav_lists_all_doc_files() -> None:
    yaml = pytest.importorskip("yaml")
    path = DOCS_DIR / REQUIRED_CONFIG
    with path.open("rb") as f:
        data = yaml.safe_load(f)
    nav = data.get("nav", [])
    flat_targets: set[str] = set()
    for entry in nav:
        if isinstance(entry, str):
            flat_targets.add(entry)
        elif isinstance(entry, dict):
            for value in entry.values():
                if isinstance(value, str):
                    flat_targets.add(value)
                elif isinstance(value, list):
                    for sub in value:
                        if isinstance(sub, str):
                            flat_targets.add(sub)
                        elif isinstance(sub, dict):
                            flat_targets.update(sub.values())
    for name in REQUIRED_DOCS:
        if name == REQUIRED_CONFIG:
            continue
        assert name in flat_targets, f"nav is missing docs/{name}"


@pytest.mark.parametrize("target", INDEX_TARGETS)
def test_index_links_to_every_other_doc(target: str) -> None:
    text = (DOCS_DIR / "index.md").read_text(encoding="utf-8")
    pattern_link = rf"\]\(\s*{re.escape(target)}\s*\)"
    pattern_auto = rf"<{re.escape(target)}>"
    assert re.search(pattern_link, text) or re.search(pattern_auto, text), (
        f"docs/index.md must link to docs/{target}"
    )


def test_architecture_has_mermaid_code_block() -> None:
    text = (DOCS_DIR / "architecture.md").read_text(encoding="utf-8")
    pattern = r"```\s*mermaid\b.*?```"
    assert re.search(pattern, text, re.DOTALL | re.IGNORECASE), (
        "docs/architecture.md must contain at least one ```mermaid code block"
    )


def test_architecture_has_two_mermaid_diagrams() -> None:
    text = (DOCS_DIR / "architecture.md").read_text(encoding="utf-8")
    pattern = r"```\s*mermaid\b(.*?)```"
    blocks = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    assert len(blocks) >= 2, (
        f"docs/architecture.md must contain at least 2 mermaid blocks. Found {len(blocks)}."
    )
    directions = " ".join(
        "LR" if "flowchart LR" in b else "TB" if "flowchart TB" in b else "?" for b in blocks
    )
    assert "LR" in directions, "expected a 'flowchart LR' (pipeline) diagram"
    assert "TB" in directions, "expected a 'flowchart TB' (renderer) diagram"


def test_installation_mentions_pip_install() -> None:
    text = (DOCS_DIR / "installation.md").read_text(encoding="utf-8")
    assert "pip install img2svg" in text, "docs/installation.md must contain 'pip install img2svg'"


def test_installation_mentions_freebsd_and_macos() -> None:
    text = (DOCS_DIR / "installation.md").read_text(encoding="utf-8")
    assert re.search(r"^#+ .*freebsd", text, re.MULTILINE | re.IGNORECASE), (
        "docs/installation.md must have a FreeBSD section"
    )
    assert re.search(r"^#+ .*macos", text, re.MULTILINE | re.IGNORECASE), (
        "docs/installation.md must have a macOS section"
    )


def test_usage_has_cli_examples() -> None:
    text = (DOCS_DIR / "usage.md").read_text(encoding="utf-8")
    invocations = re.findall(r"^\s*img2svg\s+\S+", text, re.MULTILINE)
    assert len(invocations) >= 3, (
        f"docs/usage.md must contain at least 3 `img2svg` invocations. Found {len(invocations)}."
    )


def test_usage_documents_every_cli_flag() -> None:
    cli_path = PROJECT_ROOT / "src" / "img2svg" / "cli.py"
    cli_text = cli_path.read_text(encoding="utf-8")
    flag_pattern = re.compile(r"--([a-z][a-z0-9-]*)\b")
    cli_flags = set(flag_pattern.findall(cli_text))

    text = (DOCS_DIR / "usage.md").read_text(encoding="utf-8")
    missing: list[str] = []
    for flag in sorted(cli_flags):
        if flag in {"version", "help"}:
            continue
        if not re.search(rf"--{re.escape(flag)}\b", text):
            missing.append(flag)
    assert not missing, f"docs/usage.md is missing CLI flags: {missing}"


def test_gpu_mentions_all_three_vendors() -> None:
    text = (DOCS_DIR / "gpu.md").read_text(encoding="utf-8")
    assert "NVIDIA" in text, "docs/gpu.md must mention 'NVIDIA'"
    assert "AMD" in text, "docs/gpu.md must mention 'AMD'"
    assert "Apple" in text, "docs/gpu.md must mention 'Apple'"


@pytest.mark.parametrize("filename", REQUIRED_DOCS)
def test_every_doc_has_h1_title(filename: str) -> None:
    if not filename.endswith(".md"):
        return
    text = (DOCS_DIR / filename).read_text(encoding="utf-8")
    first_line = text.splitlines()[0] if text.splitlines() else ""
    assert first_line.startswith("# "), (
        f"docs/{filename} must start with an H1 title (found: {first_line!r})"
    )


@pytest.mark.parametrize("filename", REQUIRED_DOCS)
def test_every_doc_has_at_least_two_h2_sections(filename: str) -> None:
    if not filename.endswith(".md"):
        return
    text = (DOCS_DIR / filename).read_text(encoding="utf-8")
    h2_count = len(re.findall(r"^##\s+", text, re.MULTILINE))
    assert h2_count >= 2, f"docs/{filename} must have at least 2 H2 sections, found {h2_count}"


@pytest.mark.parametrize("filename", REQUIRED_DOCS)
def test_every_doc_has_a_code_block_or_table_or_list(filename: str) -> None:
    if not filename.endswith(".md"):
        return
    text = (DOCS_DIR / filename).read_text(encoding="utf-8")
    has_code = bool(re.search(r"```", text))
    has_table = "|" in text and re.search(r"^\s*\|.*\|", text, re.MULTILINE)
    has_list = bool(re.search(r"^\s*[-*+]\s+", text, re.MULTILINE))
    assert has_code or has_table or has_list, (
        f"docs/{filename} must have a code block, table, or list"
    )


def test_mkdocs_uses_palette_preference() -> None:
    yaml = pytest.importorskip("yaml")
    path = DOCS_DIR / REQUIRED_CONFIG
    with path.open("rb") as f:
        data = yaml.safe_load(f)
    theme = data.get("theme", {})
    palette = theme.get("palette")
    assert palette is not None, "theme.palette must be set"
    if isinstance(palette, dict):
        assert palette.get("scheme") == "preference", (
            f"theme.palette.scheme must be 'preference'. Got: {palette.get('scheme')!r}"
        )
    elif isinstance(palette, list):
        schemes = [p.get("scheme") for p in palette if isinstance(p, dict)]
        assert "preference" in schemes, (
            f"one of theme.palette[*].scheme must be 'preference'. Got: {schemes}"
        )


def test_mkdocs_features_include_navigation_instant_and_tracking() -> None:
    yaml = pytest.importorskip("yaml")
    path = DOCS_DIR / REQUIRED_CONFIG
    with path.open("rb") as f:
        data = yaml.safe_load(f)
    theme = data.get("theme", {})
    features = theme.get("features", [])
    assert "navigation.instant" in features, "theme.features must include 'navigation.instant'"
    assert "navigation.tracking" in features, "theme.features must include 'navigation.tracking'"


def test_changelog_has_version_zero_one_zero() -> None:
    text = (DOCS_DIR / "changelog.md").read_text(encoding="utf-8")
    assert "0.1.0" in text, "docs/changelog.md must contain a 0.1.0 entry"


def test_troubleshooting_documents_common_errors() -> None:
    text = (DOCS_DIR / "troubleshooting.md").read_text(encoding="utf-8")
    assert "img2svg info" in text, (
        "docs/troubleshooting.md should reference `img2svg info` for diagnosis"
    )
    assert "out of memory" in text.lower(), "docs/troubleshooting.md must mention out of memory"
    assert re.search(r"no gpu", text, re.IGNORECASE), (
        "docs/troubleshooting.md must mention the no-GPU case"
    )


def test_development_documents_tdd_workflow() -> None:
    text = (DOCS_DIR / "development.md").read_text(encoding="utf-8")
    assert "pytest" in text, "docs/development.md must reference pytest"
    assert re.search(r"\bTDD\b|test-first", text, re.IGNORECASE), (
        "docs/development.md must mention TDD or test-first workflow"
    )
