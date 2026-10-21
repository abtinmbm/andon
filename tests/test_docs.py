"""The README and model card quote only numbers from the generated results tables.

The tables in ``results/*.md`` are generated from ``results/results_log.csv``, so a figure
that appears in the docs but in no table was typed by hand or has gone stale.
"""

import re
from pathlib import Path

import pytest

from andon.evaluate import PUBLISHED_BASELINE

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", ROOT / "docs" / "model_card.md"]
TABLES = sorted((ROOT / "results").glob("*.md"))

NUMBER = re.compile(r"(?<![\d.])\d+\.\d{2}(?![\d.])")  # exactly two decimals: a reported figure
PAIR = re.compile(r"(\d+\.\d{2}) (?:±|\+/-) (\d+\.\d{2})")
LINK = re.compile(r"\]\(([^)#]+)(?:#[^)]*)?\)")
VERSION = re.compile(r"Python \d+\.\d+")  # not a result


def doc_text(doc: Path) -> str:
    return VERSION.sub("", doc.read_text(encoding="utf-8"))


def published_figures() -> set[str]:
    """The published DCASE baseline (development set), per type and mean, in %."""
    aucs = [auc for auc, _ in PUBLISHED_BASELINE.values()]
    figures = {f"{100 * a:.2f}" for a in aucs}
    figures.add(f"{100 * sum(aucs) / len(aucs):.2f}")
    return figures


def table_text() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in TABLES)


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_every_figure_is_in_a_results_table(doc):
    tables = table_text()
    allowed = set(NUMBER.findall(tables)) | published_figures()
    unknown = sorted(set(NUMBER.findall(doc_text(doc))) - allowed)
    assert not unknown, f"{doc.name} quotes figures that no results table holds: {unknown}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_every_mean_and_spread_pair_is_in_a_results_table(doc):
    """A mean must be quoted with its own spread, not one from another config or type."""
    pairs = set(PAIR.findall(table_text()))
    unknown = sorted(set(PAIR.findall(doc_text(doc))) - pairs)
    assert not unknown, f"{doc.name} pairs a mean with the wrong spread: {unknown}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_relative_links_point_at_files_in_the_repo(doc):
    links = [t for t in LINK.findall(doc.read_text(encoding="utf-8")) if "://" not in t]
    assert links
    missing = [t for t in links if not (doc.parent / t).exists()]
    assert not missing, f"{doc.name} links to missing files: {missing}"


def test_readme_gives_the_two_reproduce_commands():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "uv sync\nuv run python -m andon.reproduce\n" in text


def test_the_checks_catch_a_made_up_figure():
    tables = table_text()
    assert "92.13 +/- 0.16" in tables  # the headline is in a table...
    allowed = set(NUMBER.findall(tables)) | published_figures()
    assert "92.14" not in allowed  # ...and a one-digit slip is not
    assert ("92.13", "0.17") not in set(PAIR.findall(tables))
