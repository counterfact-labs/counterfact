"""Tier-2 lint for the counterfact-debugger Agent Skill.

Cheap, deterministic checks that the skill manifest is well-formed and self-consistent
so it doesn't rot: valid frontmatter, required fields, name matches the directory, and
every script/reference the SKILL.md mentions actually exists. This guards the skill's
structure without needing an agent or network.
"""
import re
from pathlib import Path

import pytest

_SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / "counterfact-debugger"
_SKILL_MD = _SKILL_DIR / "SKILL.md"

pytestmark = pytest.mark.skipif(not _SKILL_MD.exists(), reason="skill not present")


def _frontmatter(text: str) -> dict:
    """Parse the leading --- YAML block without a yaml dependency (simple key: value)."""
    m = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert m, "SKILL.md must start with a --- frontmatter block"
    fm, body, key = {}, m.group(1), None
    for line in body.splitlines():
        if re.match(r"^\w[\w-]*:", line):
            key, _, val = line.partition(":")
            key, fm[key] = key.strip(), val.strip()
        elif key and line.strip():  # continuation of a folded/multi-line value
            fm[key] = (fm[key] + " " + line.strip()).strip()
    return fm


def test_frontmatter_has_required_fields_and_matching_name():
    fm = _frontmatter(_SKILL_MD.read_text())
    assert fm.get("name") == _SKILL_DIR.name, "frontmatter name must match the skill directory"
    desc = fm.get("description", "").strip().strip(">|").strip()
    assert len(desc) > 40, "description should be substantial enough to drive triggering"


def test_referenced_scripts_and_reference_docs_exist():
    text = _SKILL_MD.read_text()
    for rel in re.findall(r"(?:scripts|reference)/[A-Za-z0-9_./-]+\.(?:py|md)", text):
        assert (_SKILL_DIR / rel).exists(), f"SKILL.md references missing file: {rel}"


def test_core_scripts_present_and_are_python():
    for script in ("cf_diagnose.py", "llm_fn.py", "verify.py"):
        p = _SKILL_DIR / "scripts" / script
        assert p.exists(), f"missing bundled script: {script}"
        assert "def main" in p.read_text() or "make_llm_fn" in p.read_text()


def test_reference_docs_present():
    for doc in ("failure-taxonomy.md", "reading-attribution.md", "report-schema.md"):
        assert (_SKILL_DIR / "reference" / doc).exists(), f"missing reference doc: {doc}"
