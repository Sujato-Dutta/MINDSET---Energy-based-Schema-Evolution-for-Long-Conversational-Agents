"""Static integrity checks for the Overleaf paper package."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
MAIN = PAPER / "main.tex"


def balanced_braces(text: str, name: str) -> None:
    depth = 0
    for index, char in enumerate(text):
        if char == "{" and (index == 0 or text[index - 1] != "\\"):
            depth += 1
        elif char == "}" and (index == 0 or text[index - 1] != "\\"):
            depth -= 1
            if depth < 0:
                raise AssertionError(f"Extra closing brace in {name} at offset {index}")
    if depth:
        raise AssertionError(f"Unbalanced braces in {name}: depth={depth}")


def check_environments(text: str, name: str) -> None:
    stack: list[str] = []
    for match in re.finditer(r"\\(begin|end)\{([^}]+)\}", text):
        action, environment = match.groups()
        if action == "begin":
            stack.append(environment)
        elif not stack or stack.pop() != environment:
            raise AssertionError(f"Mismatched environment {environment} in {name}")
    if stack:
        raise AssertionError(f"Unclosed environments in {name}: {stack}")


def main() -> None:
    main_text = MAIN.read_text(encoding="utf-8-sig")
    bib_text = (PAPER / "references.bib").read_text(encoding="utf-8-sig")
    table_files = sorted((PAPER / "tables").glob("table_*.tex"))
    assert len(table_files) == 6, f"Expected six tables, found {len(table_files)}"
    assert (PAPER / "figures" / "mindset_architecture.png").stat().st_size > 0

    inputs = re.findall(r"\\input\{([^}]+)\}", main_text)
    assert len(inputs) == 6
    for relative in inputs:
        assert PAPER.joinpath(relative + ".tex").exists(), relative
    for relative in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", main_text):
        assert PAPER.joinpath(relative).exists(), relative

    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib_text))
    cited: set[str] = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", main_text):
        cited.update(item.strip() for item in group.split(","))
    assert len(bib_keys) == 38, f"Expected 38 BibTeX entries, found {len(bib_keys)}"
    assert cited <= bib_keys, f"Missing bibliography keys: {sorted(cited - bib_keys)}"
    assert bib_keys <= cited, f"Uncited bibliography keys: {sorted(bib_keys - cited)}"

    balanced_braces(bib_text, "references.bib")
    files = [MAIN, *table_files]
    for path in files:
        text = path.read_text(encoding="utf-8-sig")
        balanced_braces(text, path.name)
        check_environments(text, path.name)
        for artifact in ("<span", "**", "## ", "```"):
            assert artifact not in text, f"Markdown artifact {artifact!r} in {path.name}"
    for display in re.findall(r"\\\[(.*?)\\\]", main_text, flags=re.S):
        assert not re.search(r"(?<!\\text\{)\\textsc\{", display), "Invalid text small-caps command remains in math mode"

    assert r"\bibliography{references}" in main_text
    assert r"\appendix" in main_text
    assert main_text.index(r"\bibliography{references}") < main_text.index(r"\appendix")
    print(
        f"PASS: main.tex, {len(table_files)} tables, {len(bib_keys)} references, "
        f"{len(cited)} cited keys, and all figure/include paths are structurally valid."
    )


if __name__ == "__main__":
    main()



