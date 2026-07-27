"""Export docs/mindset_paper.md as an Overleaf-ready LaTeX project."""

from __future__ import annotations

import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "mindset_paper.md"
PAPER = ROOT / "paper"
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"

TABLE_NAMES = (
    "table_1_locomo",
    "table_2_mab",
    "table_3_ablation",
    "table_4a_glm",
    "table_4b_gemma",
    "table_5_efficiency",
)


def tex_escape(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
    }
    return "".join(replacements.get(char, char) for char in text)


def inline(text: str) -> str:
    placeholders: dict[str, str] = {}

    def hold(value: str) -> str:
        key = f"@@HOLD{len(placeholders)}@@"
        placeholders[key] = value
        return key

    text = re.sub(
        r'<span style="color:#1f6feb"><u>(.*?)</u></span>',
        lambda match: hold(r"\second{" + tex_escape(match.group(1)) + "}"),
        text,
    )
    text = re.sub(
        r"`([^`]+)`",
        lambda match: hold(r"\texttt{\detokenize{" + match.group(1) + "}}"),
        text,
    )
    text = re.sub(
        r"\\\((.*?)\\\)",
        lambda match: hold(r"\(" + match.group(1) + r"\)"),
        text,
    )
    text = re.sub(
        r"\[([0-9]+(?:,\s*[0-9]+)*)\]",
        lambda match: hold(
            r"\cite{" + ",".join(f"ref{item.strip()}" for item in match.group(1).split(",")) + "}"
        ),
        text,
    )
    text = re.sub(
        r"\[([^\]]+)\]\((https?://[^)]+)\)",
        lambda match: hold(r"\href{" + match.group(2) + "}{" + tex_escape(match.group(1)) + "}"),
        text,
    )
    text = tex_escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\\emph{\1}", text)
    text = text.replace("--", r"\textemdash{}")
    for key, value in placeholders.items():
        text = text.replace(key, value)
    return text


def table_tex(rows: list[list[str]], caption: str, label: str) -> str:
    column_count = len(rows[0])
    alignment = "l" + "r" * (column_count - 1)
    rendered: list[str] = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(r"\textbf{" + inline(cell) + "}" for cell in rows[0]) + r" \\",
        r"\midrule",
    ]
    for row in rows[1:]:
        values = [inline(cell) if cell != "--" else r"\textemdash{}" for cell in row]
        rendered.append(" & ".join(values) + r" \\")
    rendered.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            rf"\caption{{{inline(caption)}}}",
            rf"\label{{tab:{label}}}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(rendered)


def strip_number(title: str) -> str:
    return re.sub(r"^(?:[A-Z]\.)?\d+(?:\.\d+)*\.?\s*", "", title).strip()


def convert_markdown(lines: list[str], *, appendix: bool = False) -> tuple[str, list[str]]:
    output: list[str] = []
    includes: list[str] = []
    index = 0
    table_index = 0
    in_abstract = False
    list_mode: str | None = None

    def close_list() -> None:
        nonlocal list_mode
        if list_mode:
            output.append(rf"\end{{{list_mode}}}")
            output.append("")
            list_mode = None

    while index < len(lines):
        line = lines[index].rstrip()
        if line.startswith("# "):
            index += 1
            continue
        if line == "**Anonymous Authors**":
            index += 1
            continue
        if line.startswith("## Abstract"):
            output.append(r"\begin{abstract}")
            in_abstract = True
            index += 1
            continue
        if line.startswith("## "):
            close_list()
            if in_abstract:
                output.extend([r"\end{abstract}", ""])
                in_abstract = False
            title = strip_number(line[3:])
            if appendix and title.lower().startswith("appendix"):
                output.extend([r"\appendix", r"\section{Reproducibility Checklist}", ""])
            else:
                output.extend([rf"\section{{{inline(title)}}}", ""])
            index += 1
            continue
        if line.startswith("### "):
            close_list()
            output.extend([rf"\subsection{{{inline(strip_number(line[4:]))}}}", ""])
            index += 1
            continue
        if line.startswith("#### "):
            close_list()
            output.extend([rf"\subsubsection{{{inline(strip_number(line[5:]))}}}", ""])
            index += 1
            continue
        if line.startswith("```"):
            close_list()
            language = line[3:].strip()
            code: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].startswith("```"):
                code.append(lines[index].rstrip("\n"))
                index += 1
            options = f"[language={language}]" if language in {"Python"} else ""
            output.append(rf"\begin{{lstlisting}}{options}")
            output.extend(code)
            output.extend([r"\end{lstlisting}", ""])
            index += 1
            continue
        if line == r"\[":
            close_list()
            output.append(r"\[")
            index += 1
            while index < len(lines):
                math_line = lines[index].rstrip()
                math_line = re.sub(r"\\textsc\{([^}]+)\}", r"\\text{\\textsc{\1}}", math_line)
                output.append(math_line)
                if lines[index].rstrip() == r"\]":
                    break
                index += 1
            output.append("")
            index += 1
            continue
        if line.startswith("| "):
            close_list()
            table_rows: list[list[str]] = []
            while index < len(lines) and lines[index].startswith("|"):
                row = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", cell) for cell in row):
                    table_rows.append(row)
                index += 1
            while index < len(lines) and not lines[index].strip():
                index += 1
            caption = ""
            if index < len(lines) and lines[index].startswith("**Table"):
                caption = re.sub(r"^\*\*Table [^*]+\.\*\*\s*", "", lines[index]).strip()
                index += 1
            name = TABLE_NAMES[table_index]
            TABLES.joinpath(name + ".tex").write_text(
                table_tex(table_rows, caption, name), encoding="utf-8", newline="\n"
            )
            output.extend([rf"\input{{tables/{name}}}", ""])
            includes.append(name)
            table_index += 1
            continue
        if line.startswith("!["):
            close_list()
            match = re.match(r"!\[([^]]+)\]\(([^)]+)\)", line)
            caption = "MINDSET architecture."
            lookahead = index + 1
            while lookahead < len(lines) and not lines[lookahead].strip():
                lookahead += 1
            if lookahead < len(lines) and lines[lookahead].startswith("**Figure"):
                caption = re.sub(r"^\*\*Figure [^*]+\.\*\*\s*", "", lines[lookahead]).strip()
                index = lookahead
            output.extend(
                [
                    r"\begin{figure*}[t]",
                    r"\centering",
                    r"\includegraphics[width=\textwidth]{figures/mindset_architecture.png}",
                    rf"\caption{{{inline(caption)}}}",
                    r"\label{fig:architecture}",
                    r"\end{figure*}",
                    "",
                ]
            )
            index += 1
            continue
        numbered = re.match(r"^\d+\.\s+(.*)", line)
        bulleted = re.match(r"^-\s+(.*)", line)
        if numbered or bulleted:
            mode = "enumerate" if numbered else "itemize"
            if list_mode != mode:
                close_list()
                list_mode = mode
                output.append(rf"\begin{{{mode}}}")
            output.append(r"\item " + inline((numbered or bulleted).group(1)))
            index += 1
            continue
        if line.startswith("> "):
            close_list()
            output.extend([r"\begin{quote}\itshape", inline(line[2:]), r"\end{quote}", ""])
            index += 1
            continue
        if not line:
            close_list()
            output.append("")
        else:
            close_list()
            output.append(inline(line))
            output.append("")
        index += 1
    close_list()
    if in_abstract:
        output.extend([r"\end{abstract}", ""])
    return "\n".join(output), includes


def bibtex(reference_lines: list[str]) -> str:
    entries: list[str] = []
    pattern = re.compile(r"^\[(\d+)\]\s+(.+?)\.\s+(\d{4})\.\s+\[(.+?)\]\((.+?)\)\.\s*(.*)$")
    for line in reference_lines:
        match = pattern.match(line.strip())
        if not match:
            continue
        number, authors, year, title, url, note = match.groups()
        authors = authors.replace(" et al", " and others")
        authors = authors.replace(", and ", " and ").replace(", ", " and ")
        entries.extend(
            [
                rf"@misc{{ref{number},",
                f"  author = {{{authors}}},",
                f"  title = {{{{{tex_escape(title)}}}}},",
                f"  year = {{{year}}},",
                f"  howpublished = {{\\url{{{url}}}}},",
                f"  note = {{{tex_escape(note)}}}",
                "}",
                "",
            ]
        )
    return "\n".join(entries)


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    ref_start = next(index for index, line in enumerate(lines) if line == "## References")
    appendix_start = next(index for index, line in enumerate(lines) if line.startswith("## Appendix"))
    body_lines = lines[:ref_start]
    reference_lines = lines[ref_start + 1 : appendix_start]
    appendix_lines = lines[appendix_start:]

    PAPER.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for old in TABLES.glob("table_*.tex"):
        old.unlink()

    body, body_tables = convert_markdown(body_lines)
    appendix, appendix_tables = convert_markdown(appendix_lines, appendix=True)
    if appendix_tables:
        raise RuntimeError("Unexpected table in appendix")
    if tuple(body_tables) != TABLE_NAMES:
        raise RuntimeError(f"Unexpected table export order: {body_tables}")

    preamble = r"""\documentclass[10pt,twocolumn]{article}

\usepackage[letterpaper,margin=0.72in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{newtxtext,newtxmath}
\usepackage{microtype}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{xcolor}
\usepackage{url}
\usepackage{listings}
\usepackage{enumitem}
\usepackage{caption}
\usepackage[numbers,sort&compress]{natbib}
\usepackage[colorlinks=true,allcolors=blue]{hyperref}

\definecolor{secondblue}{HTML}{1F6FEB}
\newcommand{\second}[1]{\textcolor{secondblue}{\underline{#1}}}
\newcommand{\best}[1]{\textbf{#1}}
\setlist{nosep,leftmargin=*}
\captionsetup{font=small,labelfont=bf}
\lstset{basicstyle=\ttfamily\footnotesize,breaklines=true,columns=fullflexible,
  frame=single,framerule=0.2pt,xleftmargin=2pt,xrightmargin=2pt}
\setlength{\columnsep}{0.24in}
\setlength{\emergencystretch}{2em}

\title{MINDSET: Minimum-Energy Schema Evolution for Efficient Long-Term Conversational Memory}
\author{Anonymous Authors}
\date{}

\begin{document}
\maketitle
"""
    ending = r"""
\bibliographystyle{plainnat}
\bibliography{references}

""" + appendix + "\n\n" + r"\end{document}" + "\n"
    (PAPER / "main.tex").write_text(preamble + "\n" + body + ending, encoding="utf-8", newline="\n")
    (PAPER / "references.bib").write_text(bibtex(reference_lines), encoding="utf-8", newline="\n")
    shutil.copy2(ROOT / "docs" / "assets" / "mindset_architecture.png", FIGURES / "mindset_architecture.png")

    readme = """# Overleaf package

Upload the contents of this `paper/` directory as one Overleaf project and set
`main.tex` as the main document. Use pdfLaTeX and BibTeX (Overleaf's normal
`Recompile` action handles the required passes).

Expected layout:

```text
main.tex
references.bib
figures/mindset_architecture.png
tables/table_1_locomo.tex
tables/table_2_mab.tex
tables/table_3_ablation.tex
tables/table_4a_glm.tex
tables/table_4b_gemma.tex
tables/table_5_efficiency.tex
```

The document uses only packages available in the standard current TeX Live
distribution on Overleaf. Best values are bold; second-best values use the
`\\second{}` macro (blue and underlined).
"""
    (PAPER / "README.md").write_text(readme, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()



