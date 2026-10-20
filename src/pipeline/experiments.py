"""Turn experiments.md into JSON for the Model Lab page, so the markdown stays the single source of truth."""

import re

FIRST_SECTION = "Phases 3-4: single validation season (2024), protocol v1"


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def row_status(cells: dict[str, str]) -> str:
    """keep / drop / reference, read from the row's keep/drop column (the 2025 test table has none: reference)."""
    call = cells.get("keep/drop", "")
    text = re.sub(r"[*_`]", "", call).strip().lower()
    if text.startswith("keep"):
        return "keep"
    if text.startswith("drop"):
        return "drop"
    return "reference"


def parse_experiments(md: str) -> list[dict]:
    """Every table in experiments.md as {title, level, notes, columns, rows}, in file order.

    title = the nearest heading above the table; notes = the plain text between that heading (or the previous
    table) and the next table, plus text after the last table of a section. Each row is {cells, status}.
    """
    sections, notes, title, level = [], [], FIRST_SECTION, 2
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#"):
            hashes = len(line) - len(line.lstrip("#"))
            if hashes > 1:  # "# Experiments" is the page title, not a section
                title, level, notes = line.lstrip("#").strip(), hashes, []
            i += 1
        elif line.startswith("|") and i + 1 < len(lines) and set(lines[i + 1].replace("|", "").strip()) <= set("-: "):
            columns = _cells(line)
            i += 2
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = dict(zip(columns, _cells(lines[i])))
                rows.append({"cells": cells, "status": row_status(cells)})
                i += 1
            sections.append({"title": title, "level": level, "notes": notes, "columns": columns, "rows": rows})
            notes = []
        else:
            if line.strip():
                # Text right after a table (no heading in between) belongs to that table.
                if sections and sections[-1]["title"] == title and not notes:
                    sections[-1].setdefault("after", []).append(line.strip())
                else:
                    notes.append(line.strip())
            i += 1
    for s in sections:
        s["notes"] = " ".join(s["notes"])
        s["after"] = " ".join(s.get("after", []))
    return sections
