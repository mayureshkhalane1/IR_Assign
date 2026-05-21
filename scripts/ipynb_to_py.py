#!/usr/bin/env python3
"""Minimal ipynb -> py extractor (code cells only).

Avoids jupyter/nbconvert plugin issues.
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ipynb", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args()

    nb = json.loads(args.ipynb.read_text(encoding="utf-8"))
    out_lines = [
        f"# Auto-generated from {args.ipynb.name}",
        "# Code cells only; markdown removed.",
        "",
    ]

    cell_no = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        cell_no += 1
        src = cell.get("source", [])
        if isinstance(src, str):
            src = src.splitlines(True)
        out_lines.append(f"\n# --- cell {cell_no} ---")
        for line in src:
            # Strip IPython magics and shell escapes
            if line.lstrip().startswith("!") or line.lstrip().startswith("%"):
                out_lines.append(f"# [ipynb-magic removed] {line.rstrip()}\n")
            else:
                out_lines.append(line if line.endswith("\n") else line + "\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(out_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
