"""
Build the in-repo sample PDF from its markdown source.

Run from backend/ when the primer source changes:

python -m evaluation.build_primer_pdf

The generated file is committed, so this only needs rerunning after an
edit to sample_docs/papermind-rag-primer.md.
"""

from pathlib import Path

import pymupdf

SOURCE = Path(__file__).parent / "sample_docs" / "papermind-rag-primer.md"
OUTPUT = Path(__file__).parent / "sample_docs" / "papermind-rag-primer.pdf"


def render_markdown(doc: pymupdf.Document, source_text: str) -> None:
    """Do render markdown."""
    page = doc[0]
    y = 72.0

    def advance(amount):
        """Do advance."""
        nonlocal page, y
        y += amount
        if y > 740:
            page = doc.new_page()
            y = 72.0

    for line in source_text.splitlines():
        if not line.strip():
            advance(10)
            continue
        fontsize = 11.0
        if line.startswith("## "):
            line = line[3:]
            fontsize = 14.0
            advance(10)
        elif line.startswith("# "):
            line = line[2:]
            fontsize = 17.0
            advance(8)
        # Wrap long lines at roughly 90 characters for the page width.
        while len(line) > 90:
            cut = line.rfind(" ", 0, 90)
            head, line = line[:cut], line[cut + 1 :]
            page.insert_text((72, y), head, fontsize=fontsize)
            advance(fontsize + 6)
        page.insert_text((72, y), line, fontsize=fontsize)
        advance(fontsize + 6)


def build() -> Path:
    """Do build."""
    doc = pymupdf.open()
    doc.new_page()
    render_markdown(doc, SOURCE.read_text(encoding="utf-8"))
    doc.save(OUTPUT)
    doc.close()
    return OUTPUT


if __name__ == "__main__":
    print(f"wrote {build()}")
