"""
Build the in-repo sample PDFs from their markdown sources.

Run from backend/ after editing one of the sources:

    python -m evaluation.build_sample_pdfs

Every generated file is committed, so this only needs rerunning after an edit
to the matching markdown file in sample_docs/.
"""

from pathlib import Path

import pymupdf

DOCS_DIR = Path(__file__).parent / "sample_docs"

#: The markdown sources written for this project, and the PDF each becomes.
#: The third-party papers in sample_docs/ are not built here: they are the
#: publisher's own files and are never regenerated.
SOURCES = (
    DOCS_DIR / "papermind-rag-primer.md",
    DOCS_DIR / "papermind-eval-methods.md",
    DOCS_DIR / "papermind-team-notes.md",
)

#: The build date written into every generated file, pinned for the reason
#: ``build`` gives.
BUILD_DATE = "D:20260101000000Z"

#: Roughly the width of the page at the body size, in characters.
LINE_WIDTH = 90
BODY_SIZE = 11.0
HEADING_SIZE = 14.0
TITLE_SIZE = 17.0
TOP_MARGIN = 72.0
BOTTOM_MARGIN = 740.0


def render_markdown(doc: pymupdf.Document, source_text: str) -> None:
    """
    Write the markdown source into a new document, one page at a time.

    Headings are set larger than the body and a line that would run past the
    page width is broken on a word boundary, so a table row stays one row and a
    number never ends up split from its unit.
    """
    page = doc[0]
    y = TOP_MARGIN

    def advance(amount):
        """Move down the page, starting a new one once the margin is passed."""
        nonlocal page, y
        y += amount
        if y > BOTTOM_MARGIN:
            page = doc.new_page()
            y = TOP_MARGIN

    for line in source_text.splitlines():
        if not line.strip():
            advance(10)
            continue
        fontsize = BODY_SIZE
        if line.startswith("## "):
            line = line[3:]
            fontsize = HEADING_SIZE
            advance(10)
        elif line.startswith("# "):
            line = line[2:]
            fontsize = TITLE_SIZE
            advance(8)
        # Wrap on a word boundary at roughly the page width, so a number and
        # its unit are never split across two lines.
        while len(line) > LINE_WIDTH:
            cut = line.rfind(" ", 0, LINE_WIDTH)
            head, line = line[:cut], line[cut + 1 :]
            page.insert_text((72, y), head, fontsize=fontsize)
            advance(fontsize + 6)
        page.insert_text((72, y), line, fontsize=fontsize)
        advance(fontsize + 6)


def build(source: Path) -> Path:
    """
    Render one markdown source to the PDF beside it, and return where it went.

    The metadata is pinned, so building an unchanged source twice produces the
    same bytes. The evaluation set pins each document by content hash, and a
    hash that moved on every build would send a reviewer back through every
    case for a change nobody made.
    """
    output = source.with_suffix(".pdf")
    doc = pymupdf.open()
    doc.new_page()
    render_markdown(doc, source.read_text(encoding="utf-8"))
    doc.set_metadata(
        {"creationDate": BUILD_DATE, "modDate": BUILD_DATE, "producer": "papermind"}
    )
    doc.save(output, no_new_id=True)
    doc.close()
    return output


if __name__ == "__main__":
    for path in SOURCES:
        print(f"wrote {build(path)}")
