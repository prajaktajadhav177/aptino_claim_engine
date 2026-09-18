"""
Policy PDF ingestion.

Instead of splitting the PDF into arbitrary fixed-size pieces, we chunk it by the
policy document's own structure:
  1. We read the PDF page by page (so we always know the page number).
  2. We detect the policy's own section headings (DEFINITIONS, WHAT WE COVER,
     WHAT WE EXCLUDE, EXTENSIONS, CLAIMS PROCEDURE, STANDARD TERMS AND CONDITIONS, ...)
     using simple heuristics (short, upper-case-heavy lines).
  3. Inside the DEFINITIONS section we go one level deeper and split into one
     chunk PER DEFINED TERM (e.g. "Hospital means...", "Pre-Existing Diseases means...")
     because claim decisions usually hinge on a single definition, not the whole section.

Every chunk keeps: chunk_id, section, page (start page), and the raw text, so every
piece of retrieved evidence is traceable back to the source document.
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import List

import pdfplumber

# Known top-level headings in this policy document. Matching is case-insensitive
# and tolerant of minor OCR noise.
SECTION_HEADINGS = [
    "DEFINITIONS",
    "SCOPE OF COVER",
    "WHAT WE COVER",
    "WHAT WE EXCLUDE",
    "EXTENSIONS",
    "CLAIMS PROCEDURE",
    "STANDARD TERMS AND CONDITIONS",
    "CRITICAL ILLNESS",
]

# A defined term line looks like: "Hospital means any institution ..."
# or "Pre- Existing Diseases means any condition ..."
DEFINITION_LINE = re.compile(r"^([A-Z][A-Za-z\-/ ]{2,45}?)\s+means\b")


@dataclass
class Chunk:
    chunk_id: str
    section: str
    page: int
    text: str


def _make_chunk_id(section: str, text: str) -> str:
    h = hashlib.md5(f"{section}:{text[:80]}".encode("utf-8")).hexdigest()[:8]
    return f"chunk_{h}"


def _looks_like_heading(line: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None
    upper_ratio = sum(1 for c in stripped if c.isalpha() and c.isupper()) / max(
        1, sum(1 for c in stripped if c.isalpha())
    )
    for heading in SECTION_HEADINGS:
        if stripped.upper().startswith(heading) and upper_ratio > 0.7:
            return heading
    return None


def load_policy_pages(pdf_path: str) -> List[dict]:
    """Returns a list of {"page": n, "text": "..."} for every page of the PDF."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append({"page": i, "text": text})
    return pages


def chunk_policy(pdf_path: str) -> List[Chunk]:
    pages = load_policy_pages(pdf_path)

    # 1. Walk the whole document, tagging every line with the section heading
    #    that is currently active and the page it appeared on.
    tagged_lines = []  # (section, page, line)
    current_section = "PREAMBLE"
    for page in pages:
        for line in page["text"].split("\n"):
            heading = _looks_like_heading(line)
            if heading:
                current_section = heading
                continue  # heading line itself is not content
            tagged_lines.append((current_section, page["page"], line))

    # 2. Group consecutive lines belonging to the same section into one chunk,
    #    remembering the first page number seen for that run.
    chunks: List[Chunk] = []
    buffer: List[str] = []
    buffer_section = None
    buffer_page = None

    def flush():
        nonlocal buffer, buffer_section, buffer_page
        text = "\n".join(buffer).strip()
        if text:
            if buffer_section == "DEFINITIONS":
                chunks.extend(_split_definitions(text, buffer_page))
            else:
                chunks.append(
                    Chunk(
                        chunk_id=_make_chunk_id(buffer_section, text),
                        section=buffer_section,
                        page=buffer_page,
                        text=text,
                    )
                )
        buffer = []

    for section, page, line in tagged_lines:
        if section != buffer_section:
            flush()
            buffer_section = section
            buffer_page = page
        if buffer_page is None:
            buffer_page = page
        buffer.append(line)
    flush()

    return [c for c in chunks if c.text]


def _split_definitions(text: str, page: int) -> List[Chunk]:
    """Split the DEFINITIONS section into one chunk per defined term."""
    lines = text.split("\n")
    terms: List[List[str]] = []
    current_term = None

    for line in lines:
        match = DEFINITION_LINE.match(line.strip())
        if match:
            current_term = [line]
            terms.append(current_term)
        elif current_term is not None:
            current_term.append(line)
        # lines before the first defined term (intro text) are dropped;
        # they carry no decision-relevant content.

    chunks = []
    for term_lines in terms:
        term_text = "\n".join(term_lines).strip()
        term_name = DEFINITION_LINE.match(term_lines[0].strip()).group(1).strip()
        chunks.append(
            Chunk(
                chunk_id=_make_chunk_id(f"DEFINITIONS/{term_name}", term_text),
                section=f"DEFINITIONS / {term_name}",
                page=page,
                text=term_text,
            )
        )
    return chunks


if __name__ == "__main__":
    import sys

    chunks = chunk_policy(sys.argv[1] if len(sys.argv) > 1 else "data/policy/policy.pdf")
    print(f"Total chunks: {len(chunks)}")
    for c in chunks[:10]:
        print(c.chunk_id, "|", c.section, "| page", c.page, "|", c.text[:60].replace("\n", " "))
