"""PDF extraction: page markers, page ranges, page counts and content sniffing.

The PDF is assembled from raw PDF syntax so the test needs no PDF-writing library.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import extractors  # noqa: E402

pytest.importorskip("pypdfium2")


def _minimal_pdf(pages: list[str]) -> bytes:
    """A valid PDF with one line of Helvetica text per page."""
    objs: list[bytes] = []
    n = len(pages)
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    font_id = 3 + 2 * n
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    for i, text in enumerate(pages):
        page_id, content_id = 3 + 2 * i, 4 + 2 * i
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode()
        )
        stream = f"BT /F1 18 Tf 72 720 Td ({text}) Tj ET".encode()
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for num, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % num + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % off for off in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(out)


@pytest.fixture
def three_page_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "sample.pdf"
    path.write_bytes(_minimal_pdf(["Alpha page one", "Bravo page two", "Charlie page three"]))
    return path


def test_extract_pdf_all_pages(three_page_pdf: Path):
    text, total = extractors.extract_pdf(three_page_pdf)
    assert total == 3
    assert text.count("[page ") == 3
    for marker, word in [("[page 1]", "Alpha"), ("[page 2]", "Bravo"), ("[page 3]", "Charlie")]:
        assert marker in text and word in text
    assert text.index("Alpha") < text.index("Bravo") < text.index("Charlie")


def test_extract_pdf_page_range(three_page_pdf: Path):
    text, total = extractors.extract_pdf(three_page_pdf, first_page=1, last_page=1)
    assert total == 3
    assert "[page 2]" in text and "Bravo" in text
    assert "Alpha" not in text and "Charlie" not in text


def test_extract_pdf_range_past_end_is_clamped(three_page_pdf: Path):
    text, total = extractors.extract_pdf(three_page_pdf, first_page=2, last_page=99)
    assert total == 3
    assert text.count("[page ") == 1 and "Charlie" in text


def test_extract_text_sniffs_pdf_despite_extension(tmp_path: Path, three_page_pdf: Path):
    misnamed = tmp_path / "report.txt"
    misnamed.write_bytes(three_page_pdf.read_bytes())
    text, units = extractors.extract_text(misnamed)
    assert units == 3 and "Bravo" in text


def test_no_agpl_pdf_dependency():
    """doccat must not depend on PyMuPDF (AGPL-3.0); see the licence audit."""
    source = Path(extractors.__file__).read_text(encoding="utf-8")
    assert "import fitz" not in source and "import pymupdf" not in source
