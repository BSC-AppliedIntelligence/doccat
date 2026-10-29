"""Text extraction utilities shared by the indexer and the MCP server.

Supports PDF (pypdfium2), DOCX (python-docx), PPTX (python-pptx),
XLSX (openpyxl), Markdown, plain text, and HTML (stdlib parser).
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import threading
import zipfile
from html.parser import HTMLParser
from pathlib import Path


class LegacyFormatError(Exception):
    """Legacy binary Office file (OLE2: .doc/.ppt/.xls) that modern parsers can't read."""

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx",
                        ".md", ".markdown", ".txt", ".html", ".htm"}

MIME_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
    ".html": "text/html",
    ".htm": "text/html",
}


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Content hash used to detect changed files between index runs."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def doc_id_for(path: Path, root: Path) -> str:
    """Stable document id derived from the path relative to the root folder."""
    rel = str(path.relative_to(root))
    return hashlib.sha1(rel.encode("utf-8")).hexdigest()[:16]


# PDFium is not thread-safe: serialise calls within a process (the indexer's
# extraction pool uses processes, so this only matters for the MCP server).
_PDFIUM_LOCK = threading.Lock()


def extract_pdf(path: Path, first_page: int = 0, last_page: int | None = None) -> tuple[str, int]:
    """Extract text from a PDF. Returns (text, total_pages). Pages are 0-based, last_page inclusive."""
    import pypdfium2 as pdfium  # BSD-3-Clause / Apache-2.0 (PDFium: BSD-3-Clause)

    with _PDFIUM_LOCK:
        doc = pdfium.PdfDocument(str(path))
        try:
            total = len(doc)
            stop = total - 1 if last_page is None else min(last_page, total - 1)
            parts = []
            for i in range(max(first_page, 0), stop + 1):
                page = doc[i]
                textpage = page.get_textpage()
                try:
                    text = textpage.get_text_bounded().replace("\r\n", "\n").replace("\r", "\n")
                finally:
                    textpage.close()
                    page.close()
                parts.append(f"[page {i + 1}]\n{text}")
            return "\n".join(parts), total
        finally:
            doc.close()


def extract_docx(path: Path) -> tuple[str, int]:
    """Extract text from a DOCX (paragraphs + tables). Returns (text, paragraph_count)."""
    import docx

    d = docx.Document(str(path))
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for table in d.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return "\n".join(parts), len(parts)


def extract_pptx(path: Path) -> tuple[str, int]:
    """Extract text from a PPTX (slide by slide, including notes). Returns (text, slide_count)."""
    from pptx import Presentation

    prs = Presentation(str(path))
    parts = []
    for i, slide in enumerate(prs.slides, 1):
        lines = [f"[slide {i}]"]
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    text = "".join(run.text for run in para.runs).strip()
                    if text:
                        lines.append(text)
        notes = getattr(slide, "notes_slide", None)
        if slide.has_notes_slide and notes and notes.notes_text_frame and notes.notes_text_frame.text.strip():
            lines.append(f"[notes] {notes.notes_text_frame.text.strip()}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts), len(prs.slides)


def extract_xlsx(path: Path, max_rows_per_sheet: int = 200) -> tuple[str, int]:
    """Extract text from an XLSX (sheet by sheet, row-limited). Returns (text, sheet_count)."""
    from openpyxl import load_workbook

    wb = load_workbook(str(path), read_only=True, data_only=True)
    parts = []
    for ws in wb.worksheets:
        lines = [f"[sheet: {ws.title}]"]
        for r, row in enumerate(ws.iter_rows(values_only=True)):
            if r >= max_rows_per_sheet:
                lines.append(f"[... truncated at {max_rows_per_sheet} rows ...]")
                break
            cells = ["" if c is None else str(c) for c in row]
            if any(c.strip() for c in cells):
                lines.append(" | ".join(cells).rstrip(" |"))
        parts.append("\n".join(lines))
    n = len(wb.worksheets)
    wb.close()
    return "\n\n".join(parts), n


class _HTMLTextParser(HTMLParser):
    """Minimal HTML→text converter (stdlib only). Skips script/style, keeps block breaks."""

    _SKIP = {"script", "style", "noscript", "template", "head"}
    _BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
              "section", "article", "table", "ul", "ol", "blockquote", "pre"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and data.strip():
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        lines = [ln.strip() for ln in raw.splitlines()]
        return "\n".join(ln for ln in lines if ln)


def extract_html(path: Path) -> tuple[str, int]:
    """Extract visible text from an HTML file. Returns (text, line_count)."""
    parser = _HTMLTextParser()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    text = parser.text()
    return text, text.count("\n") + 1


def extract_plain(path: Path) -> tuple[str, int]:
    """Read a Markdown/plain-text file as-is. Returns (text, line_count)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text, text.count("\n") + 1


def sniff_kind(path: Path) -> str | None:
    """Detect the REAL file format from magic bytes/content, ignoring the extension.

    Bulk dumps often contain misnamed files (e.g. a legacy binary .ppt renamed
    to .pptx). Returns one of: 'pdf', 'docx', 'pptx', 'xlsx', 'ole2', 'html',
    or None (unknown / plain text).
    """
    try:
        head = path.open("rb").read(1024)
    except OSError:
        return None
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"\xd0\xcf\x11\xe0"):  # OLE2: legacy .doc/.ppt/.xls
        return "ole2"
    if head.startswith(b"PK\x03\x04"):  # ZIP: OOXML container
        try:
            with zipfile.ZipFile(path) as z:
                names = set(z.namelist())
                if any(n.startswith("word/") for n in names):
                    return "docx"
                if any(n.startswith("ppt/") for n in names):
                    return "pptx"
                if any(n.startswith("xl/") for n in names):
                    return "xlsx"
        except zipfile.BadZipFile:
            return None
        return None
    lowered = head.lstrip()[:100].lower()
    if lowered.startswith((b"<!doctype html", b"<html")):
        return "html"
    return None


def convert_legacy_to_pdf(path: Path, timeout: float = 180.0) -> Path | None:
    """Convert a legacy Office file to PDF with LibreOffice, if installed.

    Returns the converted PDF path (in a temp dir) or None when LibreOffice
    is not available or the conversion fails.
    """
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        return None
    tmpdir = Path(tempfile.mkdtemp(prefix="legacy_convert_"))
    try:
        subprocess.run(
            [soffice, "--headless", "--convert-to", "pdf", "--outdir", str(tmpdir), str(path)],
            capture_output=True, timeout=timeout, check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    pdfs = list(tmpdir.glob("*.pdf"))
    return pdfs[0] if pdfs else None


def extract_text(path: Path, first_page: int = 0, last_page: int | None = None) -> tuple[str, int]:
    """Extract text, trusting content sniffing over the file extension.

    Returns (text, unit_count: pages/paragraphs/slides/sheets/lines).
    Raises LegacyFormatError for OLE2 files when LibreOffice is unavailable.
    """
    kind = sniff_kind(path)

    if kind == "ole2":
        converted = convert_legacy_to_pdf(path)
        if converted is not None:
            return extract_pdf(converted, first_page, last_page)
        raise LegacyFormatError(
            f"{path.name} is a legacy binary Office file (.doc/.ppt/.xls, possibly renamed). "
            "Install LibreOffice for automatic conversion, or convert it manually to a modern format."
        )
    if kind == "pdf":
        return extract_pdf(path, first_page, last_page)
    if kind == "docx":
        return extract_docx(path)
    if kind == "pptx":
        return extract_pptx(path)
    if kind == "xlsx":
        return extract_xlsx(path)
    if kind == "html":
        return extract_html(path)

    # Unknown magic: fall back to the extension for text-like formats.
    ext = path.suffix.lower()
    if ext in (".html", ".htm"):
        return extract_html(path)
    if ext in (".md", ".markdown", ".txt"):
        return extract_plain(path)
    raise ValueError(f"Unsupported or unrecognized file type: {path.name} "
                     f"(extension {ext!r}, detected content {kind!r})")
