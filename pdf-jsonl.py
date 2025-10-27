#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF ➜ Markdown ➜ JSONL converter (publication‑ready)

Highlights vs. your original script
- Robust Abstract detection (ABSTRACT → next heading / REFERENCES / KEYWORDS)
- Dual‑language sentence splitting (Chinese & English) without NLTK
- Cleans common in‑text citations: (Author, 2018), (Author et al., 2018), [12], [3–7]
- Skips reference section; optional "--full-text" to disable
- Deterministic IDs (hash of source path + incremental index)
- CLI with argparse; proper Path handling; safe output paths
- Optional "--save-md" to keep the intermediate Markdown for debugging
- Detects likely scanned PDFs (empty text) and warns

Usage
------
python pdf_to_jsonl.py \
  --input H:\\AAPG_filtered_llm-2 \
  --output H:\\jsonl_out \
  --pattern "*.pdf" \
  --min-len 20 \
  --save-md

JSONL schema per line
---------------------
{
  "uid": "a1b2c3d4-00001",       # deterministic unique id
  "text": "... sentence ...",     # cleaned sentence
  "text_len": 123,                 # char length
  "doc_id": "a1b2c3d4",          # hash(base_path+filename)[:8]
  "source_file": "paper.pdf",     # filename only
  "pos_in_doc": [ [0, 122] ]       # conservative span (synthetic)
}

License note
------------
This file is suitable for open‑sourcing under MIT or Apache‑2.0. Make sure to
include a LICENSE at repo root and exclude any API keys from other scripts.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import re
from pathlib import Path
from typing import List, Tuple

try:
    # PyPDF2 is widely available; keep dependency minimal
    import PyPDF2
except ImportError as e:
    raise SystemExit("PyPDF2 is required. pip install PyPDF2") from e

try:
    import jsonlines  # fast JSONL writer
except ImportError as e:
    raise SystemExit("jsonlines is required. pip install jsonlines") from e

# -------- Sentence splitting patterns (EN + ZH) --------
# English: split on . ! ? when followed by space/newline + uppercase or digit
_EN_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
# Chinese: split on 。！？
_ZH_SPLIT = re.compile(r"(?<=[。！？])")

# Common false positive abbreviations in English to avoid over‑splitting
_EN_ABBR = {
    "e.g.", "i.e.", "al.", "et al.", "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.",
    "Fig.", "Figs.", "Eq.", "Eqs.", "No.", "Nos.", "vs.", "etc.", "cf.",
}

# Headings that often delimit abstract
_ABSTRACT_KEYS = [
    "ABSTRACT", "SUMMARY", "ABSTRAKT", "ABRÉGÉ", "RESUMEN",
]
_NEXT_SECTION_HINTS = [
    "KEYWORDS", "INDEX TERMS", "HIGHLIGHTS", "INTRODUCTION", "1.",
    "REFERENCES", "ACKNOWLEDGEMENTS", "ACKNOWLEDGMENTS",
]

# Patterns to clean citations and references
_PAREN_CITATION = re.compile(r"\((?:[A-Z][A-Za-z\-']+(?:\s+[A-Z][A-Za-z\-']+)*)\s*(?:et\s+al\.)?,?\s*\d{4}[a-z]?\)" )
_NUM_CITATION = re.compile(r"\[(?:\d{1,3})(?:\s*[,\-–]\s*\d{1,3})*\]")
_DOI_URL = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
_MULTISPACE = re.compile(r"[\t \u00A0\u2009]{2,}")


def _read_pdf_text(pdf_path: Path) -> str:
    """Extract plain text from PDF using PyPDF2. Return concatenated pages."""
    text_parts: List[str] = []
    with pdf_path.open("rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            text_parts.append(t)
    return "\n".join(text_parts)


def _find_abstract_span(full_text: str) -> Tuple[int, int]:
    """Return (start, end) of the abstract region within full_text.
    If no abstract heading found, return (0, len(full_text)).
    """
    U = full_text.upper()
    start = -1
    for key in _ABSTRACT_KEYS:
        s = U.find(key)
        if s != -1 and (start == -1 or s < start):
            start = s
    if start == -1:
        return 0, len(full_text)

    # Find the next section boundary after start
    end = len(full_text)
    for hint in _NEXT_SECTION_HINTS:
        h = U.find(hint, start + 8)
        if h != -1:
            end = min(end, h)

    # Also stop at a likely ALL‑CAPS heading on its own line
    caps_heading = re.search(r"\n\s*[A-Z][A-Z0-9 \-]{3,}\s*\n", U[start+8:])
    if caps_heading:
        end = min(end, start + 8 + caps_heading.start())

    # Stop at REFERENCES if it appears earlier than other hints
    ref_idx = U.find("REFERENCES", start)
    if ref_idx != -1:
        end = min(end, ref_idx)

    return start, max(start + 200, end)  # ensure minimum length


def _truncate_at_references(text: str) -> str:
    U = text.upper()
    k = U.find("REFERENCES")
    if k != -1:
        return text[:k]
    return text


def _clean_text(s: str) -> str:
    # Remove in‑text citations & DOIs
    s = _PAREN_CITATION.sub("", s)
    s = _NUM_CITATION.sub("", s)
    s = _DOI_URL.sub("", s)
    # Collapse excessive spaces
    s = _MULTISPACE.sub(" ", s)
    # Trim noisy hyphenations across linebreaks
    s = re.sub(r"-\n", "", s)
    s = re.sub(r"\n+", " ", s)
    return s.strip()


def _split_sentences(text: str) -> List[str]:
    """Heuristic sentence splitter for English + Chinese."""
    if not text:
        return []
    # First, split by Chinese punctuation while keeping the separator as end
    parts: List[str] = []
    for chunk in _ZH_SPLIT.split(text):
        if not chunk:
            continue
        # Now split English within this chunk
        subparts = _EN_SPLIT.split(chunk)
        parts.extend(subparts)

    # Post‑process: merge pieces that are abbreviations at the tail
    merged: List[str] = []
    buffer = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if any(p.endswith(abbr) for abbr in _EN_ABBR):
            buffer = (buffer + " " + p).strip()
            continue
        if buffer:
            merged.append((buffer + " " + p).strip())
            buffer = ""
        else:
            merged.append(p)
    if buffer:
        merged.append(buffer)

    # Final tidy
    return [m.strip() for m in merged if m.strip()]


def pdf_to_markdown(pdf_path: Path, abstract_only: bool = True) -> str:
    full_text = _read_pdf_text(pdf_path)
    if not full_text or len(full_text.strip()) < 30:
        print(f"[WARN] Likely scanned or empty text: {pdf_path.name}")
        return ""

    if abstract_only:
        s, e = _find_abstract_span(full_text)
        text = full_text[s:e]
    else:
        text = _truncate_at_references(full_text)

    text = _clean_text(text)
    # Convert to Markdown paragraphs (sentences separated by blank line)
    sentences = _split_sentences(text)
    return "\n\n".join(sentences)


def markdown_to_jsonl(markdown_content: str, jsonl_path: Path, doc_id: str, source_filename: str, min_len: int = 20) -> int:
    paragraphs = [p.strip() for p in markdown_content.split("\n\n") if p.strip()]
    idx = 1
    written = 0
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(str(jsonl_path), mode="w") as writer:
        for paragraph in paragraphs:
            for sentence in _split_sentences(paragraph):
                s = sentence.strip()
                if not s or len(s) < min_len:
                    continue
                entry = {
                    "uid": f"{doc_id}-{idx:05d}",
                    "text_len": len(s),
                    "text": s,
                    "doc_id": doc_id,
                    "pos_in_doc": [[0, len(s) - 1]],  # conservative
                    "source_file": source_filename,
                }
                writer.write(entry)
                idx += 1
                written += 1
    return written


def process_single_pdf(pdf_path: Path, output_dir: Path, save_md: bool = False, min_len: int = 20, abstract_only: bool = True) -> int:
    source_filename = pdf_path.name
    # Deterministic short doc id from absolute path
    doc_id = hashlib.sha1(str(pdf_path.resolve()).encode("utf-8")).hexdigest()[:8]

    md = pdf_to_markdown(pdf_path, abstract_only=abstract_only)
    if not md:
        print(f"[SKIP] No extractable text: {source_filename}")
        return 0

    if save_md:
        md_path = output_dir / f"{pdf_path.stem}.md"
        try:
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text(md, encoding="utf-8")
        except Exception as e:
            print(f"[WARN] Failed to save markdown {md_path.name}: {e}")

    jsonl_path = output_dir / f"{pdf_path.stem}.jsonl"
    count = markdown_to_jsonl(md, jsonl_path, doc_id, source_filename, min_len=min_len)
    print(f"[OK] {source_filename} -> {count} entries")
    return count


def batch_convert_pdfs(input_dir: Path, output_dir: Path, file_pattern: str = "*.pdf", save_md: bool = False, min_len: int = 20, abstract_only: bool = True) -> Tuple[int, int, int]:
    pdf_files = [Path(p) for p in glob.glob(str(input_dir / file_pattern))]
    if not pdf_files:
        print(f"[INFO] No files matched {file_pattern} in {input_dir}")
        return 0, 0, 0

    total_files = len(pdf_files)
    ok_files = 0
    total_entries = 0

    print(f"[INFO] Found {total_files} PDFs. Starting conversion…")
    print("-" * 60)

    for p in pdf_files:
        try:
            n = process_single_pdf(p, output_dir, save_md=save_md, min_len=min_len, abstract_only=abstract_only)
            if n > 0:
                ok_files += 1
                total_entries += n
        except Exception as e:
            print(f"[ERR] {p.name}: {e}")
        print("-" * 40)

    print("[DONE] Batch finished.")
    print(f"[STAT] Files total: {total_files} | succeeded: {ok_files} | failed: {total_files - ok_files}")
    print(f"[STAT] Entries written: {total_entries}")
    return total_files, ok_files, total_entries


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Convert PDFs to JSONL with optional Markdown debug output.")
    ap.add_argument("--input", required=True, type=Path, help="Input directory of PDFs")
    ap.add_argument("--output", required=True, type=Path, help="Output directory for JSONL/MD")
    ap.add_argument("--pattern", default="*.pdf", help="Glob pattern, e.g., '*.pdf' or '*paper*.pdf'")
    ap.add_argument("--save-md", action="store_true", help="Save intermediate Markdown files")
    ap.add_argument("--min-len", type=int, default=20, help="Minimum sentence length to keep")
    ap.add_argument("--full-text", action="store_true", help="Use full text (truncate at REFERENCES) instead of ABSTRACT only")
    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()

    input_dir: Path = args.input
    output_dir: Path = args.output
    pattern: str = args.pattern
    save_md: bool = args.save_md
    min_len: int = args.min_len
    abstract_only: bool = not args.full_text

    if not input_dir.exists():
        raise SystemExit(f"Input dir does not exist: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_convert_pdfs(input_dir, output_dir, file_pattern=pattern, save_md=save_md, min_len=min_len, abstract_only=abstract_only)


if __name__ == "__main__":
    main()
