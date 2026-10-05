# LexAgent v3.0 | chunking.py
"""Deterministic, provenance-preserving chunking for legal retrieval."""

from __future__ import annotations

import hashlib
from typing import Iterable, List

from src.data.corpus_schema import CorpusDocument


def chunk_document(
    document: CorpusDocument,
    chunk_words: int = 450,
    overlap_words: int = 75,
) -> List[CorpusDocument]:
    """Split a legal document into overlapping word chunks without losing provenance."""
    if chunk_words <= 0 or overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("Require chunk_words > overlap_words >= 0")
    words = document.text.split()
    if not words:
        return []
    step = chunk_words - overlap_words
    chunks: List[CorpusDocument] = []
    for index, start in enumerate(range(0, len(words), step)):
        text = " ".join(words[start:start + chunk_words])
        if not text:
            continue
        digest = hashlib.sha256(
            f"{document.doc_id}:{index}:{text}".encode("utf-8")
        ).hexdigest()[:16]
        chunks.append(CorpusDocument(
            doc_id=f"{document.doc_id}::chunk::{index}::{digest}",
            text=text,
            source=document.source,
            case_name=document.case_name,
            year=document.year,
            court=document.court,
            volume=document.volume,
            reporter=document.reporter,
            first_page=document.first_page,
            metadata={
                **document.metadata,
                "parent_doc_id": document.doc_id,
                "chunk_index": index,
                "word_start": start,
                "word_end": min(start + chunk_words, len(words)),
            },
        ))
        if start + chunk_words >= len(words):
            break
    return chunks


def chunk_corpus(documents: Iterable[CorpusDocument]) -> List[CorpusDocument]:
    """Return retrieval chunks for every source document."""
    return [chunk for doc in documents for chunk in chunk_document(doc)]
