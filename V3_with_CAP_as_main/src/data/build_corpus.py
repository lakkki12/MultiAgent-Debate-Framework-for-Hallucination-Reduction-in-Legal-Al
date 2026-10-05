# LexAgent v3.0 | build_corpus.py
"""Corpus loading and indexing utilities for LexAgent v3.0."""

import json
import os
from typing import List

from config import CORPUS_JSON_PATH
from src.data.corpus_schema import CorpusDocument


def load_corpus(path: str = CORPUS_JSON_PATH) -> List[CorpusDocument]:
    """Load canonical retrieval chunks from JSON file."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        return []
    return [CorpusDocument.from_dict(row) for row in rows]
