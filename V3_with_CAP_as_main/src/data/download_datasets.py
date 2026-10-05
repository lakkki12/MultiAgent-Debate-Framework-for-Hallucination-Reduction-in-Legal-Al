# LexAgent v3.0 | download_datasets.py
"""Evaluation datasets download pipeline for LexAgent v3.0 (CaseHOLD & LegalBench)."""

import json
import os
import sys
from typing import Dict, List, Any

from config import CASEHOLD_DATASET_PATH, LEGALBENCH_DATASET_PATH, DRIVE_BASE
from src.utils.logger import get_logger

logger = get_logger(__name__)


def download_casehold(save_path: str = CASEHOLD_DATASET_PATH, split: str = "test", limit: int = 1000) -> str:
    """Download CaseHOLD dataset for Citation Accuracy Score (CAS) evaluation."""
    if os.path.exists(save_path):
        logger.info("CaseHOLD already exists at: %s", save_path)
        return save_path

    from datasets import load_dataset

    logger.info("Downloading CaseHOLD (coastalcph/lex_glue, split=%s, limit=%s)...", split, limit)
    try:
        ds = load_dataset("coastalcph/lex_glue", "case_hold", split=split)
    except Exception as e:
        logger.warning("First attempt failed for CaseHOLD: %s. Retrying...", e)
        ds = load_dataset("coastalcph/lex_glue", "case_hold", split=split)

    total_len = len(ds)
    n = min(limit, total_len) if limit else total_len
    samples = []

    for i in range(n):
        item = dict(ds[i])
        label = item.get("label", 0)
        endings = item.get("endings", [])
        true_holding = endings[label] if (isinstance(endings, list) and 0 <= label < len(endings)) else ""
        item["answer"] = true_holding
        item["holding_0"] = true_holding
        samples.append(item)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    logger.info("CaseHOLD saved: %d samples -> %s", len(samples), save_path)
    return save_path


def download_legalbench(save_path: str = LEGALBENCH_DATASET_PATH) -> str:
    """Download LegalBench evaluation subtasks."""
    if os.path.exists(save_path):
        logger.info("LegalBench already exists at: %s", save_path)
        return save_path

    from datasets import load_dataset

    subtasks = ["contract_qa", "rule_qa"]
    samples_per_task = 100
    all_samples: List[Dict[str, Any]] = []

    for subtask in subtasks:
        logger.info("Downloading LegalBench/%s...", subtask)
        try:
            ds = load_dataset("nguha/legalbench", subtask, split="test", trust_remote_code=True)
        except Exception as e:
            logger.warning("Failed downloading LegalBench/%s: %s", subtask, e)
            continue

        n = min(samples_per_task, len(ds))
        for i in range(n):
            sample = dict(ds[i])
            sample["subtask"] = subtask
            all_samples.append(sample)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    logger.info("LegalBench saved: %d samples -> %s", len(all_samples), save_path)
    return save_path
