# LexAgent v3.0 | cap_ingest.py
"""
CAP (Caselaw Access Project) Multi-Volume Ingestion Engine.
Supports parallel multi-threaded downloading, volume-level checkpointing,
and auto-resume across 1 to 600+ volumes.
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Iterable, List, Set, Tuple

import requests

from config import (
    CAP_BASE_URL,
    CAP_RAW_DIR,
    CAP_CHUNKS_DIR,
    CORPUS_JSON_PATH,
    CORPUS_MANIFEST_PATH,
    KNOWN_CASES_PATH,
    LANDMARK_CAP_VOLUMES,
)
from src.data.chunking import chunk_corpus
from src.data.corpus_schema import CorpusDocument
from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_doc_volume(doc) -> str:
    """Robustly extract volume number string from a CorpusDocument or dict."""
    vol = getattr(doc, "volume", None) if hasattr(doc, "volume") else (doc.get("volume") if isinstance(doc, dict) else None)
    if isinstance(vol, dict):
        return str(vol.get("volume_number") or vol.get("volume") or "").strip()
    if vol:
        s = str(vol).strip()
        m = re.search(r"['\"]?volume_number['\"]?\s*:\s*['\"]?(\d+)['\"]?", s)
        if m:
            return m.group(1)
        m = re.match(r"^\d+$", s)
        if m:
            return s
    meta = getattr(doc, "metadata", {}) if hasattr(doc, "metadata") else (doc.get("metadata", {}) if isinstance(doc, dict) else {})
    if isinstance(meta, dict):
        m_vol = meta.get("volume")
        if isinstance(m_vol, dict):
            return str(m_vol.get("volume_number") or "").strip()
        if m_vol:
            s = str(m_vol).strip()
            m = re.search(r"['\"]?volume_number['\"]?\s*:\s*['\"]?(\d+)['\"]?", s)
            if m:
                return m.group(1)
            m = re.match(r"^\d+$", s)
            if m:
                return s
        for c in meta.get("citations", []):
            c_str = c.get("cite", "") if isinstance(c, dict) else str(c)
            m = re.search(r"(\d+)\s+U\.?\s*S\.?", c_str, re.IGNORECASE)
            if m:
                return m.group(1)
    doc_id = getattr(doc, "doc_id", "") if hasattr(doc, "doc_id") else (doc.get("doc_id", "") if isinstance(doc, dict) else "")
    m = re.search(r"(?:vol_?|CAP_?)(\d+)", doc_id, re.IGNORECASE)
    if m:
        return m.group(1)
    text_sample = f"{getattr(doc, 'case_name', '')} {getattr(doc, 'text', '')[:200]}" if hasattr(doc, "text") else ""
    m = re.search(r"(\d+)\s+U\.?\s*S\.?", text_sample, re.IGNORECASE)
    if m:
        return m.group(1)
    return ""


def parse_volume_selection(mode_or_range: str) -> List[str]:
    """Parse a volume specification into a sorted list of volume strings.
    
    Supported formats:
      - "landmark": Preset of 27 landmark constitutional volumes.
      - "modern": Volumes 300 to 585 (~1940 to 2018).
      - "all": Volumes 1 to 600+.
      - "400-450": Range from 400 to 450 inclusive.
      - "410,347,384": Explicit comma-separated volume list.
    """
    cleaned = str(mode_or_range).strip().lower()

    if cleaned == "landmark":
        return list(LANDMARK_CAP_VOLUMES)

    if cleaned == "modern":
        return [str(v) for v in range(300, 586)]

    if cleaned == "all":
        return [str(v) for v in range(1, 601)]

    if "-" in cleaned:
        parts = cleaned.split("-")
        try:
            start, end = int(parts[0]), int(parts[1])
            return [str(v) for v in range(start, end + 1)]
        except ValueError:
            pass

    if "," in cleaned:
        return [v.strip() for v in cleaned.split(",") if v.strip()]

    # Single volume
    return [cleaned]


def _text_from_casebody(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return ""
    body = value.get("data", value)
    if isinstance(body, str):
        return body
    if not isinstance(body, dict):
        return ""
    parts = [str(body.get("head_matter", ""))]
    for opinion in body.get("opinions", []) or []:
        if isinstance(opinion, dict):
            parts.append(str(opinion.get("text", "")))
    return "\n\n".join(part for part in parts if part).strip()


def _case_url(record: dict, volume: str) -> str:
    for key in ("casebody_url", "case_url", "download_url"):
        if record.get(key):
            return record[key]
    direct_url = record.get("url")
    if isinstance(direct_url, str) and direct_url.endswith(".json"):
        return direct_url
    casebody = record.get("casebody")
    if isinstance(casebody, dict) and casebody.get("url"):
        return casebody["url"]
    for key in ("case_file_name", "file_name", "filename"):
        filename = record.get(key)
        if filename:
            filename = str(filename)
            if filename.startswith("http"):
                return filename
            if not filename.endswith(".json"):
                filename += ".json"
            return f"{CAP_BASE_URL}/us/{volume}/cases/{filename.lstrip('/')}"
    first_page = record.get("first_page")
    if first_page is not None:
        try:
            return f"{CAP_BASE_URL}/us/{volume}/cases/{int(first_page):04d}-01.json"
        except (TypeError, ValueError):
            pass
    raise ValueError(f"CAP metadata has no static case filename for record {record.get('id')}")


def _download_single_case(record: dict, volume: str, timeout: int = 30) -> dict | None:
    try:
        url = _case_url(record, volume)
        res = requests.get(url, timeout=timeout)
        res.raise_for_status()
        full = res.json()
        merged = {**record, **full}
        casebody = merged.get("casebody", full)
        if _text_from_casebody(casebody):
            merged["casebody"] = casebody
            return merged
    except Exception:
        pass
    return None


def get_downloaded_volumes(raw_dir: str = CAP_RAW_DIR) -> Tuple[Set[str], int]:
    """Scan raw_dir for existing downloaded CAP volume files.
    
    Returns:
        (set_of_downloaded_volume_strings, max_volume_int)
    """
    if not os.path.exists(raw_dir):
        return set(), 0

    downloaded = set()
    max_vol = 0

    for fname in os.listdir(raw_dir):
        if fname.startswith("cap_raw_us_") and fname.endswith(".json"):
            vol_str = fname[len("cap_raw_us_"):-len(".json")]
            fpath = os.path.join(raw_dir, fname)
            try:
                if os.path.getsize(fpath) > 10:
                    downloaded.add(vol_str)
                    if vol_str.isdigit():
                        max_vol = max(max_vol, int(vol_str))
            except OSError:
                pass

    return downloaded, max_vol


def download_cap_volume(volume: str, raw_dir: str = CAP_RAW_DIR, max_workers: int = 10, timeout: int = 45) -> list[dict]:
    """Download or load a single checkpointed CAP volume.
    
    If raw JSON file exists, loads instantly (checkpoint resume).
    Otherwise, downloads cases using a thread worker pool.
    """
    os.makedirs(raw_dir, exist_ok=True)
    cache_file = os.path.join(raw_dir, f"cap_raw_us_{volume}.json")

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("status") == "not_found":
                    return []
                cases = data.get("cases", [])
                if cases:
                    logger.debug("Loaded Volume %s from cache (%d cases).", volume, len(cases))
                    return cases
        except Exception as e:
            logger.warning("Cache load failed for volume %s: %s. Re-fetching...", volume, e)

    metadata_url = f"{CAP_BASE_URL}/us/{volume}/CasesMetadata.json"
    try:
        response = requests.get(metadata_url, timeout=timeout)
        response.raise_for_status()
        metadata = response.json()
    except Exception as e:
        logger.warning("Could not fetch metadata for Volume %s from %s: %s", volume, metadata_url, e)
        # Write sentinel so we remember this volume was checked and does not exist in CAP
        try:
            with open(cache_file, "w", encoding="utf-8") as handle:
                json.dump({"volume": volume, "cases": [], "status": "not_found"}, handle)
        except Exception:
            pass
        return []

    if not isinstance(metadata, list):
        return []

    cases: list[dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_download_single_case, rec, volume, timeout): rec
            for rec in metadata
        }
        for future in as_completed(future_map):
            result = future.result()
            if result:
                cases.append(result)

    with open(cache_file, "w", encoding="utf-8") as handle:
        json.dump({"volume": volume, "cases": cases}, handle, ensure_ascii=False)

    logger.info("Volume %s: %d cases downloaded and cached.", volume, len(cases))
    return cases


def download_cap_volumes(
    volumes: List[str],
    raw_dir: str = CAP_RAW_DIR,
    max_workers: int = 10,
) -> list[dict]:
    """Download multiple CAP volumes with pre-check, resuming from the latest downloaded volume."""
    os.makedirs(raw_dir, exist_ok=True)
    downloaded_set, max_vol = get_downloaded_volumes(raw_dir)

    to_download = [v for v in volumes if v not in downloaded_set]
    already_downloaded_count = len(volumes) - len(to_download)

    print("\n" + "=" * 78)
    print("  [CAP VOLUMES DOWNLOAD PRE-CHECK & RESUME]")
    print(f"    Raw Volumes Directory: {raw_dir}")
    print(f"    Target Volumes:        {len(volumes)} total ({volumes[0]} ... {volumes[-1]})")
    print(f"    Already Downloaded:    {already_downloaded_count} volumes on disk (highest: Vol {max_vol})")
    print(f"    Remaining to Download: {len(to_download)} volumes")
    print("=" * 78)

    if not to_download:
        print(f"  [OK] All requested volumes (up to Vol {max_vol}) are ALREADY downloaded on disk!")
        print("  [OK] Skipping network download. Loading cases from local disk cache...")
    else:
        print(f"  -> Resuming download starting from next missing volume: Vol {to_download[0]} ({len(to_download)} remaining)...")
        for i, vol in enumerate(to_download, 1):
            download_cap_volume(vol, raw_dir=raw_dir, max_workers=max_workers)
            if i % 10 == 0 or i == len(to_download):
                print(f"     Downloaded [{i}/{len(to_download)}] missing volumes...")

    # Load cases from cache
    all_cases: list[dict] = []
    print(f"\n  -> Loading cases from {len(volumes)} volume caches...")
    for i, vol in enumerate(volumes, 1):
        cases = download_cap_volume(vol, raw_dir=raw_dir, max_workers=max_workers)
        all_cases.extend(cases)
        if i % 50 == 0 or i == len(volumes):
            print(f"     Loaded [{i}/{len(volumes)}] volumes | Total cases: {len(all_cases):,}")

    return all_cases


import pickle


def extract_known_cases_from_chunks(chunks: Iterable[Any]) -> Set[str]:
    """Extract case names and citation references from chunks for CCE verification."""
    known_cases: Set[str] = set()
    for chunk in chunks:
        if isinstance(chunk, dict):
            c_name = chunk.get("case_name") or ""
            vol = str(chunk.get("volume") or "")
            rep = chunk.get("reporter") or "U.S."
            page = chunk.get("first_page") or 0
            citations = (chunk.get("metadata") or {}).get("citations", [])
        else:
            c_name = getattr(chunk, "case_name", "") or ""
            vol = str(getattr(chunk, "volume", "") or "")
            rep = getattr(chunk, "reporter", "U.S.") or "U.S."
            page = getattr(chunk, "first_page", 0) or 0
            citations = (getattr(chunk, "metadata", {}) or {}).get("citations", [])

        if c_name:
            known_cases.add(c_name.strip())
        if vol and page:
            known_cases.add(f"{vol} {rep} {page}".strip())
        for cite_item in citations:
            if isinstance(cite_item, dict) and cite_item.get("cite"):
                known_cases.add(cite_item["cite"].strip())
            elif isinstance(cite_item, str) and cite_item.strip():
                known_cases.add(cite_item.strip())
    return known_cases


def chunk_cap_volume(
    volume: str,
    raw_dir: str = CAP_RAW_DIR,
    chunks_dir: str = CAP_CHUNKS_DIR,
    max_workers: int = 10,
) -> Tuple[List[CorpusDocument], Set[str]]:
    """Download, parse, chunk, and cache a single CAP volume.
    
    If the volume chunks file exists in chunks_dir, loads directly from disk.
    Otherwise, reads or downloads raw cases, builds chunks, and caches them.
    """
    os.makedirs(chunks_dir, exist_ok=True)
    chunk_cache_file = os.path.join(chunks_dir, f"cap_chunks_us_{volume}.json")

    if os.path.exists(chunk_cache_file):
        try:
            with open(chunk_cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                raw_chunks = data.get("chunks", [])
                chunks = [CorpusDocument.from_dict(c) if isinstance(c, dict) else c for c in raw_chunks]
                known_cases = set(data.get("known_cases", []))
                if not known_cases:
                    known_cases = extract_known_cases_from_chunks(chunks)
                logger.debug("Loaded Volume %s chunks from cache (%d chunks).", volume, len(chunks))
                return chunks, known_cases
        except Exception as e:
            logger.warning("Chunk cache load failed for volume %s: %s. Re-chunking...", volume, e)

    cases = download_cap_volume(volume, raw_dir=raw_dir, max_workers=max_workers)
    if not cases:
        return [], set()

    documents: list[CorpusDocument] = []
    known_cases: Set[str] = set()

    for record in cases:
        text = _text_from_casebody(record.get("casebody"))
        if len(text) < 100:
            continue
        case_id = str(record.get("id", ""))
        name = record.get("name_abbreviation") or record.get("name") or ""
        court = (record.get("court") or {}).get("name", "") if isinstance(record.get("court"), dict) else ""
        date = str(record.get("decision_date") or "")
        try:
            year = int(date[:4])
        except ValueError:
            year = 0

        first_page_int = 0
        try:
            first_page_int = int(record.get("first_page") or 0)
        except (ValueError, TypeError):
            pass

        vol_val = str(record.get("volume", "") or "")
        if not vol_val and record.get("volume_number"):
            vol_val = str(record.get("volume_number"))

        citations = record.get("citations") or []
        digest = hashlib.sha256((case_id + text).encode("utf-8")).hexdigest()[:24]

        if name:
            known_cases.add(name.strip())
        if vol_val and first_page_int:
            known_cases.add(f"{vol_val} U.S. {first_page_int}".strip())
        for cite_item in citations:
            if isinstance(cite_item, dict) and cite_item.get("cite"):
                known_cases.add(cite_item["cite"].strip())

        documents.append(CorpusDocument(
            doc_id=f"CAP::{case_id}::{digest}",
            text=text,
            source="CAP",
            case_name=name,
            year=year,
            court=court,
            volume=vol_val,
            reporter="U.S.",
            first_page=first_page_int,
            metadata={
                "cap_id": case_id,
                "volume": vol_val,
                "decision_date": date,
                "docket": record.get("docket_number"),
                "citations": citations,
                "authority_identity": "CAP_CANONICAL"
            },
        ))

    chunks = chunk_corpus(documents)

    try:
        with open(chunk_cache_file, "w", encoding="utf-8") as handle:
            json.dump({
                "volume": volume,
                "chunks": [c.to_dict() for c in chunks],
                "known_cases": sorted(list(known_cases))
            }, handle, ensure_ascii=False)
    except Exception as e:
        logger.warning("Failed to write chunk cache for volume %s: %s", volume, e)

    return chunks, known_cases


def build_cap_corpus_in_portions(
    volumes: List[str],
    raw_dir: str = CAP_RAW_DIR,
    chunks_dir: str = CAP_CHUNKS_DIR,
    portion_size: int = 25,
    max_workers: int = 10,
    volume_tag: str = "multi",
) -> Tuple[List[CorpusDocument], Set[str]]:
    """Build CAP corpus by processing volumes in manageable portions with disk checkpointing.
    
    Portioning ensures progress is never lost if interrupted:
    - Chunks are cached per volume to cap_chunks_cache/
    - Checkpoints of the accumulated corpus and known cases are saved after each portion.
    """
    total_volumes = len(volumes)
    total_portions = (total_volumes + portion_size - 1) // portion_size
    all_chunks: List[CorpusDocument] = []
    all_known_cases: Set[str] = set()

    target_vols = set(str(v).strip() for v in volumes)
    is_subset = volume_tag.lower() not in ("all", "full", "multi") and len(target_vols) < 300
    subset_corpus_path = os.path.join(os.path.dirname(CORPUS_JSON_PATH), f"cap_corpus_{volume_tag}.json")

    # 1. Check if specific subset corpus already exists (e.g. cap_corpus_landmark.json)
    if is_subset and os.path.exists(subset_corpus_path):
        try:
            with open(subset_corpus_path, "r", encoding="utf-8") as f:
                raw_list = json.load(f)
            if isinstance(raw_list, list) and len(raw_list) > 0:
                print(f"  [CHECKPOINT] Found existing '{volume_tag}' corpus: {subset_corpus_path} ({len(raw_list):,} chunks).")
                all_chunks = [CorpusDocument.from_dict(d) if isinstance(d, dict) else d for d in raw_list]
                all_known_cases = extract_known_cases_from_chunks(all_chunks)
                print(f"  [CHECKPOINT] Extracted {len(all_known_cases):,} known cases & citations from '{volume_tag}' corpus.")
                return all_chunks, all_known_cases
        except Exception as e:
            logger.warning("Could not load existing %s: %s", subset_corpus_path, e)

    # 2. If canonical master corpus exists, try to filter it
    if os.path.exists(CORPUS_JSON_PATH):
        try:
            with open(CORPUS_JSON_PATH, "r", encoding="utf-8") as f:
                raw_list = json.load(f)
            if isinstance(raw_list, list) and len(raw_list) > 0:
                all_chunks = [CorpusDocument.from_dict(d) if isinstance(d, dict) else d for d in raw_list]
                
                # Filter to target volumes if running subset
                if is_subset:
                    filtered = [c for c in all_chunks if extract_doc_volume(c) in target_vols]
                    if filtered:
                        print(f"  [CHECKPOINT] Filtered existing master corpus to {len(filtered):,} chunks for '{volume_tag}' ({len(target_vols)} volumes).")
                        all_known_cases = extract_known_cases_from_chunks(filtered)
                        os.makedirs(os.path.dirname(subset_corpus_path), exist_ok=True)
                        with open(subset_corpus_path, "w", encoding="utf-8") as handle:
                            json.dump([chunk.to_dict() for chunk in filtered], handle, ensure_ascii=False)
                        return filtered, all_known_cases
                    else:
                        print(f"  [CHECKPOINT] Master corpus did not contain volume tags for '{volume_tag}'. Building directly from {total_volumes} volume caches...")
                else:
                    print(f"  [CHECKPOINT] Found existing master corpus: {CORPUS_JSON_PATH} ({len(raw_list):,} chunks).")
                    all_known_cases = extract_known_cases_from_chunks(all_chunks)
                    return all_chunks, all_known_cases
        except Exception as e:
            logger.warning("Could not load existing %s: %s. Building in portions...", CORPUS_JSON_PATH, e)

    all_chunks = []
    print(f"\n  Processing {total_volumes} volumes in {total_portions} portions (Portion size: {portion_size} volumes)...")

    for p_idx in range(total_portions):
        v_start = p_idx * portion_size
        v_end = min(v_start + portion_size, total_volumes)
        portion_vols = volumes[v_start:v_end]

        print(f"\n  [Portion {p_idx + 1}/{total_portions}] Processing volumes {portion_vols[0]} to {portion_vols[-1]} ({len(portion_vols)} volumes)...")

        portion_chunks: List[CorpusDocument] = []
        for vol in portion_vols:
            v_chunks, v_known = chunk_cap_volume(vol, raw_dir=raw_dir, chunks_dir=chunks_dir, max_workers=max_workers)
            portion_chunks.extend(v_chunks)
            all_known_cases.update(v_known)

        all_chunks.extend(portion_chunks)
        print(f"  [Portion {p_idx + 1}/{total_portions}] Complete: +{len(portion_chunks):,} chunks | Accumulated: {len(all_chunks):,} chunks | Known cases: {len(all_known_cases):,}")

        # Checkpoint corpus after each portion
        corpus_out_file = CORPUS_JSON_PATH if not is_subset else subset_corpus_path
        os.makedirs(os.path.dirname(corpus_out_file), exist_ok=True)
        with open(corpus_out_file, "w", encoding="utf-8") as handle:
            json.dump([chunk.to_dict() for chunk in all_chunks], handle, ensure_ascii=False)

        # Checkpoint known cases
        os.makedirs(os.path.dirname(KNOWN_CASES_PATH), exist_ok=True)
        with open(KNOWN_CASES_PATH, "wb") as handle:
            pickle.dump(all_known_cases, handle)

    write_cap_manifest(all_chunks, volume_tag=volume_tag)
    return all_chunks, all_known_cases


def build_cap_corpus(cases: Iterable[dict], volume_tag: str = "multi") -> Tuple[List[CorpusDocument], Set[str]]:
    """Convert raw CAP case bodies into retrieval chunks and extract known cases."""
    # If cases is empty but corpus JSON exists, load existing corpus directly
    if not cases and os.path.exists(CORPUS_JSON_PATH):
        try:
            with open(CORPUS_JSON_PATH, "r", encoding="utf-8") as f:
                raw_list = json.load(f)
            if isinstance(raw_list, list) and len(raw_list) > 0:
                chunks = [CorpusDocument.from_dict(d) if isinstance(d, dict) else d for d in raw_list]
                known_cases = extract_known_cases_from_chunks(chunks)
                logger.info("Loaded %d chunks from existing corpus %s", len(chunks), CORPUS_JSON_PATH)
                return chunks, known_cases
        except Exception as e:
            logger.warning("Failed to load existing %s: %s", CORPUS_JSON_PATH, e)

    documents: list[CorpusDocument] = []
    known_cases: Set[str] = set()

    for record in cases:
        text = _text_from_casebody(record.get("casebody"))
        if len(text) < 100:
            continue
        case_id = str(record.get("id", ""))
        name = record.get("name_abbreviation") or record.get("name") or ""
        court = (record.get("court") or {}).get("name", "") if isinstance(record.get("court"), dict) else ""
        date = str(record.get("decision_date") or "")
        try:
            year = int(date[:4])
        except ValueError:
            year = 0

        first_page_int = 0
        try:
            first_page_int = int(record.get("first_page") or 0)
        except (ValueError, TypeError):
            pass

        vol_val = str(record.get("volume", "") or "")
        if not vol_val and record.get("volume_number"):
            vol_val = str(record.get("volume_number"))

        citations = record.get("citations") or []
        digest = hashlib.sha256((case_id + text).encode("utf-8")).hexdigest()[:24]

        if name:
            known_cases.add(name.strip())
        if vol_val and first_page_int:
            known_cases.add(f"{vol_val} U.S. {first_page_int}".strip())

        for cite_item in citations:
            if isinstance(cite_item, dict) and cite_item.get("cite"):
                known_cases.add(cite_item["cite"].strip())

        documents.append(CorpusDocument(
            doc_id=f"CAP::{case_id}::{digest}",
            text=text,
            source="CAP",
            case_name=name,
            year=year,
            court=court,
            volume=vol_val,
            reporter="U.S.",
            first_page=first_page_int,
            metadata={
                "cap_id": case_id,
                "volume": vol_val,
                "decision_date": date,
                "docket": record.get("docket_number"),
                "citations": citations,
                "authority_identity": "CAP_CANONICAL"
            },
        ))

    chunks = chunk_corpus(documents)
    os.makedirs(os.path.dirname(CORPUS_JSON_PATH), exist_ok=True)
    with open(CORPUS_JSON_PATH, "w", encoding="utf-8") as handle:
        json.dump([chunk.to_dict() for chunk in chunks], handle, ensure_ascii=False)

    logger.info("CAP Corpus: %d cases -> %d chunks saved to %s", len(documents), len(chunks), CORPUS_JSON_PATH)
    return chunks, known_cases


def write_cap_manifest(chunks: list[CorpusDocument], volume_tag: str = "multi") -> None:
    corpus_hash = hashlib.sha256(open(CORPUS_JSON_PATH, "rb").read()).hexdigest() if os.path.exists(CORPUS_JSON_PATH) else ""
    manifest = {
        "source": "Caselaw Access Project (Harvard)",
        "license": "CC0-1.0",
        "volumes": volume_tag,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chunks": len(chunks),
        "corpus_sha256": corpus_hash,
        "is_primary": True,
    }
    os.makedirs(os.path.dirname(CORPUS_MANIFEST_PATH), exist_ok=True)
    with open(CORPUS_MANIFEST_PATH, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    logger.info("Manifest written to: %s", CORPUS_MANIFEST_PATH)
