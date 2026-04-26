"""
Stage 8: Storage Layer
- Primary: MongoDB (stores transcript, summary, metadata)
- Fallback: Local JSON file storage
- Supports search/retrieval of past sessions
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any


LOCAL_STORAGE_PATH = Path(__file__).parent.parent / "tmp" / "sessions.json"


def save_to_db(data: Dict[str, Any]) -> bool:
    """
    Save session data to MongoDB (or local JSON fallback).

    Args:
        data: Dict containing 'filename', 'transcript', 'summary', 'segments'

    Returns:
        True on success, False on failure
    """
    record = {
        "filename": data.get("filename", "unknown"),
        "transcript": data.get("transcript", ""),
        "summary": data.get("summary", {}),
        "segments": data.get("segments", []),
        "created_at": datetime.utcnow().isoformat(),
        "word_count": len(data.get("transcript", "").split()),
    }

    # Try MongoDB first
    if _save_to_mongo(record):
        return True

    # Fallback: local JSON
    return _save_to_local_json(record)


def search_past_sessions(query: str = "", limit: int = 20) -> List[Dict[str, Any]]:
    """
    Retrieve past sessions from MongoDB or local storage.

    Args:
        query: Optional search string (searches filename and transcript)
        limit: Max number of results

    Returns:
        List of session dicts
    """
    # Try MongoDB
    results = _search_mongo(query, limit)
    if results is not None:
        return results

    # Fallback: local JSON
    return _search_local_json(query, limit)


# ─── MongoDB Implementation ────────────────────────────────────────────────────

def _save_to_mongo(record: Dict) -> bool:
    """Save record to MongoDB."""
    mongo_uri = os.getenv("MONGO_URI", "")
    if not mongo_uri:
        return False

    try:
        from pymongo import MongoClient
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        db = client["audiomind"]
        col = db["sessions"]

        # Don't store full segments list if too large (>100 segments)
        if len(record.get("segments", [])) > 100:
            record["segments"] = record["segments"][:100]

        result = col.insert_one(record)
        print(f"[Storage] Saved to MongoDB: {result.inserted_id}")
        return True

    except Exception as e:
        print(f"[Storage] MongoDB save failed: {e}")
        return False


def _search_mongo(query: str, limit: int) -> Optional[List[Dict]]:
    """Search sessions in MongoDB."""
    mongo_uri = os.getenv("MONGO_URI", "")
    if not mongo_uri:
        return None

    try:
        from pymongo import MongoClient
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        db = client["audiomind"]
        col = db["sessions"]

        if query:
            filter_q = {
                "$or": [
                    {"filename": {"$regex": query, "$options": "i"}},
                    {"transcript": {"$regex": query, "$options": "i"}},
                ]
            }
        else:
            filter_q = {}

        projection = {"transcript": 0, "segments": 0}  # exclude large fields
        cursor = col.find(filter_q, projection).sort("created_at", -1).limit(limit)
        results = []
        for doc in cursor:
            doc["_id"] = str(doc["_id"])
            results.append(doc)
        return results

    except Exception as e:
        print(f"[Storage] MongoDB search failed: {e}")
        return None


# ─── Local JSON Fallback ───────────────────────────────────────────────────────

def _save_to_local_json(record: Dict) -> bool:
    """Save record to local JSON file."""
    try:
        LOCAL_STORAGE_PATH.parent.mkdir(parents=True, exist_ok=True)

        sessions = _load_local_sessions()

        # Keep only lightweight info in local storage
        lightweight = {
            "filename": record["filename"],
            "summary": record.get("summary", {}),
            "created_at": record["created_at"],
            "word_count": record.get("word_count", 0),
            "transcript_preview": record.get("transcript", "")[:300],
        }
        sessions.append(lightweight)

        # Keep last 50 sessions
        sessions = sessions[-50:]

        with open(LOCAL_STORAGE_PATH, "w", encoding="utf-8") as f:
            json.dump(sessions, f, indent=2, ensure_ascii=False)

        print(f"[Storage] Saved locally: {LOCAL_STORAGE_PATH}")
        return True

    except Exception as e:
        print(f"[Storage] Local save failed: {e}")
        return False


def _search_local_json(query: str, limit: int) -> List[Dict]:
    """Search local JSON sessions."""
    sessions = _load_local_sessions()
    if query:
        q = query.lower()
        sessions = [s for s in sessions if
                    q in s.get("filename", "").lower() or
                    q in s.get("transcript_preview", "").lower()]
    return list(reversed(sessions))[:limit]


def _load_local_sessions() -> List[Dict]:
    """Load all local sessions from JSON file."""
    if not LOCAL_STORAGE_PATH.exists():
        return []
    try:
        with open(LOCAL_STORAGE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []
