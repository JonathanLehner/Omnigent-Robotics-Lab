"""Shared research record: one SQLite store, every handoff goes through it.

Objects are typed by `kind` and linked by id, so any decision can be traced back
to the results, runs, specs, hypotheses and evidence it rests on.
"""

import json
import os
import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("LAB_DB", ROOT / "record" / "lab.db"))

# kind -> (id prefix, required fields)
KINDS = {
    "evidence": ("E", ["source", "claim", "relevance"]),
    "scene": ("S", ["tier", "split", "spec_path", "image_path"]),
    "hypothesis": ("H", ["statement", "predicted_effect"]),
    "experiment": ("X", ["hypothesis_ids", "scene_ids", "method", "episodes", "expected_learning", "cost"]),
    "run": ("R", ["experiment_id", "method", "commit", "seeds", "results_path"]),
    "result": ("RS", ["run_ids", "metrics", "interpretation"]),
    "decision": ("D", ["result_ids", "change", "rationale"]),
}
HYPOTHESIS_STATUS = {"open", "supported", "refuted", "reopened"}
EXPERIMENT_STATUS = {"candidate", "selected", "rejected", "done"}


def _db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute(
        "CREATE TABLE IF NOT EXISTS objects (id TEXT PRIMARY KEY, kind TEXT, author TEXT,"
        " created REAL, updated REAL, data TEXT)"
    )
    return db


def _next_id(db, kind):
    prefix = KINDS[kind][0]
    n = db.execute("SELECT COUNT(*) FROM objects WHERE kind=?", (kind,)).fetchone()[0]
    return f"{prefix}-{n + 1:03d}"


def add(kind: str, author: str, data: dict, obj_id: str | None = None) -> str:
    """Validate and insert an object; returns its id."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; one of {sorted(KINDS)}")
    missing = [f for f in KINDS[kind][1] if f not in data]
    if missing:
        raise ValueError(f"{kind} missing fields: {missing}")
    if not author:
        raise ValueError("author (the agent writing this) is required")
    with _db() as db:
        _check_links(db, kind, data)
        obj_id = obj_id or _next_id(db, kind)
        now = time.time()
        db.execute("INSERT INTO objects VALUES (?,?,?,?,?,?)", (obj_id, kind, author, now, now, json.dumps(data)))
    return obj_id


def _check_links(db, kind, data):
    """Every *_ids / *_id field must point at existing objects: no dangling citations."""
    refs = []
    for key, val in data.items():
        if key.endswith("_ids") and isinstance(val, list):
            refs += val
        elif key.endswith("_id") and isinstance(val, str):
            refs.append(val)
    for ref in refs:
        if not db.execute("SELECT 1 FROM objects WHERE id=?", (ref,)).fetchone():
            raise ValueError(f"{kind} links to unknown id {ref!r}")
    if kind == "decision" and not data["result_ids"]:
        raise ValueError("a decision must cite at least one result id")


def get(obj_id: str) -> dict | None:
    with _db() as db:
        row = db.execute("SELECT id, kind, author, created, updated, data FROM objects WHERE id=?", (obj_id,)).fetchone()
    return _row(row) if row else None


def update(obj_id: str, author: str, fields: dict) -> dict:
    """Merge fields into an object (status changes, reviewer verdicts). History is kept in `history`."""
    obj = get(obj_id)
    if obj is None:
        raise ValueError(f"unknown id {obj_id!r}")
    data = obj["data"]
    if "status" in fields:
        allowed = HYPOTHESIS_STATUS if obj["kind"] == "hypothesis" else EXPERIMENT_STATUS
        if fields["status"] not in allowed:
            raise ValueError(f"status must be one of {sorted(allowed)}")
    data.setdefault("history", []).append({"t": time.time(), "by": author, "set": fields})
    data.update(fields)
    with _db() as db:
        _check_links(db, obj["kind"], fields)
        db.execute("UPDATE objects SET data=?, updated=? WHERE id=?", (json.dumps(data), time.time(), obj_id))
    return get(obj_id)


def query(kind: str | None = None, contains: str | None = None, limit: int = 50) -> list[dict]:
    sql, args = "SELECT id, kind, author, created, updated, data FROM objects WHERE 1=1", []
    if kind:
        sql, args = sql + " AND kind=?", args + [kind]
    if contains:
        sql, args = sql + " AND data LIKE ?", args + [f"%{contains}%"]
    with _db() as db:
        rows = db.execute(sql + " ORDER BY created DESC LIMIT ?", args + [limit]).fetchall()
    return [_row(r) for r in rows]


def _row(r):
    return {"id": r[0], "kind": r[1], "author": r[2], "created": r[3], "updated": r[4], "data": json.loads(r[5])}


def export(path: Path = ROOT / "record" / "export.json") -> str:
    """Dump the full record (oldest first) for judges and for reconstruction."""
    objs = list(reversed(query(limit=100000)))
    path.write_text(json.dumps(objs, indent=1))
    return str(path)


if __name__ == "__main__":
    import tempfile

    DB_PATH = Path(tempfile.mkdtemp()) / "t.db"
    e = add("evidence", "scout", {"source": "arXiv:2604.08508", "claim": "c", "relevance": "r"})
    h = add("hypothesis", "method_designer", {"statement": "s", "predicted_effect": "p", "evidence_ids": [e], "status": "open"})
    try:
        add("decision", "pi", {"result_ids": ["RS-999"], "change": "x", "rationale": "y"})
        raise AssertionError("dangling link accepted")
    except ValueError:
        pass
    assert update(h, "pi", {"status": "reopened"})["data"]["status"] == "reopened"
    assert [o["id"] for o in query("hypothesis")] == ["H-001"]
    print("record ok")
