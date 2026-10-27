"""Compare isolated unified-cache databases without writing either database.

Only volatile timestamp columns are excluded. BLOB payloads, file revisions,
cache signatures and processing results are compared byte-for-byte.
"""
from __future__ import annotations

import argparse
import hashlib
import sqlite3
from pathlib import Path


def database_contents(path: Path) -> dict[str, tuple[int, str]]:
    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        tables = [row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        result = {}
        for table in tables:
            columns = [row[1] for row in db.execute(f'PRAGMA table_info("{table}")')]
            stable = [name for name in columns if name not in {"updated_at", "created_at"}]
            query = f'SELECT {", ".join(chr(34) + name + chr(34) for name in stable)} FROM "{table}" ORDER BY '
            query += ", ".join(str(index) for index in range(1, len(stable) + 1))
            digest = hashlib.sha256()
            count = 0
            for row in db.execute(query):
                for value in row:
                    if isinstance(value, bytes):
                        digest.update(b"B" + len(value).to_bytes(8, "little") + value)
                    else:
                        encoded = repr(value).encode("utf-8")
                        digest.update(b"V" + len(encoded).to_bytes(8, "little") + encoded)
                count += 1
            result[table] = count, digest.hexdigest()
        return result
    finally:
        db.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args(argv)
    baseline = database_contents(args.baseline.resolve())
    candidate = database_contents(args.candidate.resolve())
    for table in sorted(baseline.keys() | candidate.keys()):
        left, right = baseline.get(table), candidate.get(table)
        print(f"{table}: {'MATCH' if left == right else 'DIFFER'}; rows={left[0] if left else None}/{right[0] if right else None}")
    return 0 if baseline == candidate else 1


if __name__ == "__main__":
    raise SystemExit(main())
