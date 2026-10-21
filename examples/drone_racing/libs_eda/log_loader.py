import json
import numpy as np


def load_jsonl(file_path, skip_first_line=True):
    """Load flat JSONL records into a dictionary of 1D float arrays.

    skip_first_line=True preserves the behavior of the original project loader.
    Set it to False if every non-empty JSON line is a data record.
    """
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)

    if skip_first_line and rows:
        rows = rows[1:]
    if not rows:
        return {}

    keys = list(dict.fromkeys(key for row in rows for key in row))
    result = {}
    for key in keys:
        values = []
        for row in rows:
            try:
                values.append(float(row.get(key, np.nan)))
            except (TypeError, ValueError):
                values.append(np.nan)
        result[key] = np.asarray(values, dtype=np.float32)

    print("loading log :", file_path)
    print("records     :", len(rows))
    print("keys        :", keys)
    print()
    return result
