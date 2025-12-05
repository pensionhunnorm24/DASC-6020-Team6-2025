"""
data_loader.py
Robust dataset preparation that handles:
- flat records with canonical keys
- nested records with 'input' and 'expected_output'
- common synonyms and explicit mapping via --map
Saves train.csv (750) and test.csv (250) to data/
"""

import json
import argparse
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1].joinpath("data")

# canonical names expected by the rest of the pipeline
CANONICAL = [
    "trip_duration_days",
    "miles_traveled",
    "total_receipts_amount",
    "reimbursement_amount"
]

# common synonyms to try mapping automatically
SYNONYMS = {
    "trip_duration_days": {"trip_duration_days", "trip_days", "days", "duration", "trip_duration"},
    "miles_traveled": {"miles_traveled", "miles", "distance", "miles_driven"},
    "total_receipts_amount": {"total_receipts_amount", "total_receipts", "receipts", "amount", "total_amount"},
    "reimbursement_amount": {"reimbursement_amount", "reimbursed", "reimbursement", "paid_amount", "amount_paid", "expected_output"}
}

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def extract_flat_or_nested(record):
    """
    Return a flat dict mapping keys to values.
    If record has 'input' and 'expected_output', merge them:
      - fields from record['input'] become top-level
      - record['expected_output'] is mapped to reimbursement_amount if numeric
    """
    if isinstance(record, dict):
        # nested case: input + expected_output
        if 'input' in record and 'expected_output' in record:
            flat = {}
            inp = record.get('input') or {}
            if isinstance(inp, dict):
                flat.update(inp)
            # expected_output may be a number or dict
            eo = record.get('expected_output')
            if isinstance(eo, dict):
                # if expected_output is dict, try to find reimbursement-like key
                # prefer keys that match synonyms
                for k, v in eo.items():
                    flat[k] = v
            else:
                # numeric expected_output -> map to reimbursement_amount
                flat['reimbursement_amount'] = eo
            return flat
        # otherwise assume already flat
        return record.copy()
    # not a dict -> return empty
    return {}

def infer_mapping(sample_record: dict):
    """
    Given a sample flat record (dict), attempt to map its keys to the canonical names.
    Returns a dict mapping canonical -> actual_key or None if mapping incomplete.
    """
    actual_keys = set(sample_record.keys())
    mapping = {}
    for canon in CANONICAL:
        # first check exact match
        if canon in actual_keys:
            mapping[canon] = canon
            continue
        # try synonyms
        found = None
        for syn in SYNONYMS.get(canon, set()):
            if syn in actual_keys:
                found = syn
                break
        if found:
            mapping[canon] = found
            continue
        # try case-insensitive match
        for k in actual_keys:
            if k.lower() == canon.lower():
                mapping[canon] = k
                found = k
                break
        if canon not in mapping:
            mapping[canon] = None
    return mapping

def prepare_datasets(json_path: str | Path, seed: int = 42, explicit_map: dict | None = None):
    """
    Read JSON, map keys to canonical names, create DataFrame, shuffle, split into train/test,
    and save CSV files to data/ directory.
    explicit_map: optional dict mapping canonical_name -> actual_key (overrides auto-detection)
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    records = load_json(Path(json_path))

    if not isinstance(records, list) or len(records) == 0:
        raise ValueError("Input JSON must be a non-empty list of records (list of dicts).")

    # convert first record to flat form to inspect keys
    sample_flat = extract_flat_or_nested(records[0])
    mapping = infer_mapping(sample_flat)

    # apply explicit_map overrides if provided
    if explicit_map:
        for k, v in explicit_map.items():
            if k in CANONICAL:
                mapping[k] = v

    # check mapping completeness
    missing = [k for k, v in mapping.items() if v is None]
    if missing:
        available = list(sample_flat.keys())
        raise ValueError(
            "Could not find required fields in JSON. Missing mappings for: "
            f"{missing}. Available keys in sample record: {available}. "
            "You can provide explicit mappings via --map canonical:actual (e.g. --map trip_duration_days:days)."
        )

    # build list of flat records using extract_flat_or_nested
    flat_records = []
    for rec in records:
        flat = extract_flat_or_nested(rec)
        # rename keys to canonical using mapping
        renamed = {}
        for canon in CANONICAL:
            actual_key = mapping[canon]
            # if actual_key not present in this flat record, set NaN
            renamed[canon] = flat.get(actual_key, None)
        flat_records.append(renamed)

    df = pd.DataFrame(flat_records)

    # coerce types and fill missing sensibly
    df['trip_duration_days'] = pd.to_numeric(df['trip_duration_days'], errors='coerce').fillna(0).astype(int)
    df['miles_traveled'] = pd.to_numeric(df['miles_traveled'], errors='coerce').fillna(0).astype(int)
    df['total_receipts_amount'] = pd.to_numeric(df['total_receipts_amount'], errors='coerce').fillna(0.0).astype(float)
    df['reimbursement_amount'] = pd.to_numeric(df['reimbursement_amount'], errors='coerce').fillna(0.0).astype(float)

    # shuffle and split
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    train = df.iloc[:750].copy()
    test = df.iloc[750:].copy()

    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    print(f"Saved train.csv ({len(train)}) and test.csv ({len(test)}) to {DATA_DIR}")
    return train_path, test_path

if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser(description="Prepare train/test CSVs from public_cases.json")
    parser.add_argument("json_path", help="Path to public_cases.json")
    parser.add_argument("--map", action="append",
                        help="Explicit mapping in form canonical:actual_key. Can be repeated. Example: --map trip_duration_days:days")
    args = parser.parse_args()

    explicit = {}
    if args.map:
        for m in args.map:
            if ":" not in m:
                print("Invalid --map format. Use canonical:actual_key")
                sys.exit(1)
            canon, actual = m.split(":", 1)
            explicit[canon] = actual

    prepare_datasets(args.json_path, explicit_map=explicit)