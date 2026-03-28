"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

from __future__ import annotations

import hashlib
import pandas as pd

from pathlib import Path
from typing import Optional, Tuple, Union


def _read_csv_or_empty(
    path: Optional[Union[str, Path]],
    required_cols: Tuple[str, ...],
) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=list(required_cols))

    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=list(required_cols))

    df = pd.read_csv(p, dtype=str).fillna("")
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{p} is missing required columns: {missing}")
    return df


def _choose_exact_representatives(
    exact_duplicates_df: pd.DataFrame,
    sort_keys: Tuple[str, ...] = (
        "group_id",
        "dataset",
        "subject_id",
        "session_id",
        "candidate",
        "resolved_path",
    ),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = (
        "group_id",
        "dataset",
        "subject_id",
        "session_id",
        "candidate",
        "resolved_path",
    )
    missing = [c for c in required if c not in exact_duplicates_df.columns]
    if missing:
        raise ValueError(f"exact_duplicates_df is missing required columns: {missing}")

    if exact_duplicates_df.empty:
        reps = pd.DataFrame(columns=list(required))
        removed = pd.DataFrame(
            columns=[
                *required,
                "kept_candidate",
                "kept_resolved_path",
                "remove_reason",
            ]
        )
        return reps, removed

    df = exact_duplicates_df.copy()
    for c in required:
        df[c] = df[c].fillna("").astype(str)

    df = df.sort_values(list(sort_keys), kind="mergesort").reset_index(drop=True)

    reps = (
        df.groupby("group_id", sort=False, as_index=False).nth(0).reset_index(drop=True)
    )

    reps_meta = reps[
        [
            "group_id",
            "candidate",
            "resolved_path",
        ]
    ].rename(
        columns={
            "candidate": "kept_candidate",
            "resolved_path": "kept_resolved_path",
        }
    )

    removed = df.merge(reps_meta, on="group_id", how="left")
    removed = removed[removed["resolved_path"] != removed["kept_resolved_path"]].copy()
    removed["remove_reason"] = "EXACT_DUPLICATE_REMOVED"

    return reps, removed


def make_scan_uid(resolved_path: str) -> str:
    s = str(resolved_path).strip()
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def prepare_no_exact_dup(
    valid_csv: Union[str, Path],
    exact_duplicates_csv: Optional[Union[str, Path]],
    out_keep_csv: Union[str, Path],
    out_removed_csv: Union[str, Path],
    hash_errors_csv: Optional[Union[str, Path]] = None,
    final_sort_keys: Tuple[str, ...] = (
        "dataset",
        "subject_id",
        "session_id",
        "candidate",
        "resolved_path",
    ),
) -> tuple[Path, Path]:
    valid_csv = Path(valid_csv)
    out_keep_csv = Path(out_keep_csv)
    out_removed_csv = Path(out_removed_csv)

    valid_required = (
        "dataset",
        "subject_id",
        "session_id",
        "candidate",
        "resolved_path",
    )
    exact_duplicates_required = (
        "group_id",
        "dataset",
        "subject_id",
        "session_id",
        "candidate",
        "resolved_path",
    )
    hash_error_required = (
        "resolved_path",
        "error",
    )

    valid_df = _read_csv_or_empty(valid_csv, valid_required)
    if valid_df.empty:
        raise ValueError(f"{valid_csv} is empty or missing required rows.")

    exact_duplicates_df = _read_csv_or_empty(
        exact_duplicates_csv, exact_duplicates_required
    )
    hash_errors_df = _read_csv_or_empty(hash_errors_csv, hash_error_required)

    reps_df, exact_dup_removed_df = _choose_exact_representatives(exact_duplicates_df)

    exact_dup_removed_paths = set(
        exact_dup_removed_df["resolved_path"].dropna().astype(str).str.strip().tolist()
    )
    hash_failed_paths = set(
        hash_errors_df["resolved_path"].dropna().astype(str).str.strip().tolist()
    )

    keep_mask = (
        ~valid_df["resolved_path"]
        .astype(str)
        .isin(exact_dup_removed_paths | hash_failed_paths)
    )
    keep_df = valid_df[keep_mask].copy()
    removed_base_df = valid_df[~keep_mask].copy()

    exact_dup_removed_out = removed_base_df[
        removed_base_df["resolved_path"].astype(str).isin(exact_dup_removed_paths)
    ].copy()
    if not exact_dup_removed_out.empty:
        exact_dup_removed_out = exact_dup_removed_out.merge(
            exact_dup_removed_df[
                [
                    "resolved_path",
                    "kept_candidate",
                    "kept_resolved_path",
                    "remove_reason",
                ]
            ].drop_duplicates(subset=["resolved_path"]),
            on="resolved_path",
            how="left",
        )

    hash_removed_out = removed_base_df[
        removed_base_df["resolved_path"].astype(str).isin(hash_failed_paths)
    ].copy()
    if not hash_removed_out.empty:
        hash_removed_out = hash_removed_out.merge(
            hash_errors_df[["resolved_path", "error"]].drop_duplicates(
                subset=["resolved_path"]
            ),
            on="resolved_path",
            how="left",
        )
        hash_removed_out["remove_reason"] = "HASH_FAILED"

    removed_df = pd.concat(
        [exact_dup_removed_out, hash_removed_out],
        ignore_index=True,
        sort=False,
    )

    if not removed_df.empty:
        removed_df = removed_df.drop_duplicates(subset=["resolved_path"], keep="first")
        removed_df = removed_df.sort_values(
            ["remove_reason", "dataset", "subject_id", "session_id", "candidate"],
            kind="mergesort",
        ).reset_index(drop=True)

    keep_df = keep_df.sort_values(list(final_sort_keys), kind="mergesort").reset_index(
        drop=True
    )
    keep_df["scan_uid"] = keep_df["resolved_path"].map(make_scan_uid)

    keep_cols = [
        "dataset",
        "subject_id",
        "session_id",
        "candidate",
        "resolved_path",
        "scan_uid",
    ]
    keep_existing = [c for c in keep_cols if c in keep_df.columns]
    keep_other = [c for c in keep_df.columns if c not in keep_existing]
    keep_df = keep_df[keep_existing + keep_other]

    removed_cols_preferred = [
        "remove_reason",
        "dataset",
        "subject_id",
        "session_id",
        "candidate",
        "resolved_path",
        "kept_candidate",
        "kept_resolved_path",
        "error",
    ]
    if not removed_df.empty:
        removed_existing = [
            c for c in removed_cols_preferred if c in removed_df.columns
        ]
        removed_other = [c for c in removed_df.columns if c not in removed_existing]
        removed_df = removed_df[removed_existing + removed_other]
    else:
        removed_df = pd.DataFrame(columns=removed_cols_preferred)

    out_keep_csv.parent.mkdir(parents=True, exist_ok=True)
    out_removed_csv.parent.mkdir(parents=True, exist_ok=True)

    keep_df.to_csv(out_keep_csv, index=False)
    removed_df.to_csv(out_removed_csv, index=False)

    n_valid = len(valid_df)
    n_exact_dup_removed = len(exact_dup_removed_out)
    n_hash_removed = len(hash_removed_out)
    n_keep = len(keep_df)

    print(f"[NEAR PREP] valid input rows: {n_valid:,}")
    print(f"[NEAR PREP] exact duplicate rows removed: {n_exact_dup_removed:,}")
    print(f"[NEAR PREP] hash-failed rows removed: {n_hash_removed:,}")
    print(f"[NEAR PREP] rows kept for near stage: {n_keep:,}")
    print(f"[NEAR PREP] saved keep table -> {out_keep_csv}")
    print(f"[NEAR PREP] saved removed table -> {out_removed_csv}")

    if not reps_df.empty:
        print(
            f"[NEAR PREP] exact duplicate groups seen: {reps_df['group_id'].nunique():,}"
        )

    return out_keep_csv, out_removed_csv
