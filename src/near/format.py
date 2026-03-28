"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import pandas as pd


def atomic_write_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(out_path)


def _first_present_column(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"missing required column; tried aliases: {list(candidates)}")


def _read_scan_candidates(in_csv: Union[str, Path]) -> pd.DataFrame:
    in_csv = Path(in_csv).expanduser().resolve()
    df = pd.read_csv(in_csv, dtype=str).fillna("")
    df.columns = [str(c).strip() for c in df.columns]

    canonical_cols: Dict[str, Sequence[str]] = {
        "dataset_a": ["dataset_a"],
        "subject_id_a": ["subject_id_a"],
        "session_id_a": ["session_id_a"],
        "scan_uid_a": ["scan_uid_a"],
        "src_path_a": ["src_path_a", "path_a", "resolved_path_a"],
        "dataset_b": ["dataset_b"],
        "subject_id_b": ["subject_id_b"],
        "session_id_b": ["session_id_b"],
        "scan_uid_b": ["scan_uid_b"],
        "src_path_b": ["src_path_b", "path_b", "resolved_path_b"],
        "similarity": ["similarity"],
    }

    rename_map: Dict[str, str] = {}
    for canon, aliases in canonical_cols.items():
        actual = _first_present_column(df, aliases)
        rename_map[actual] = canon

    df = df.rename(columns=rename_map).copy()

    required_cols = list(canonical_cols.keys())
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{in_csv} is missing required columns: {missing}")

    for c in required_cols:
        df[c] = df[c].fillna("").astype(str)

    df["similarity"] = pd.to_numeric(df["similarity"], errors="raise").astype(float)

    return df[
        [
            "dataset_a",
            "subject_id_a",
            "session_id_a",
            "scan_uid_a",
            "src_path_a",
            "dataset_b",
            "subject_id_b",
            "session_id_b",
            "scan_uid_b",
            "src_path_b",
            "similarity",
        ]
    ].copy()


def build_scan_level_review(
    scan_candidates_csv: Union[str, Path],
    out_csv: Union[str, Path],
    max_candidates_per_query: Optional[int] = None,
) -> Path:
    scan_candidates_csv = Path(scan_candidates_csv).expanduser().resolve()
    out_csv = Path(out_csv).expanduser().resolve()

    df = _read_scan_candidates(scan_candidates_csv)

    if df.empty:
        out_df = pd.DataFrame(
            columns=[
                "query_dataset",
                "query_subject_id",
                "query_session_id",
                "query_scan_uid",
                "query_src_path",
                "n_candidates",
            ]
        )
        atomic_write_csv(out_df, out_csv)
        print(f"[REVIEW FORMAT] scan-level rows: 0")
        print(f"[REVIEW FORMAT] saved -> {out_csv}")
        return out_csv

    rows_long: List[Dict[str, object]] = []

    for _, r in df.iterrows():
        rows_long.append(
            {
                "query_dataset": r["dataset_a"],
                "query_subject_id": r["subject_id_a"],
                "query_session_id": r["session_id_a"],
                "query_scan_uid": r["scan_uid_a"],
                "query_src_path": r["src_path_a"],
                "candidate_dataset": r["dataset_b"],
                "candidate_subject_id": r["subject_id_b"],
                "candidate_session_id": r["session_id_b"],
                "candidate_scan_uid": r["scan_uid_b"],
                "candidate_src_path": r["src_path_b"],
                "similarity": float(r["similarity"]),
            }
        )
        rows_long.append(
            {
                "query_dataset": r["dataset_b"],
                "query_subject_id": r["subject_id_b"],
                "query_session_id": r["session_id_b"],
                "query_scan_uid": r["scan_uid_b"],
                "query_src_path": r["src_path_b"],
                "candidate_dataset": r["dataset_a"],
                "candidate_subject_id": r["subject_id_a"],
                "candidate_session_id": r["session_id_a"],
                "candidate_scan_uid": r["scan_uid_a"],
                "candidate_src_path": r["src_path_a"],
                "similarity": float(r["similarity"]),
            }
        )

    long_df = pd.DataFrame(rows_long)
    long_df = long_df.sort_values(
        [
            "query_dataset",
            "query_subject_id",
            "query_session_id",
            "query_scan_uid",
            "similarity",
            "candidate_dataset",
            "candidate_subject_id",
            "candidate_session_id",
            "candidate_scan_uid",
        ],
        ascending=[True, True, True, True, False, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    grouped = list(
        long_df.groupby(
            [
                "query_dataset",
                "query_subject_id",
                "query_session_id",
                "query_scan_uid",
                "query_src_path",
            ],
            sort=False,
        )
    )

    if max_candidates_per_query is None:
        max_slots = max(len(g) for _, g in grouped)
    else:
        max_slots = max(0, int(max_candidates_per_query))

    rows_out: List[Dict[str, object]] = []

    for key, group_df in grouped:
        (
            query_dataset,
            query_subject_id,
            query_session_id,
            query_scan_uid,
            query_src_path,
        ) = key

        group_df = group_df.sort_values(
            [
                "similarity",
                "candidate_dataset",
                "candidate_subject_id",
                "candidate_session_id",
                "candidate_scan_uid",
            ],
            ascending=[False, True, True, True, True],
            kind="mergesort",
        ).reset_index(drop=True)

        if max_candidates_per_query is not None:
            group_df = group_df.head(int(max_candidates_per_query)).copy()

        row: Dict[str, object] = {
            "query_dataset": query_dataset,
            "query_subject_id": query_subject_id,
            "query_session_id": query_session_id,
            "query_scan_uid": query_scan_uid,
            "query_src_path": query_src_path,
            "n_candidates": int(len(group_df)),
        }

        for i in range(max_slots):
            j = i + 1
            if i < len(group_df):
                r = group_df.iloc[i]
                row[f"candidate_{j}_dataset"] = r["candidate_dataset"]
                row[f"candidate_{j}_subject_id"] = r["candidate_subject_id"]
                row[f"candidate_{j}_session_id"] = r["candidate_session_id"]
                row[f"candidate_{j}_scan_uid"] = r["candidate_scan_uid"]
                row[f"candidate_{j}_src_path"] = r["candidate_src_path"]
                row[f"candidate_{j}_similarity"] = float(r["similarity"])
            else:
                row[f"candidate_{j}_dataset"] = ""
                row[f"candidate_{j}_subject_id"] = ""
                row[f"candidate_{j}_session_id"] = ""
                row[f"candidate_{j}_scan_uid"] = ""
                row[f"candidate_{j}_src_path"] = ""
                row[f"candidate_{j}_similarity"] = ""

        rows_out.append(row)

    base_cols = [
        "query_dataset",
        "query_subject_id",
        "query_session_id",
        "query_scan_uid",
        "query_src_path",
        "n_candidates",
    ]

    candidate_cols: List[str] = []
    for j in range(1, max_slots + 1):
        candidate_cols.extend(
            [
                f"candidate_{j}_dataset",
                f"candidate_{j}_subject_id",
                f"candidate_{j}_session_id",
                f"candidate_{j}_scan_uid",
                f"candidate_{j}_src_path",
                f"candidate_{j}_similarity",
            ]
        )

    out_df = pd.DataFrame(rows_out, columns=base_cols + candidate_cols)
    out_df = out_df.sort_values(
        [
            "query_dataset",
            "query_subject_id",
            "query_session_id",
            "query_scan_uid",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    atomic_write_csv(out_df, out_csv)

    print(f"[REVIEW FORMAT] scan-level rows: {len(out_df):,}")
    print(f"[REVIEW FORMAT] saved -> {out_csv}")

    return out_csv


def build_review_table(
    review_dir: Union[str, Path],
    scan_candidates_csv: Union[str, Path],
    max_candidates_per_query: Optional[int] = None,
) -> Path:
    review_dir = Path(review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)

    review_csv = review_dir / "review_candidates.csv"

    build_scan_level_review(
        scan_candidates_csv=scan_candidates_csv,
        out_csv=review_csv,
        max_candidates_per_query=max_candidates_per_query,
    )

    print(f"[REVIEW FORMAT] outputs saved under -> {review_dir}")

    return review_csv
