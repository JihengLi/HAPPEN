"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
import numpy as np


def atomic_write_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(out_path)


def _subject_key(dataset: str, subject_id: str) -> str:
    return f"{str(dataset).strip()}|{str(subject_id).strip()}"


def _read_raw_neighbors(
    in_csv: Union[str, Path],
) -> pd.DataFrame:
    in_csv = Path(in_csv)

    required_cols = [
        "query_dataset",
        "query_subject_id",
        "query_session_id",
        "query_candidate",
        "query_resolved_path",
        "query_scan_uid",
        "cand_dataset",
        "cand_subject_id",
        "cand_session_id",
        "cand_candidate",
        "cand_resolved_path",
        "cand_scan_uid",
        "rank",
        "similarity",
    ]

    df = pd.read_csv(in_csv, dtype=str).fillna("")
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{in_csv} is missing required columns: {missing}")

    for c in required_cols:
        df[c] = df[c].fillna("").astype(str)

    df["rank"] = pd.to_numeric(df["rank"], errors="raise").astype(int)
    df["similarity"] = pd.to_numeric(df["similarity"], errors="raise").astype(float)

    return df


def build_scan_candidates(
    raw_neighbors_csv: Union[str, Path],
    out_csv: Union[str, Path],
    min_similarity: float,
) -> Path:
    raw_neighbors_csv = Path(raw_neighbors_csv)
    out_csv = Path(out_csv)

    df = _read_raw_neighbors(raw_neighbors_csv)
    if df.empty:
        out_df = pd.DataFrame(
            columns=[
                "dataset_a",
                "subject_id_a",
                "session_id_a",
                "candidate_a",
                "resolved_path_a",
                "scan_uid_a",
                "dataset_b",
                "subject_id_b",
                "session_id_b",
                "candidate_b",
                "resolved_path_b",
                "scan_uid_b",
                "similarity",
                "best_rank",
                "n_directions",
            ]
        )
        atomic_write_csv(out_df, out_csv)
        return out_csv

    df = df[df["similarity"] >= float(min_similarity)].copy()

    if df.empty:
        out_df = pd.DataFrame(
            columns=[
                "dataset_a",
                "subject_id_a",
                "session_id_a",
                "candidate_a",
                "resolved_path_a",
                "scan_uid_a",
                "dataset_b",
                "subject_id_b",
                "session_id_b",
                "candidate_b",
                "resolved_path_b",
                "scan_uid_b",
                "similarity",
                "best_rank",
                "n_directions",
            ]
        )
        atomic_write_csv(out_df, out_csv)
        return out_csv

    df["query_subject_key"] = df.apply(
        lambda r: _subject_key(r["query_dataset"], r["query_subject_id"]), axis=1
    )
    df["cand_subject_key"] = df.apply(
        lambda r: _subject_key(r["cand_dataset"], r["cand_subject_id"]), axis=1
    )

    # only keep cross-ID pairs
    df = df[df["query_subject_key"] != df["cand_subject_key"]].copy()

    if df.empty:
        out_df = pd.DataFrame(
            columns=[
                "dataset_a",
                "subject_id_a",
                "session_id_a",
                "candidate_a",
                "resolved_path_a",
                "scan_uid_a",
                "dataset_b",
                "subject_id_b",
                "session_id_b",
                "candidate_b",
                "resolved_path_b",
                "scan_uid_b",
                "similarity",
                "best_rank",
                "n_directions",
            ]
        )
        atomic_write_csv(out_df, out_csv)
        return out_csv

    rows: List[Dict[str, object]] = []

    for _, r in df.iterrows():
        qa = str(r["query_scan_uid"])
        qb = str(r["cand_scan_uid"])

        if qa == qb:
            continue

        if qa < qb:
            row = {
                "dataset_a": r["query_dataset"],
                "subject_id_a": r["query_subject_id"],
                "session_id_a": r["query_session_id"],
                "candidate_a": r["query_candidate"],
                "resolved_path_a": r["query_resolved_path"],
                "scan_uid_a": r["query_scan_uid"],
                "dataset_b": r["cand_dataset"],
                "subject_id_b": r["cand_subject_id"],
                "session_id_b": r["cand_session_id"],
                "candidate_b": r["cand_candidate"],
                "resolved_path_b": r["cand_resolved_path"],
                "scan_uid_b": r["cand_scan_uid"],
                "similarity": float(r["similarity"]),
                "best_rank": int(r["rank"]),
                "n_directions": 1,
            }
        else:
            row = {
                "dataset_a": r["cand_dataset"],
                "subject_id_a": r["cand_subject_id"],
                "session_id_a": r["cand_session_id"],
                "candidate_a": r["cand_candidate"],
                "resolved_path_a": r["cand_resolved_path"],
                "scan_uid_a": r["cand_scan_uid"],
                "dataset_b": r["query_dataset"],
                "subject_id_b": r["query_subject_id"],
                "session_id_b": r["query_session_id"],
                "candidate_b": r["query_candidate"],
                "resolved_path_b": r["query_resolved_path"],
                "scan_uid_b": r["query_scan_uid"],
                "similarity": float(r["similarity"]),
                "best_rank": int(r["rank"]),
                "n_directions": 1,
            }

        rows.append(row)

    pair_df = pd.DataFrame(
        rows,
        columns=[
            "dataset_a",
            "subject_id_a",
            "session_id_a",
            "candidate_a",
            "resolved_path_a",
            "scan_uid_a",
            "dataset_b",
            "subject_id_b",
            "session_id_b",
            "candidate_b",
            "resolved_path_b",
            "scan_uid_b",
            "similarity",
            "best_rank",
            "n_directions",
        ],
    )

    if pair_df.empty:
        atomic_write_csv(pair_df, out_csv)
        return out_csv

    pair_df = pair_df.sort_values(
        ["scan_uid_a", "scan_uid_b", "similarity", "best_rank"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    agg_rows: List[Dict[str, object]] = []
    for (a_uid, b_uid), g in pair_df.groupby(["scan_uid_a", "scan_uid_b"], sort=False):
        g = g.sort_values(
            ["similarity", "best_rank"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)

        best = g.iloc[0].to_dict()
        best["n_directions"] = int(len(g))
        best["best_rank"] = int(g["best_rank"].min())
        best["similarity"] = float(g["similarity"].max())
        agg_rows.append(best)

    out_df = pd.DataFrame(
        agg_rows,
        columns=[
            "dataset_a",
            "subject_id_a",
            "session_id_a",
            "candidate_a",
            "resolved_path_a",
            "scan_uid_a",
            "dataset_b",
            "subject_id_b",
            "session_id_b",
            "candidate_b",
            "resolved_path_b",
            "scan_uid_b",
            "similarity",
            "best_rank",
            "n_directions",
        ],
    )

    out_df = out_df.sort_values(
        ["scan_uid_a", "scan_uid_b", "similarity"],
        ascending=[True, True, False],
        kind="mergesort",
    ).reset_index(drop=True)

    atomic_write_csv(out_df, out_csv)

    print(f"[NEAR GROUP] raw neighbor rows: {len(df):,}")
    print(f"[NEAR GROUP] scan candidate rows: {len(out_df):,}")
    print(f"[NEAR GROUP] saved -> {out_csv}")

    return out_csv


class UnionFind:
    def __init__(self):
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}

    def add(self, x: str) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: str) -> str:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return

        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def _read_review_decisions(
    in_csv: Union[str, Path],
) -> pd.DataFrame:
    in_csv = Path(in_csv)

    required_cols = [
        "dataset_1",
        "subject_id_1",
        "session_id_1",
        "path_1",
        "dataset_2",
        "subject_id_2",
        "session_id_2",
        "path_2",
        "QA_status",
        "reason",
        "date",
    ]

    df = pd.read_csv(in_csv, dtype=str).fillna("")
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{in_csv} is missing required columns: {missing}")

    for c in required_cols:
        df[c] = df[c].fillna("").astype(str).str.strip()

    return df


def build_subject_edges(
    review_decisions_csv: Union[str, Path],
    out_csv: Union[str, Path],
) -> Path:
    review_decisions_csv = Path(review_decisions_csv)
    out_csv = Path(out_csv)

    df = _read_review_decisions(review_decisions_csv)

    out_cols = [
        "dataset_a",
        "subject_id_a",
        "subject_key_a",
        "dataset_b",
        "subject_id_b",
        "subject_key_b",
        "n_yes_scan_pairs",
    ]

    if df.empty:
        atomic_write_csv(pd.DataFrame(columns=out_cols), out_csv)
        return out_csv

    df = df[df["QA_status"].str.lower() == "yes"].copy()

    if df.empty:
        atomic_write_csv(pd.DataFrame(columns=out_cols), out_csv)
        print("[NEAR GROUP] no confirmed ('yes') review decisions found")
        print(f"[NEAR GROUP] saved -> {out_csv}")
        return out_csv

    df = df[
        (df["dataset_1"] != "")
        & (df["subject_id_1"] != "")
        & (df["dataset_2"] != "")
        & (df["subject_id_2"] != "")
    ].copy()

    if df.empty:
        atomic_write_csv(pd.DataFrame(columns=out_cols), out_csv)
        print("[NEAR GROUP] no valid confirmed subject pairs after filtering empty IDs")
        print(f"[NEAR GROUP] saved -> {out_csv}")
        return out_csv

    df["subject_key_1"] = df.apply(
        lambda r: _subject_key(r["dataset_1"], r["subject_id_1"]),
        axis=1,
    )
    df["subject_key_2"] = df.apply(
        lambda r: _subject_key(r["dataset_2"], r["subject_id_2"]),
        axis=1,
    )

    df = df[df["subject_key_1"] != df["subject_key_2"]].copy()

    if df.empty:
        atomic_write_csv(pd.DataFrame(columns=out_cols), out_csv)
        print("[NEAR GROUP] no confirmed cross-subject review decisions found")
        print(f"[NEAR GROUP] saved -> {out_csv}")
        return out_csv

    # canonicalize undirected subject pair ordering
    flip = df["subject_key_1"] > df["subject_key_2"]

    df["dataset_a"] = np.where(flip, df["dataset_2"], df["dataset_1"])
    df["subject_id_a"] = np.where(flip, df["subject_id_2"], df["subject_id_1"])
    df["subject_key_a"] = np.where(flip, df["subject_key_2"], df["subject_key_1"])

    df["dataset_b"] = np.where(flip, df["dataset_1"], df["dataset_2"])
    df["subject_id_b"] = np.where(flip, df["subject_id_1"], df["subject_id_2"])
    df["subject_key_b"] = np.where(flip, df["subject_key_1"], df["subject_key_2"])

    rows: List[Dict[str, object]] = []

    for (ska, skb), g in df.groupby(["subject_key_a", "subject_key_b"], sort=False):
        best = g.iloc[0]

        rows.append(
            {
                "dataset_a": best["dataset_a"],
                "subject_id_a": best["subject_id_a"],
                "subject_key_a": ska,
                "dataset_b": best["dataset_b"],
                "subject_id_b": best["subject_id_b"],
                "subject_key_b": skb,
                "n_yes_scan_pairs": int(len(g)),
            }
        )

    out_df = pd.DataFrame(rows, columns=out_cols)

    if not out_df.empty:
        out_df = out_df.sort_values(
            ["subject_key_a", "subject_key_b"],
            kind="mergesort",
        ).reset_index(drop=True)

    atomic_write_csv(out_df, out_csv)

    print(f"[NEAR GROUP] confirmed subject edge rows: {len(out_df):,}")
    print(f"[NEAR GROUP] saved -> {out_csv}")

    return out_csv


def _read_subject_edges(
    in_csv: Union[str, Path],
) -> pd.DataFrame:
    in_csv = Path(in_csv)

    required_cols = [
        "dataset_a",
        "subject_id_a",
        "subject_key_a",
        "dataset_b",
        "subject_id_b",
        "subject_key_b",
        "n_yes_scan_pairs",
    ]

    df = pd.read_csv(in_csv, dtype=str).fillna("")
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{in_csv} is missing required columns: {missing}")

    for c in required_cols:
        df[c] = df[c].fillna("").astype(str)

    df["n_yes_scan_pairs"] = pd.to_numeric(
        df["n_yes_scan_pairs"], errors="raise"
    ).astype(int)

    return df


def build_subject_groups(
    subject_edges_csv: Union[str, Path],
    out_csv: Union[str, Path],
) -> Path:
    subject_edges_csv = Path(subject_edges_csv)
    out_csv = Path(out_csv)

    df = _read_subject_edges(subject_edges_csv)

    if df.empty:
        out_df = pd.DataFrame(
            columns=[
                "group_id",
                "group_size",
                "dataset",
                "subject_id",
                "subject_key",
            ]
        )
        atomic_write_csv(out_df, out_csv)
        return out_csv

    uf = UnionFind()
    subject_meta: Dict[str, Tuple[str, str]] = {}

    for _, r in df.iterrows():
        ska = str(r["subject_key_a"])
        skb = str(r["subject_key_b"])

        dsa = str(r["dataset_a"])
        dsb = str(r["dataset_b"])
        sia = str(r["subject_id_a"])
        sib = str(r["subject_id_b"])

        uf.add(ska)
        uf.add(skb)
        uf.union(ska, skb)

        subject_meta[ska] = (dsa, sia)
        subject_meta[skb] = (dsb, sib)

    comp_map: Dict[str, List[str]] = {}
    for key in sorted(subject_meta.keys()):
        root = uf.find(key)
        comp_map.setdefault(root, []).append(key)

    rows: List[Dict[str, object]] = []
    comps = [sorted(v) for v in comp_map.values() if len(v) > 1]
    comps = sorted(comps, key=lambda xs: (len(xs), xs[0]))

    for i, comp in enumerate(comps, start=1):
        group_id = f"group_{i:06d}"
        group_size = int(len(comp))

        for subject_key in comp:
            ds, sid = subject_meta[subject_key]
            rows.append(
                {
                    "group_id": group_id,
                    "group_size": group_size,
                    "dataset": ds,
                    "subject_id": sid,
                    "subject_key": subject_key,
                }
            )

    out_df = pd.DataFrame(
        rows,
        columns=[
            "group_id",
            "group_size",
            "dataset",
            "subject_id",
            "subject_key",
        ],
    )

    if not out_df.empty:
        out_df = out_df.sort_values(
            ["group_id", "subject_key"],
            kind="mergesort",
        ).reset_index(drop=True)

    atomic_write_csv(out_df, out_csv)

    print(
        f"[NEAR GROUP] subject groups: {out_df['group_id'].nunique() if not out_df.empty else 0:,}"
    )
    print(f"[NEAR GROUP] subject-group rows: {len(out_df):,}")
    print(f"[NEAR GROUP] saved -> {out_csv}")

    return out_csv
