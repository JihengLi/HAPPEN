"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd


def atomic_write_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(out_path)


def _pair_key(scan_uid_a: str, scan_uid_b: str) -> str:
    a = str(scan_uid_a).strip()
    b = str(scan_uid_b).strip()
    x, y = sorted([a, b])
    return f"{x}__{y}"


def png_rel_path(
    dataset: str,
    subject_id: str,
    session_id: str,
    scan_uid: str,
) -> str:
    base = Path("assets") / "png" / str(dataset).strip() / str(subject_id).strip()
    ses = str(session_id).strip()
    if ses:
        base = base / ses
    return (base / f"{str(scan_uid).strip()}.png").as_posix()


def pair_asset_rel_path(
    kind: str,
    scan_uid_a: str,
    scan_uid_b: str,
) -> str:
    if kind not in {"diff", "checkerboard"}:
        raise ValueError(f"unknown pair asset kind: {kind}")
    return (
        Path("assets") / kind / f"{_pair_key(scan_uid_a, scan_uid_b)}.png"
    ).as_posix()


def _read_scan_candidates(in_csv: Union[str, Path]) -> pd.DataFrame:
    in_csv = Path(in_csv)

    required_cols = [
        "dataset_a",
        "subject_id_a",
        "session_id_a",
        "scan_uid_a",
        "dataset_b",
        "subject_id_b",
        "session_id_b",
        "scan_uid_b",
        "similarity",
        "best_rank",
        "n_directions",
    ]

    df = pd.read_csv(in_csv, dtype=str).fillna("")
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{in_csv} is missing required columns: {missing}")

    for c in required_cols:
        df[c] = df[c].fillna("").astype(str)

    df["similarity"] = pd.to_numeric(df["similarity"], errors="raise").astype(float)
    df["best_rank"] = pd.to_numeric(df["best_rank"], errors="raise").astype(int)
    df["n_directions"] = pd.to_numeric(df["n_directions"], errors="raise").astype(int)

    return df


def _read_subject_edges(in_csv: Union[str, Path]) -> pd.DataFrame:
    in_csv = Path(in_csv)

    required_cols = [
        "dataset_a",
        "subject_id_a",
        "subject_key_a",
        "dataset_b",
        "subject_id_b",
        "subject_key_b",
        "max_similarity",
        "n_scan_pairs",
        "exemplar_scan_uid_a",
        "exemplar_scan_uid_b",
    ]

    df = pd.read_csv(in_csv, dtype=str).fillna("")
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{in_csv} is missing required columns: {missing}")

    for c in required_cols:
        df[c] = df[c].fillna("").astype(str)

    df["max_similarity"] = pd.to_numeric(df["max_similarity"], errors="raise").astype(
        float
    )
    df["n_scan_pairs"] = pd.to_numeric(df["n_scan_pairs"], errors="raise").astype(int)

    return df


def _read_subject_groups(in_csv: Union[str, Path]) -> pd.DataFrame:
    in_csv = Path(in_csv)

    required_cols = [
        "group_id",
        "group_size",
        "dataset",
        "subject_id",
        "subject_key",
    ]

    df = pd.read_csv(in_csv, dtype=str).fillna("")
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{in_csv} is missing required columns: {missing}")

    for c in required_cols:
        df[c] = df[c].fillna("").astype(str)

    df["group_size"] = pd.to_numeric(df["group_size"], errors="raise").astype(int)

    return df


def _register_scan_meta(
    meta_map: Dict[str, Tuple[str, str, str]],
    scan_uid: str,
    dataset: str,
    subject_id: str,
    session_id: str,
) -> None:
    key = str(scan_uid)
    value = (str(dataset), str(subject_id), str(session_id))
    if key in meta_map and meta_map[key] != value:
        raise ValueError(
            f"inconsistent scan metadata for scan_uid={key}: "
            f"{meta_map[key]} vs {value}"
        )
    meta_map[key] = value


def build_scan_level_review(
    scan_candidates_csv: Union[str, Path],
    out_csv: Union[str, Path],
    max_candidates_per_query: Optional[int] = None,
) -> Path:
    scan_candidates_csv = Path(scan_candidates_csv)
    out_csv = Path(out_csv)

    df = _read_scan_candidates(scan_candidates_csv)

    if df.empty:
        out_df = pd.DataFrame(
            columns=[
                "query_dataset",
                "query_subject_id",
                "query_session_id",
                "query_scan_uid",
                "query_png_path",
                "n_candidates",
            ]
        )
        atomic_write_csv(out_df, out_csv)
        return out_csv

    rows: List[Dict[str, object]] = []

    for _, r in df.iterrows():
        rows.append(
            {
                "query_dataset": r["dataset_a"],
                "query_subject_id": r["subject_id_a"],
                "query_session_id": r["session_id_a"],
                "query_scan_uid": r["scan_uid_a"],
                "query_png_path": png_rel_path(
                    r["dataset_a"],
                    r["subject_id_a"],
                    r["session_id_a"],
                    r["scan_uid_a"],
                ),
                "candidate_dataset": r["dataset_b"],
                "candidate_subject_id": r["subject_id_b"],
                "candidate_session_id": r["session_id_b"],
                "candidate_scan_uid": r["scan_uid_b"],
                "candidate_png_path": png_rel_path(
                    r["dataset_b"],
                    r["subject_id_b"],
                    r["session_id_b"],
                    r["scan_uid_b"],
                ),
                "candidate_diff_path": pair_asset_rel_path(
                    "diff",
                    r["scan_uid_a"],
                    r["scan_uid_b"],
                ),
                "candidate_checkerboard_path": pair_asset_rel_path(
                    "checkerboard",
                    r["scan_uid_a"],
                    r["scan_uid_b"],
                ),
                "similarity": float(r["similarity"]),
            }
        )
        rows.append(
            {
                "query_dataset": r["dataset_b"],
                "query_subject_id": r["subject_id_b"],
                "query_session_id": r["session_id_b"],
                "query_scan_uid": r["scan_uid_b"],
                "query_png_path": png_rel_path(
                    r["dataset_b"],
                    r["subject_id_b"],
                    r["session_id_b"],
                    r["scan_uid_b"],
                ),
                "candidate_dataset": r["dataset_a"],
                "candidate_subject_id": r["subject_id_a"],
                "candidate_session_id": r["session_id_a"],
                "candidate_scan_uid": r["scan_uid_a"],
                "candidate_png_path": png_rel_path(
                    r["dataset_a"],
                    r["subject_id_a"],
                    r["session_id_a"],
                    r["scan_uid_a"],
                ),
                "candidate_diff_path": pair_asset_rel_path(
                    "diff",
                    r["scan_uid_a"],
                    r["scan_uid_b"],
                ),
                "candidate_checkerboard_path": pair_asset_rel_path(
                    "checkerboard",
                    r["scan_uid_a"],
                    r["scan_uid_b"],
                ),
                "similarity": float(r["similarity"]),
            }
        )

    long_df = pd.DataFrame(rows)
    long_df = long_df.sort_values(
        [
            "query_dataset",
            "query_subject_id",
            "query_session_id",
            "query_scan_uid",
            "similarity",
            "candidate_scan_uid",
        ],
        ascending=[True, True, True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    grouped = list(
        long_df.groupby(
            [
                "query_dataset",
                "query_subject_id",
                "query_session_id",
                "query_scan_uid",
                "query_png_path",
            ],
            sort=False,
        )
    )

    if max_candidates_per_query is None:
        max_slots = max(len(g) for _, g in grouped)
    else:
        max_slots = max(0, int(max_candidates_per_query))

    rows_out: List[Dict[str, object]] = []

    for key, g in grouped:
        (
            query_dataset,
            query_subject_id,
            query_session_id,
            query_scan_uid,
            query_png_path,
        ) = key

        g = g.sort_values(
            ["similarity", "candidate_scan_uid"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)

        if max_candidates_per_query is not None:
            g = g.head(int(max_candidates_per_query)).copy()

        row: Dict[str, object] = {
            "query_dataset": query_dataset,
            "query_subject_id": query_subject_id,
            "query_session_id": query_session_id,
            "query_scan_uid": query_scan_uid,
            "query_png_path": query_png_path,
            "n_candidates": int(len(g)),
        }

        for i in range(max_slots):
            j = i + 1
            if i < len(g):
                r = g.iloc[i]
                row[f"candidate_{j}_dataset"] = r["candidate_dataset"]
                row[f"candidate_{j}_subject_id"] = r["candidate_subject_id"]
                row[f"candidate_{j}_session_id"] = r["candidate_session_id"]
                row[f"candidate_{j}_scan_uid"] = r["candidate_scan_uid"]
                row[f"candidate_{j}_similarity"] = float(r["similarity"])
                row[f"candidate_{j}_png_path"] = r["candidate_png_path"]
                row[f"candidate_{j}_diff_path"] = r["candidate_diff_path"]
                row[f"candidate_{j}_checkerboard_path"] = r[
                    "candidate_checkerboard_path"
                ]
            else:
                row[f"candidate_{j}_dataset"] = ""
                row[f"candidate_{j}_subject_id"] = ""
                row[f"candidate_{j}_session_id"] = ""
                row[f"candidate_{j}_scan_uid"] = ""
                row[f"candidate_{j}_similarity"] = ""
                row[f"candidate_{j}_png_path"] = ""
                row[f"candidate_{j}_diff_path"] = ""
                row[f"candidate_{j}_checkerboard_path"] = ""

        rows_out.append(row)

    base_cols = [
        "query_dataset",
        "query_subject_id",
        "query_session_id",
        "query_scan_uid",
        "query_png_path",
        "n_candidates",
    ]
    cand_cols: List[str] = []
    for j in range(1, max_slots + 1):
        cand_cols.extend(
            [
                f"candidate_{j}_dataset",
                f"candidate_{j}_subject_id",
                f"candidate_{j}_session_id",
                f"candidate_{j}_scan_uid",
                f"candidate_{j}_similarity",
                f"candidate_{j}_png_path",
                f"candidate_{j}_diff_path",
                f"candidate_{j}_checkerboard_path",
            ]
        )

    out_df = pd.DataFrame(rows_out, columns=base_cols + cand_cols)
    out_df = out_df.sort_values(
        ["query_dataset", "query_subject_id", "query_session_id", "query_scan_uid"],
        kind="mergesort",
    ).reset_index(drop=True)

    atomic_write_csv(out_df, out_csv)

    print(f"[REVIEW FORMAT] scan-level rows: {len(out_df):,}")
    print(f"[REVIEW FORMAT] saved -> {out_csv}")

    return out_csv


def build_subject_level_review(
    scan_candidates_csv: Union[str, Path],
    subject_edges_csv: Union[str, Path],
    subject_groups_csv: Union[str, Path],
    out_csv: Union[str, Path],
    max_candidates_per_query: Optional[int] = None,
) -> Path:
    scan_candidates_csv = Path(scan_candidates_csv)
    subject_edges_csv = Path(subject_edges_csv)
    subject_groups_csv = Path(subject_groups_csv)
    out_csv = Path(out_csv)

    scan_df = _read_scan_candidates(scan_candidates_csv)
    edges_df = _read_subject_edges(subject_edges_csv)
    groups_df = _read_subject_groups(subject_groups_csv)

    if groups_df.empty:
        out_df = pd.DataFrame(
            columns=[
                "group_id",
                "group_size",
                "query_dataset",
                "query_subject_id",
                "n_candidates",
            ]
        )
        atomic_write_csv(out_df, out_csv)
        return out_csv

    scan_meta: Dict[str, Tuple[str, str, str]] = {}
    for _, r in scan_df.iterrows():
        _register_scan_meta(
            scan_meta,
            r["scan_uid_a"],
            r["dataset_a"],
            r["subject_id_a"],
            r["session_id_a"],
        )
        _register_scan_meta(
            scan_meta,
            r["scan_uid_b"],
            r["dataset_b"],
            r["subject_id_b"],
            r["session_id_b"],
        )

    group_to_subjects: Dict[str, List[Tuple[str, str, str]]] = {}
    subject_to_group: Dict[str, str] = {}
    group_to_size: Dict[str, int] = {}

    for _, r in groups_df.iterrows():
        gid = str(r["group_id"])
        gsize = int(r["group_size"])
        ds = str(r["dataset"])
        sid = str(r["subject_id"])
        skey = str(r["subject_key"])

        group_to_subjects.setdefault(gid, []).append((skey, ds, sid))
        subject_to_group[skey] = gid
        group_to_size[gid] = gsize

    adjacency: Dict[str, List[Dict[str, object]]] = {}

    for _, r in edges_df.iterrows():
        ska = str(r["subject_key_a"])
        skb = str(r["subject_key_b"])

        if ska not in subject_to_group or skb not in subject_to_group:
            continue

        gid_a = subject_to_group[ska]
        gid_b = subject_to_group[skb]
        if gid_a != gid_b:
            raise ValueError(
                f"subject edge crosses groups unexpectedly: {gid_a} vs {gid_b}"
            )

        ex_uid_a = str(r["exemplar_scan_uid_a"])
        ex_uid_b = str(r["exemplar_scan_uid_b"])

        if ex_uid_a not in scan_meta:
            raise ValueError(
                f"exemplar_scan_uid_a not found in scan metadata: {ex_uid_a}"
            )
        if ex_uid_b not in scan_meta:
            raise ValueError(
                f"exemplar_scan_uid_b not found in scan metadata: {ex_uid_b}"
            )

        _, _, ex_ses_a = scan_meta[ex_uid_a]
        _, _, ex_ses_b = scan_meta[ex_uid_b]

        adjacency.setdefault(ska, []).append(
            {
                "candidate_dataset": str(r["dataset_b"]),
                "candidate_subject_id": str(r["subject_id_b"]),
                "similarity": float(r["max_similarity"]),
                "n_scan_pairs": int(r["n_scan_pairs"]),
                "query_exemplar_session_id": ex_ses_a,
                "query_exemplar_scan_uid": ex_uid_a,
                "query_exemplar_png_path": png_rel_path(
                    r["dataset_a"],
                    r["subject_id_a"],
                    ex_ses_a,
                    ex_uid_a,
                ),
                "candidate_exemplar_session_id": ex_ses_b,
                "candidate_exemplar_scan_uid": ex_uid_b,
                "candidate_exemplar_png_path": png_rel_path(
                    r["dataset_b"],
                    r["subject_id_b"],
                    ex_ses_b,
                    ex_uid_b,
                ),
                "diff_path": pair_asset_rel_path("diff", ex_uid_a, ex_uid_b),
                "checkerboard_path": pair_asset_rel_path(
                    "checkerboard",
                    ex_uid_a,
                    ex_uid_b,
                ),
            }
        )

        adjacency.setdefault(skb, []).append(
            {
                "candidate_dataset": str(r["dataset_a"]),
                "candidate_subject_id": str(r["subject_id_a"]),
                "similarity": float(r["max_similarity"]),
                "n_scan_pairs": int(r["n_scan_pairs"]),
                "query_exemplar_session_id": ex_ses_b,
                "query_exemplar_scan_uid": ex_uid_b,
                "query_exemplar_png_path": png_rel_path(
                    r["dataset_b"],
                    r["subject_id_b"],
                    ex_ses_b,
                    ex_uid_b,
                ),
                "candidate_exemplar_session_id": ex_ses_a,
                "candidate_exemplar_scan_uid": ex_uid_a,
                "candidate_exemplar_png_path": png_rel_path(
                    r["dataset_a"],
                    r["subject_id_a"],
                    ex_ses_a,
                    ex_uid_a,
                ),
                "diff_path": pair_asset_rel_path("diff", ex_uid_a, ex_uid_b),
                "checkerboard_path": pair_asset_rel_path(
                    "checkerboard",
                    ex_uid_a,
                    ex_uid_b,
                ),
            }
        )

    max_slots = 0
    for gid, members in group_to_subjects.items():
        for skey, _, _ in members:
            n = len(adjacency.get(skey, []))
            if max_candidates_per_query is not None:
                n = min(n, int(max_candidates_per_query))
            max_slots = max(max_slots, n)

    rows_out: List[Dict[str, object]] = []

    for gid in sorted(group_to_subjects.keys()):
        members = sorted(group_to_subjects[gid], key=lambda x: x[0])
        gsize = int(group_to_size[gid])

        for skey, ds, sid in members:
            nbrs = adjacency.get(skey, [])
            nbrs = sorted(
                nbrs,
                key=lambda x: (
                    -float(x["similarity"]),
                    str(x["candidate_dataset"]),
                    str(x["candidate_subject_id"]),
                ),
            )

            if max_candidates_per_query is not None:
                nbrs = nbrs[: int(max_candidates_per_query)]

            row: Dict[str, object] = {
                "group_id": gid,
                "group_size": gsize,
                "query_dataset": ds,
                "query_subject_id": sid,
                "n_candidates": int(len(nbrs)),
            }

            for i in range(max_slots):
                j = i + 1
                if i < len(nbrs):
                    n = nbrs[i]
                    row[f"candidate_{j}_dataset"] = n["candidate_dataset"]
                    row[f"candidate_{j}_subject_id"] = n["candidate_subject_id"]
                    row[f"candidate_{j}_similarity"] = float(n["similarity"])
                    row[f"candidate_{j}_n_scan_pairs"] = int(n["n_scan_pairs"])
                    row[f"candidate_{j}_query_exemplar_session_id"] = n[
                        "query_exemplar_session_id"
                    ]
                    row[f"candidate_{j}_query_exemplar_scan_uid"] = n[
                        "query_exemplar_scan_uid"
                    ]
                    row[f"candidate_{j}_query_exemplar_png_path"] = n[
                        "query_exemplar_png_path"
                    ]
                    row[f"candidate_{j}_candidate_exemplar_session_id"] = n[
                        "candidate_exemplar_session_id"
                    ]
                    row[f"candidate_{j}_candidate_exemplar_scan_uid"] = n[
                        "candidate_exemplar_scan_uid"
                    ]
                    row[f"candidate_{j}_candidate_exemplar_png_path"] = n[
                        "candidate_exemplar_png_path"
                    ]
                    row[f"candidate_{j}_diff_path"] = n["diff_path"]
                    row[f"candidate_{j}_checkerboard_path"] = n["checkerboard_path"]
                else:
                    row[f"candidate_{j}_dataset"] = ""
                    row[f"candidate_{j}_subject_id"] = ""
                    row[f"candidate_{j}_similarity"] = ""
                    row[f"candidate_{j}_n_scan_pairs"] = ""
                    row[f"candidate_{j}_query_exemplar_session_id"] = ""
                    row[f"candidate_{j}_query_exemplar_scan_uid"] = ""
                    row[f"candidate_{j}_query_exemplar_png_path"] = ""
                    row[f"candidate_{j}_candidate_exemplar_session_id"] = ""
                    row[f"candidate_{j}_candidate_exemplar_scan_uid"] = ""
                    row[f"candidate_{j}_candidate_exemplar_png_path"] = ""
                    row[f"candidate_{j}_diff_path"] = ""
                    row[f"candidate_{j}_checkerboard_path"] = ""

            rows_out.append(row)

    base_cols = [
        "group_id",
        "group_size",
        "query_dataset",
        "query_subject_id",
        "n_candidates",
    ]
    cand_cols: List[str] = []
    for j in range(1, max_slots + 1):
        cand_cols.extend(
            [
                f"candidate_{j}_dataset",
                f"candidate_{j}_subject_id",
                f"candidate_{j}_similarity",
                f"candidate_{j}_n_scan_pairs",
                f"candidate_{j}_query_exemplar_session_id",
                f"candidate_{j}_query_exemplar_scan_uid",
                f"candidate_{j}_query_exemplar_png_path",
                f"candidate_{j}_candidate_exemplar_session_id",
                f"candidate_{j}_candidate_exemplar_scan_uid",
                f"candidate_{j}_candidate_exemplar_png_path",
                f"candidate_{j}_diff_path",
                f"candidate_{j}_checkerboard_path",
            ]
        )

    out_df = pd.DataFrame(rows_out, columns=base_cols + cand_cols)
    out_df = out_df.sort_values(
        ["group_id", "query_dataset", "query_subject_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    atomic_write_csv(out_df, out_csv)

    print(f"[REVIEW FORMAT] subject-level rows: {len(out_df):,}")
    print(f"[REVIEW FORMAT] saved -> {out_csv}")

    return out_csv


def build_review_tables(
    review_dir: Union[str, Path],
    scan_candidates_csv: Union[str, Path],
    subject_edges_csv: Union[str, Path],
    subject_groups_csv: Union[str, Path],
    max_scan_candidates_per_query: Optional[int] = None,
    max_subject_candidates_per_query: Optional[int] = None,
) -> Tuple[Path, Path]:
    review_dir = Path(review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)

    scan_level_review_csv = review_dir / "scan_level_review.csv"
    subject_level_review_csv = review_dir / "subject_level_review.csv"

    build_scan_level_review(
        scan_candidates_csv=scan_candidates_csv,
        out_csv=scan_level_review_csv,
        max_candidates_per_query=max_scan_candidates_per_query,
    )

    build_subject_level_review(
        scan_candidates_csv=scan_candidates_csv,
        subject_edges_csv=subject_edges_csv,
        subject_groups_csv=subject_groups_csv,
        out_csv=subject_level_review_csv,
        max_candidates_per_query=max_subject_candidates_per_query,
    )

    print(f"[REVIEW FORMAT] outputs saved under -> {review_dir}")

    return scan_level_review_csv, subject_level_review_csv
