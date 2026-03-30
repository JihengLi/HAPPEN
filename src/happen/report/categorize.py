"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd

from pathlib import Path
from typing import List, Optional, Tuple, Union


def report_by_dataset(
    in_csv: Union[str, Path],
    out_dir: Union[str, Path],
    modality: Optional[str] = None,
    sort_cols_detail: Tuple[str, ...] = ("group_id", "resolved_path"),
) -> bool:
    in_csv = Path(in_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(in_csv, dtype=str)
    for c in ["dataset", "group_id", "resolved_path", "modality"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    df[["dataset", "group_id", "resolved_path", "modality"]] = (
        df[["dataset", "group_id", "resolved_path", "modality"]].fillna("").astype(str)
    )
    if modality:
        df = df[df["modality"] == modality].copy()
    df = df[df["dataset"] != ""].copy()
    if df.empty:
        return False

    datasets_in_csv: List[str] = df["dataset"].dropna().astype(str).tolist()
    seen_order: List[str] = []
    seen_set = set()
    for d in datasets_in_csv:
        if d and d not in seen_set:
            seen_order.append(d)
            seen_set.add(d)

    wrote_any = False

    for ds in seen_order:
        df_d = df[df["dataset"] == ds].copy()
        if df_d.empty:
            continue
        sizes = df_d.groupby("group_id")["resolved_path"].nunique()
        keep_gids = set(sizes.index[sizes.values > 1])
        df_dups = df_d[df_d["group_id"].isin(keep_gids)].copy()
        if df_dups.empty:
            continue
        out_csv_ds = out_dir / f"{ds}.csv"
        df_dups.sort_values(list(sort_cols_detail), kind="mergesort").to_csv(
            out_csv_ds,
            index=False,
        )
        wrote_any = True
    return wrote_any


def report_by_category(
    in_dir: Union[str, Path],
    out_dir: Union[str, Path],
    out_stats_dir: Union[str, Path],
    modality: Optional[str] = None,
) -> bool:
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if out_stats_dir is None:
        raise ValueError("out_stats_dir is required")
    out_stats_dir = Path(out_stats_dir)
    out_stats_dir.mkdir(parents=True, exist_ok=True)

    required_cols = [
        "group_id",
        "can_hash",
        "modality",
        "dataset",
        "subject_id",
        "session_id",
        "candidate",
        "resolved_path",
    ]

    dfs = []
    for p in sorted(in_dir.glob("*.csv")):
        dfp = pd.read_csv(p, dtype=str, usecols=lambda c: c in required_cols)
        missing = [c for c in required_cols if c not in dfp.columns]
        if missing:
            raise ValueError(f"{p} missing column: {missing}")
        dfs.append(dfp.copy())

    if not dfs:
        raise SystemExit(f"[ERROR] directory is empty: {in_dir}")

    df0 = pd.concat(dfs, ignore_index=True)

    if not df0.columns.is_unique:
        df0 = df0.loc[:, ~df0.columns.duplicated()]

    for c in [
        "dataset",
        "subject_id",
        "session_id",
        "candidate",
        "resolved_path",
        "modality",
    ]:
        df0[c] = df0[c].fillna("").astype(str)

    if modality:
        df0 = df0[df0["modality"] == modality].copy()

    gsize_all = df0.groupby("group_id").size()
    dup_gids_all = set(gsize_all[gsize_all > 1].index)

    def _write_dataset_stats(detail_df: pd.DataFrame, stats_path: Path) -> None:
        if detail_df.empty:
            pd.DataFrame(columns=["dataset", "n_images"]).to_csv(
                stats_path, index=False
            )
            return
        rp_cols = [c for c in detail_df.columns if c.endswith("_resolved_path")]
        if not rp_cols:
            pd.DataFrame(columns=["dataset", "n_images"]).to_csv(
                stats_path, index=False
            )
            return
        melted = detail_df.melt(
            id_vars=["dataset"], value_vars=rp_cols, value_name="resolved_path"
        )[["dataset", "resolved_path"]]
        melted["resolved_path"] = (
            melted["resolved_path"].fillna("").astype(str).str.strip()
        )
        melted = melted[melted["resolved_path"] != ""]
        stats = (
            melted.groupby("dataset")["resolved_path"]
            .nunique()
            .reset_index(name="n_images")
            .sort_values(["n_images", "dataset"], ascending=[False, True])
        )
        stats.to_csv(stats_path, index=False)

    # ---------------- within sessions ----------------
    df = df0.copy()
    df["triple"] = df["dataset"] + "|" + df["subject_id"] + "|" + df["session_id"]
    triple_uni = df.groupby("group_id")["triple"].nunique()
    first_subj = df.groupby("group_id")["subject_id"].first()
    keep_gids = set(triple_uni[triple_uni == 1].index) & set(
        first_subj[first_subj != ""].index
    )
    df_keep = df[df["group_id"].isin(keep_gids)].copy()

    rows = []
    max_n = 0
    for gid, sub in df_keep.groupby("group_id", sort=False):
        ds = sub.iloc[0]["dataset"]
        subid = sub.iloc[0]["subject_id"]
        ses = sub.iloc[0]["session_id"]
        sub = sub.sort_values(["resolved_path", "candidate"], kind="mergesort")
        pairs = list(zip(sub["candidate"].tolist(), sub["resolved_path"].tolist()))
        max_n = max(max_n, len(pairs))
        rows.append(
            {"dataset": ds, "subject_id": subid, "session_id": ses, "_pairs": pairs}
        )

    dup_cols = []
    for i in range(1, max_n + 1):
        dup_cols += [f"duplicate_{i}_candidate", f"duplicate_{i}_resolved_path"]

    out_rows = []
    for r in rows:
        base = {
            "dataset": r["dataset"],
            "subject_id": r["subject_id"],
            "session_id": r["session_id"],
        }
        pairs = r["_pairs"]
        for i in range(max_n):
            cand = pairs[i][0] if i < len(pairs) else ""
            rp = pairs[i][1] if i < len(pairs) else ""
            base[f"duplicate_{i+1}_candidate"] = cand
            base[f"duplicate_{i+1}_resolved_path"] = rp
        out_rows.append(base)

    out_df = pd.DataFrame(
        out_rows, columns=["dataset", "subject_id", "session_id"] + dup_cols
    )
    out_df = out_df.sort_values(
        ["dataset", "subject_id", "session_id"], kind="mergesort"
    )
    out_csv = out_dir / "within_sessions.csv"
    out_df.to_csv(out_csv, index=False)
    print(
        f"[within_sessions] input rows: {len(df0):,}, kept groups: {len(rows):,}, max duplicates per group: {max_n}"
    )
    print(f"[within_sessions] saved -> {out_csv}")

    _write_dataset_stats(out_df, out_stats_dir / "within_sessions.csv")

    # ---------------- across sessions within subjects ----------------
    df = df0[df0["group_id"].isin(dup_gids_all)].copy()
    df_nonempty_ses = df[df["session_id"] != ""].copy()
    sess_counts = (
        df_nonempty_ses.groupby(["group_id", "dataset", "subject_id"])["session_id"]
        .nunique()
        .reset_index(name="n_sessions")
    )
    targets = sess_counts[sess_counts["n_sessions"] >= 2].drop(columns=["n_sessions"])
    df_keep = df_nonempty_ses.merge(
        targets, on=["group_id", "dataset", "subject_id"], how="inner"
    )

    rows = []
    max_sessions = 0
    df_keep = df_keep.sort_values(
        [
            "group_id",
            "dataset",
            "subject_id",
            "session_id",
            "resolved_path",
            "candidate",
        ],
        kind="mergesort",
    )
    for (gid, ds, subj), sub in df_keep.groupby(
        ["group_id", "dataset", "subject_id"], sort=False
    ):
        reps = sub.groupby("session_id", sort=True, dropna=False).nth(0).reset_index()
        if reps.shape[0] < 2:
            continue
        max_sessions = max(max_sessions, reps.shape[0])
        rows.append(
            {
                "dataset": ds,
                "subject_id": subj,
                "_sessions": reps["session_id"].tolist(),
                "_cands": reps["candidate"].tolist(),
                "_rpaths": reps["resolved_path"].tolist(),
            }
        )

    out_cols = ["dataset", "subject_id"]
    for i in range(1, max_sessions + 1):
        out_cols += [
            f"session_id_{i}",
            f"duplicate_{i}_candidate",
            f"duplicate_{i}_resolved_path",
        ]

    out_rows = []
    for r in rows:
        base = {"dataset": r["dataset"], "subject_id": r["subject_id"]}
        for i in range(max_sessions):
            sid = r["_sessions"][i] if i < len(r["_sessions"]) else ""
            cand = r["_cands"][i] if i < len(r["_cands"]) else ""
            rpat = r["_rpaths"][i] if i < len(r["_rpaths"]) else ""
            base[f"session_id_{i+1}"] = sid
            base[f"duplicate_{i+1}_candidate"] = cand
            base[f"duplicate_{i+1}_resolved_path"] = rpat
        out_rows.append(base)

    out_df = pd.DataFrame(out_rows, columns=out_cols)
    out_df = out_df.sort_values(["dataset", "subject_id"], kind="mergesort")
    out_csv = out_dir / "within_subjects.csv"
    out_df.to_csv(out_csv, index=False)
    print(
        f"[within_subjects] input rows: {len(df0):,}, output groups: {len(out_df):,}, max sessions per row: {max_sessions}"
    )
    print(f"[within_subjects] saved -> {out_csv}")

    _write_dataset_stats(out_df, out_stats_dir / "within_subjects.csv")

    # ---------------- across subjects within datasets ----------------
    df = df0[df0["group_id"].isin(dup_gids_all)].copy()
    df = df.sort_values(
        [
            "group_id",
            "dataset",
            "subject_id",
            "session_id",
            "resolved_path",
            "candidate",
        ],
        kind="mergesort",
    )

    rows = []
    max_items = 0
    for (gid, ds), sub in df.groupby(["group_id", "dataset"], sort=False):
        unique_subjects = {s for s in sub["subject_id"].astype(str) if s}
        if len(unique_subjects) < 2:
            continue
        items = list(
            zip(
                sub["subject_id"].tolist(),
                sub["session_id"].tolist(),
                sub["candidate"].tolist(),
                sub["resolved_path"].tolist(),
            )
        )
        max_items = max(max_items, len(items))
        rows.append({"dataset": ds, "_items": items})

    out_cols = ["dataset"]
    for i in range(1, max_items + 1):
        out_cols += [
            f"subject_id_{i}",
            f"session_id_{i}",
            f"duplicate_{i}_candidate",
            f"duplicate_{i}_resolved_path",
        ]

    out_rows = []
    for r in rows:
        base = {"dataset": r["dataset"]}
        items = r["_items"]
        for i in range(max_items):
            if i < len(items):
                subid, sesid, cand, rpath = items[i]
            else:
                subid = sesid = cand = rpath = ""
            base[f"subject_id_{i+1}"] = subid
            base[f"session_id_{i+1}"] = sesid
            base[f"duplicate_{i+1}_candidate"] = cand
            base[f"duplicate_{i+1}_resolved_path"] = rpath
        out_rows.append(base)

    out_df = pd.DataFrame(out_rows, columns=out_cols)
    out_df = out_df.sort_values(
        ["dataset", "subject_id_1", "session_id_1"], kind="mergesort"
    )
    out_csv = out_dir / "within_datasets.csv"
    out_df.to_csv(out_csv, index=False)
    print(
        f"[within_datasets] input rows: {len(df0):,}, output groups: {len(out_df):,}, max items per row: {max_items}"
    )
    print(f"[within_datasets] saved -> {out_csv}")

    _write_dataset_stats(out_df, out_stats_dir / "within_datasets.csv")

    return True


def report_by_category_across_datasets(
    in_csv: Union[str, Path],
    out_dir: Union[str, Path],
    out_stats_dir: Union[str, Path],
    modality: Optional[str] = None,
) -> bool:
    in_csv = Path(in_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    required = [
        "group_id",
        "can_hash",
        "modality",
        "dataset",
        "subject_id",
        "session_id",
        "candidate",
        "resolved_path",
    ]
    df = pd.read_csv(in_csv, dtype=str)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required].copy()
    for c in required:
        df[c] = df[c].fillna("").astype(str)

    if modality:
        df = df[df["modality"] == modality].copy()

    df = df[df["dataset"] != ""].copy()
    if df.empty:
        (out_dir / "set_hist.csv").write_text("", encoding="utf-8")
        pd.DataFrame(
            columns=[
                "dataset_1",
                "subject_id_1",
                "session_id_1",
                "duplicate_1_candidate",
                "duplicate_1_resolved_path",
            ]
        ).to_csv(out_dir / "across_datasets.csv", index=False)
        return False

    gsize = df.groupby("group_id").size()
    dup_gids = set(gsize[gsize > 1].index)
    df = df[df["group_id"].isin(dup_gids)].copy()
    if df.empty:
        (out_dir / "set_hist.csv").write_text("", encoding="utf-8")
        pd.DataFrame(
            columns=[
                "dataset_1",
                "subject_id_1",
                "session_id_1",
                "duplicate_1_candidate",
                "duplicate_1_resolved_path",
            ]
        ).to_csv(out_dir / "across_datasets.csv", index=False)
        return False

    def _clean_set(s: pd.Series) -> list[str]:
        vals = {x.strip() for x in s.dropna().astype(str) if x and x.strip()}
        return sorted(vals)

    ds_sets = (
        df.groupby("group_id", dropna=False)["dataset"]
        .apply(_clean_set)
        .reset_index(name="datasets")
    )
    ds_sets["set_size"] = ds_sets["datasets"].map(len)

    cross_gids = set(ds_sets.loc[ds_sets["set_size"] >= 2, "group_id"])

    ds_sets["set_key"] = ds_sets["datasets"].map(lambda xs: "|".join(xs) if xs else "")
    ds_sets_cross = ds_sets.loc[ds_sets["set_size"] >= 2, ["group_id", "set_key"]]
    df_cross = df.merge(ds_sets_cross, on="group_id", how="inner")
    set_hist = (
        df_cross.groupby("set_key", dropna=False)["resolved_path"]
        .size()
        .reset_index(name="n_images")
        .sort_values(["n_images", "set_key"], ascending=[False, True])
    )
    set_hist.to_csv(out_stats_dir / "across_datasets.csv", index=False)

    df = df[df["group_id"].isin(cross_gids)].copy()
    df = df.sort_values(
        [
            "group_id",
            "dataset",
            "subject_id",
            "session_id",
            "resolved_path",
            "candidate",
        ],
        kind="mergesort",
    )

    rows, max_items = [], 0
    for gid, sub in df.groupby("group_id", sort=False):
        items = list(
            zip(
                sub["dataset"].tolist(),
                sub["subject_id"].tolist(),
                sub["session_id"].tolist(),
                sub["candidate"].tolist(),
                sub["resolved_path"].tolist(),
            )
        )
        if len(items) < 2:
            continue
        max_items = max(max_items, len(items))
        rows.append({"_items": items})

    if rows:
        out_cols = []
        for i in range(1, max_items + 1):
            out_cols += [
                f"dataset_{i}",
                f"subject_id_{i}",
                f"session_id_{i}",
                f"duplicate_{i}_candidate",
                f"duplicate_{i}_resolved_path",
            ]
        out_rows = []
        for r in rows:
            base = {}
            items = r["_items"]
            for i in range(max_items):
                if i < len(items):
                    ds, subj, ses, cand, rpath = items[i]
                else:
                    ds = subj = ses = cand = rpath = ""
                base[f"dataset_{i+1}"] = ds
                base[f"subject_id_{i+1}"] = subj
                base[f"session_id_{i+1}"] = ses
                base[f"duplicate_{i+1}_candidate"] = cand
                base[f"duplicate_{i+1}_resolved_path"] = rpath
            out_rows.append(base)
        out_df = pd.DataFrame(out_rows, columns=out_cols)
        sort_cols = [
            c
            for c in ["dataset_1", "subject_id_1", "session_id_1"]
            if c in out_df.columns
        ]
        if sort_cols:
            out_df = out_df.sort_values(sort_cols, kind="mergesort")
    else:
        out_df = pd.DataFrame(
            columns=[
                "dataset_1",
                "subject_id_1",
                "session_id_1",
                "duplicate_1_candidate",
                "duplicate_1_resolved_path",
            ]
        )

    out_df.to_csv(out_dir / "across_datasets.csv", index=False)
    return len(rows) > 0
