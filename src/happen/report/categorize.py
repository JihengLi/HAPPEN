"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd


BASE_REQUIRED_COLS = [
    "group_id",
    "modality",
    "dataset",
    "subject_id",
    "session_id",
    "candidate",
    "resolved_path",
]


def _read_base_csv(
    in_csv: Union[str, Path],
    modality: Optional[str] = None,
) -> pd.DataFrame:
    in_csv = Path(in_csv)
    df = pd.read_csv(in_csv, dtype=str)

    missing = [c for c in BASE_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{in_csv} missing required columns: {missing}")

    df = df[BASE_REQUIRED_COLS].copy()
    for c in BASE_REQUIRED_COLS:
        df[c] = df[c].fillna("").astype(str)

    if modality:
        df = df[df["modality"] == modality].copy()

    return df.reset_index(drop=True)


def _read_base_dir(
    in_dir: Union[str, Path],
    modality: Optional[str] = None,
) -> pd.DataFrame:
    in_dir = Path(in_dir)

    dfs: List[pd.DataFrame] = []
    for p in sorted(in_dir.glob("*.csv")):
        dfp = pd.read_csv(p, dtype=str, usecols=lambda c: c in BASE_REQUIRED_COLS)
        missing = [c for c in BASE_REQUIRED_COLS if c not in dfp.columns]
        if missing:
            raise ValueError(f"{p} missing required columns: {missing}")
        dfp = dfp[BASE_REQUIRED_COLS].copy()
        for c in BASE_REQUIRED_COLS:
            dfp[c] = dfp[c].fillna("").astype(str)
        dfs.append(dfp)

    if not dfs:
        return pd.DataFrame(columns=BASE_REQUIRED_COLS)

    df = pd.concat(dfs, ignore_index=True)
    if modality:
        df = df[df["modality"] == modality].copy()

    return df.reset_index(drop=True)


def _write_csv(df: pd.DataFrame, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


def _sort_if_not_empty(
    df: pd.DataFrame,
    sort_cols: List[str],
) -> pd.DataFrame:
    if df.empty:
        return df
    keep = [c for c in sort_cols if c in df.columns]
    if not keep:
        return df
    return df.sort_values(keep, kind="mergesort").reset_index(drop=True)


def _empty_dataset_stats_csv(out_csv: Path) -> None:
    _write_csv(pd.DataFrame(columns=["dataset", "n_images"]), out_csv)


def _write_dataset_stats(detail_df: pd.DataFrame, stats_path: Path) -> None:
    if detail_df.empty:
        _empty_dataset_stats_csv(stats_path)
        return

    rp_cols = [c for c in detail_df.columns if c.endswith("_resolved_path")]
    if not rp_cols:
        _empty_dataset_stats_csv(stats_path)
        return

    id_vars = [c for c in detail_df.columns if c not in rp_cols]
    melted = detail_df.melt(
        id_vars=[c for c in id_vars if c == "dataset"],
        value_vars=rp_cols,
        value_name="resolved_path",
    )

    if "dataset" not in melted.columns:
        _empty_dataset_stats_csv(stats_path)
        return

    melted["resolved_path"] = melted["resolved_path"].fillna("").astype(str).str.strip()
    melted = melted[melted["resolved_path"] != ""][["dataset", "resolved_path"]]

    if melted.empty:
        _empty_dataset_stats_csv(stats_path)
        return

    stats = (
        melted.groupby("dataset")["resolved_path"]
        .nunique()
        .reset_index(name="n_images")
    )
    stats = _sort_if_not_empty(stats, ["n_images", "dataset"])
    if not stats.empty:
        stats = stats.sort_values(
            ["n_images", "dataset"], ascending=[False, True], kind="mergesort"
        ).reset_index(drop=True)

    _write_csv(stats, stats_path)


def report_by_dataset(
    in_csv: Union[str, Path],
    out_dir: Union[str, Path],
    modality: Optional[str] = None,
    sort_cols_detail: Tuple[str, ...] = ("group_id", "resolved_path"),
) -> bool:
    in_csv = Path(in_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _read_base_csv(in_csv, modality=modality)
    df = df[df["dataset"] != ""].copy()

    if df.empty:
        return False

    datasets_in_csv: List[str] = df["dataset"].tolist()
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

        df_dups = _sort_if_not_empty(df_dups, list(sort_cols_detail))
        out_csv_ds = out_dir / f"{ds}.csv"
        _write_csv(df_dups, out_csv_ds)
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
    out_stats_dir = Path(out_stats_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_stats_dir.mkdir(parents=True, exist_ok=True)

    df0 = _read_base_dir(in_dir, modality=modality)

    # ---------- 1) within sessions ----------
    within_sessions_cols = ["dataset", "subject_id", "session_id"]
    within_sessions_rows = []

    max_n_within_sessions = 0
    if not df0.empty:
        df = df0.copy()
        df["triple"] = df["dataset"] + "|" + df["subject_id"] + "|" + df["session_id"]

        triple_uni = df.groupby("group_id")["triple"].nunique()
        first_subj = df.groupby("group_id")["subject_id"].first()

        keep_gids = set(triple_uni[triple_uni == 1].index) & set(
            first_subj[first_subj != ""].index
        )
        df_keep = df[df["group_id"].isin(keep_gids)].copy()

        for gid, sub in df_keep.groupby("group_id", sort=False):
            ds = sub.iloc[0]["dataset"]
            subid = sub.iloc[0]["subject_id"]
            ses = sub.iloc[0]["session_id"]

            sub = sub.sort_values(["resolved_path", "candidate"], kind="mergesort")
            pairs = list(zip(sub["candidate"].tolist(), sub["resolved_path"].tolist()))
            max_n_within_sessions = max(max_n_within_sessions, len(pairs))

            within_sessions_rows.append(
                {
                    "dataset": ds,
                    "subject_id": subid,
                    "session_id": ses,
                    "_pairs": pairs,
                }
            )

    duplicate_cols = []
    for i in range(1, max_n_within_sessions + 1):
        duplicate_cols += [f"duplicate_{i}_candidate", f"duplicate_{i}_resolved_path"]

    out_rows = []
    for r in within_sessions_rows:
        base = {
            "dataset": r["dataset"],
            "subject_id": r["subject_id"],
            "session_id": r["session_id"],
        }
        pairs = r["_pairs"]
        for i in range(max_n_within_sessions):
            cand = pairs[i][0] if i < len(pairs) else ""
            rp = pairs[i][1] if i < len(pairs) else ""
            base[f"duplicate_{i+1}_candidate"] = cand
            base[f"duplicate_{i+1}_resolved_path"] = rp
        out_rows.append(base)

    out_df = pd.DataFrame(
        out_rows,
        columns=within_sessions_cols + duplicate_cols,
    )
    out_df = _sort_if_not_empty(out_df, ["dataset", "subject_id", "session_id"])
    _write_csv(out_df, out_dir / "within_sessions.csv")
    _write_dataset_stats(out_df, out_stats_dir / "within_sessions.csv")

    print(
        f"[within_sessions] input rows: {len(df0):,}, "
        f"kept groups: {len(within_sessions_rows):,}, "
        f"max duplicates per group: {max_n_within_sessions}"
    )
    print(f"[within_sessions] saved -> {out_dir / 'within_sessions.csv'}")

    # ---------- 2) across sessions within subjects ----------
    within_subjects_base_cols = ["dataset", "subject_id"]
    within_subjects_rows = []
    max_sessions = 0

    if not df0.empty:
        df = df0.copy()
        gsize_all = df.groupby("group_id").size()
        dup_gids_all = set(gsize_all[gsize_all > 1].index)

        df = df[df["group_id"].isin(dup_gids_all)].copy()
        df_nonempty_ses = df[df["session_id"] != ""].copy()

        if not df_nonempty_ses.empty:
            sess_counts = (
                df_nonempty_ses.groupby(["group_id", "dataset", "subject_id"])[
                    "session_id"
                ]
                .nunique()
                .reset_index(name="n_sessions")
            )
            targets = sess_counts[sess_counts["n_sessions"] >= 2].drop(
                columns=["n_sessions"]
            )
            df_keep = df_nonempty_ses.merge(
                targets,
                on=["group_id", "dataset", "subject_id"],
                how="inner",
            )

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
                reps = (
                    sub.groupby("session_id", sort=True, dropna=False)
                    .nth(0)
                    .reset_index()
                )
                if reps.shape[0] < 2:
                    continue

                max_sessions = max(max_sessions, reps.shape[0])
                within_subjects_rows.append(
                    {
                        "dataset": ds,
                        "subject_id": subj,
                        "_sessions": reps["session_id"].tolist(),
                        "_cands": reps["candidate"].tolist(),
                        "_rpaths": reps["resolved_path"].tolist(),
                    }
                )

    within_subjects_cols = within_subjects_base_cols.copy()
    for i in range(1, max_sessions + 1):
        within_subjects_cols += [
            f"session_id_{i}",
            f"duplicate_{i}_candidate",
            f"duplicate_{i}_resolved_path",
        ]

    out_rows = []
    for r in within_subjects_rows:
        base = {"dataset": r["dataset"], "subject_id": r["subject_id"]}
        for i in range(max_sessions):
            sid = r["_sessions"][i] if i < len(r["_sessions"]) else ""
            cand = r["_cands"][i] if i < len(r["_cands"]) else ""
            rpat = r["_rpaths"][i] if i < len(r["_rpaths"]) else ""
            base[f"session_id_{i+1}"] = sid
            base[f"duplicate_{i+1}_candidate"] = cand
            base[f"duplicate_{i+1}_resolved_path"] = rpat
        out_rows.append(base)

    out_df = pd.DataFrame(out_rows, columns=within_subjects_cols)
    out_df = _sort_if_not_empty(out_df, ["dataset", "subject_id"])
    _write_csv(out_df, out_dir / "within_subjects.csv")
    _write_dataset_stats(out_df, out_stats_dir / "within_subjects.csv")

    print(
        f"[within_subjects] input rows: {len(df0):,}, "
        f"output groups: {len(out_df):,}, "
        f"max sessions per row: {max_sessions}"
    )
    print(f"[within_subjects] saved -> {out_dir / 'within_subjects.csv'}")

    # ---------- 3) across subjects within datasets ----------
    within_datasets_base_cols = ["dataset"]
    within_datasets_rows = []
    max_items_within_datasets = 0

    if not df0.empty:
        df = df0.copy()
        gsize_all = df.groupby("group_id").size()
        dup_gids_all = set(gsize_all[gsize_all > 1].index)
        df = df[df["group_id"].isin(dup_gids_all)].copy()

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
            max_items_within_datasets = max(max_items_within_datasets, len(items))
            within_datasets_rows.append({"dataset": ds, "_items": items})

    within_datasets_cols = within_datasets_base_cols.copy()
    for i in range(1, max_items_within_datasets + 1):
        within_datasets_cols += [
            f"subject_id_{i}",
            f"session_id_{i}",
            f"duplicate_{i}_candidate",
            f"duplicate_{i}_resolved_path",
        ]

    out_rows = []
    for r in within_datasets_rows:
        base = {"dataset": r["dataset"]}
        items = r["_items"]
        for i in range(max_items_within_datasets):
            if i < len(items):
                subid, sesid, cand, rpath = items[i]
            else:
                subid = sesid = cand = rpath = ""
            base[f"subject_id_{i+1}"] = subid
            base[f"session_id_{i+1}"] = sesid
            base[f"duplicate_{i+1}_candidate"] = cand
            base[f"duplicate_{i+1}_resolved_path"] = rpath
        out_rows.append(base)

    out_df = pd.DataFrame(out_rows, columns=within_datasets_cols)
    out_df = _sort_if_not_empty(out_df, ["dataset", "subject_id_1", "session_id_1"])
    _write_csv(out_df, out_dir / "within_datasets.csv")
    _write_dataset_stats(out_df, out_stats_dir / "within_datasets.csv")

    print(
        f"[within_datasets] input rows: {len(df0):,}, "
        f"output groups: {len(out_df):,}, "
        f"max items per row: {max_items_within_datasets}"
    )
    print(f"[within_datasets] saved -> {out_dir / 'within_datasets.csv'}")

    return any(
        [
            len(within_sessions_rows) > 0,
            len(within_subjects_rows) > 0,
            len(within_datasets_rows) > 0,
        ]
    )


def report_by_category_across_datasets(
    in_csv: Union[str, Path],
    out_dir: Union[str, Path],
    out_stats_dir: Union[str, Path],
    modality: Optional[str] = None,
) -> bool:
    in_csv = Path(in_csv)
    out_dir = Path(out_dir)
    out_stats_dir = Path(out_stats_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_stats_dir.mkdir(parents=True, exist_ok=True)

    df = _read_base_csv(in_csv, modality=modality)
    df = df[df["dataset"] != ""].copy()

    stats_csv = out_stats_dir / "across_datasets.csv"
    out_csv = out_dir / "across_datasets.csv"

    empty_cols = [
        "dataset_1",
        "subject_id_1",
        "session_id_1",
        "duplicate_1_candidate",
        "duplicate_1_resolved_path",
    ]

    if df.empty:
        _write_csv(pd.DataFrame(columns=["set_key", "n_images"]), stats_csv)
        _write_csv(pd.DataFrame(columns=empty_cols), out_csv)
        return False

    gsize = df.groupby("group_id").size()
    dup_gids = set(gsize[gsize > 1].index)
    df = df[df["group_id"].isin(dup_gids)].copy()

    if df.empty:
        _write_csv(pd.DataFrame(columns=["set_key", "n_images"]), stats_csv)
        _write_csv(pd.DataFrame(columns=empty_cols), out_csv)
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

    if ds_sets_cross.empty:
        _write_csv(pd.DataFrame(columns=["set_key", "n_images"]), stats_csv)
        _write_csv(pd.DataFrame(columns=empty_cols), out_csv)
        return False

    df_cross = df.merge(ds_sets_cross, on="group_id", how="inner")
    set_hist = (
        df_cross.groupby("set_key", dropna=False)["resolved_path"]
        .size()
        .reset_index(name="n_images")
    )
    if not set_hist.empty:
        set_hist = set_hist.sort_values(
            ["n_images", "set_key"], ascending=[False, True], kind="mergesort"
        ).reset_index(drop=True)
    _write_csv(set_hist, stats_csv)

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

    rows = []
    max_items = 0
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

    out_cols = []
    for i in range(1, max_items + 1):
        out_cols += [
            f"dataset_{i}",
            f"subject_id_{i}",
            f"session_id_{i}",
            f"duplicate_{i}_candidate",
            f"duplicate_{i}_resolved_path",
        ]

    if not out_cols:
        out_cols = empty_cols

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
    out_df = _sort_if_not_empty(out_df, ["dataset_1", "subject_id_1", "session_id_1"])
    _write_csv(out_df, out_csv)

    print(
        f"[across_datasets] input rows: {len(df):,}, "
        f"output groups: {len(out_df):,}, "
        f"max items per row: {max_items}"
    )
    print(f"[across_datasets] saved -> {out_csv}")

    return len(rows) > 0


def run_categorize_reports(
    in_csv: Union[str, Path],
    by_dataset_dir: Union[str, Path],
    by_category_dir: Union[str, Path],
    by_category_stats_dir: Union[str, Path],
    modality: Optional[str] = None,
) -> bool:
    wrote_by_dataset = report_by_dataset(
        in_csv=in_csv,
        out_dir=by_dataset_dir,
        modality=modality,
    )

    wrote_by_category = report_by_category(
        in_dir=by_dataset_dir,
        out_dir=by_category_dir,
        out_stats_dir=by_category_stats_dir,
        modality=modality,
    )

    wrote_across = report_by_category_across_datasets(
        in_csv=in_csv,
        out_dir=by_category_dir,
        out_stats_dir=by_category_stats_dir,
        modality=modality,
    )

    return wrote_by_dataset or wrote_by_category or wrote_across
