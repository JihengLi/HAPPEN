"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd


BASE_REQUIRED_COLS = [
    "group_id",
    "dataset",
    "subject_id",
    "subject_key",
]


def _read_base_csv(
    in_csv: Union[str, Path],
) -> pd.DataFrame:
    in_csv = Path(in_csv)
    df = pd.read_csv(in_csv, dtype=str)

    missing = [c for c in BASE_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{in_csv} missing required columns: {missing}")

    df = df[BASE_REQUIRED_COLS].copy()
    for c in BASE_REQUIRED_COLS:
        df[c] = df[c].fillna("").astype(str).str.strip()

    df = df.drop_duplicates(subset=BASE_REQUIRED_COLS).reset_index(drop=True)
    return df


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
    _write_csv(pd.DataFrame(columns=["dataset", "n_subjects"]), out_csv)


def _empty_across_dataset_stats_csv(out_csv: Path) -> None:
    _write_csv(pd.DataFrame(columns=["set_key", "n_subjects"]), out_csv)


def _unique_nonempty_preserve_order(values: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()

    for x in values:
        x = str(x).strip()
        if not x or x in seen:
            continue
        out.append(x)
        seen.add(x)

    return out


def _extract_dataset_subject_pairs(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        return pd.DataFrame(columns=["dataset", "subject_id"])

    cols = set(detail_df.columns)

    # Case 1: dataset + subject_id_i
    if "dataset" in cols:
        subject_cols = [
            c
            for c in detail_df.columns
            if c == "subject_id" or c.startswith("subject_id_")
        ]
        if not subject_cols:
            return pd.DataFrame(columns=["dataset", "subject_id"])

        pieces = []
        for sc in subject_cols:
            tmp = detail_df[["dataset", sc]].rename(columns={sc: "subject_id"})
            pieces.append(tmp)

        pairs = pd.concat(pieces, ignore_index=True)

    # Case 2: dataset_i + subject_id_i
    else:
        idxs: List[int] = []
        i = 1
        while True:
            ds_col = f"dataset_{i}"
            sid_col = f"subject_id_{i}"
            if ds_col in cols and sid_col in cols:
                idxs.append(i)
                i += 1
                continue
            if ds_col not in cols and sid_col not in cols:
                break
            i += 1

        if not idxs:
            return pd.DataFrame(columns=["dataset", "subject_id"])

        pieces = []
        for i in idxs:
            tmp = detail_df[[f"dataset_{i}", f"subject_id_{i}"]].rename(
                columns={
                    f"dataset_{i}": "dataset",
                    f"subject_id_{i}": "subject_id",
                }
            )
            pieces.append(tmp)

        pairs = pd.concat(pieces, ignore_index=True)

    pairs["dataset"] = pairs["dataset"].fillna("").astype(str).str.strip()
    pairs["subject_id"] = pairs["subject_id"].fillna("").astype(str).str.strip()

    pairs = pairs[(pairs["dataset"] != "") & (pairs["subject_id"] != "")][
        ["dataset", "subject_id"]
    ].drop_duplicates()

    return pairs.reset_index(drop=True)


def _write_dataset_stats(detail_df: pd.DataFrame, stats_path: Path) -> None:
    pairs = _extract_dataset_subject_pairs(detail_df)

    if pairs.empty:
        _empty_dataset_stats_csv(stats_path)
        return

    stats = (
        pairs.groupby("dataset")["subject_id"].nunique().reset_index(name="n_subjects")
    )

    stats = _sort_if_not_empty(stats, ["n_subjects", "dataset"])
    if not stats.empty:
        stats = stats.sort_values(
            ["n_subjects", "dataset"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)

    _write_csv(stats, stats_path)


def _write_across_dataset_set_stats(
    df: pd.DataFrame,
    stats_path: Path,
) -> None:
    if df.empty:
        _empty_across_dataset_stats_csv(stats_path)
        return

    df = df[
        (df["dataset"] != "") & (df["subject_id"] != "") & (df["subject_key"] != "")
    ].copy()

    if df.empty:
        _empty_across_dataset_stats_csv(stats_path)
        return

    df = df.drop_duplicates(subset=["group_id", "dataset", "subject_key"]).copy()

    def _clean_set(s: pd.Series) -> list[str]:
        vals = {x.strip() for x in s.dropna().astype(str) if x and x.strip()}
        return sorted(vals)

    ds_sets = (
        df.groupby("group_id", dropna=False)["dataset"]
        .apply(_clean_set)
        .reset_index(name="datasets")
    )
    ds_sets["set_size"] = ds_sets["datasets"].map(len)
    ds_sets["set_key"] = ds_sets["datasets"].map(lambda xs: "|".join(xs) if xs else "")

    ds_sets_cross = ds_sets.loc[
        ds_sets["set_size"] >= 2,
        ["group_id", "set_key"],
    ]

    if ds_sets_cross.empty:
        _empty_across_dataset_stats_csv(stats_path)
        return

    df_cross = df.merge(ds_sets_cross, on="group_id", how="inner")

    stats = (
        df_cross.groupby("set_key", dropna=False).size().reset_index(name="n_subjects")
    )

    if not stats.empty:
        stats = stats.sort_values(
            ["n_subjects", "set_key"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)

    _write_csv(stats, stats_path)


def report_by_dataset(
    in_csv: Union[str, Path],
    out_dir: Union[str, Path],
    sort_cols_detail: Tuple[str, ...] = ("group_id", "subject_id"),
) -> bool:
    in_csv = Path(in_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _read_base_csv(in_csv)
    df = df[
        (df["dataset"] != "") & (df["subject_id"] != "") & (df["subject_key"] != "")
    ].copy()

    if df.empty:
        return False

    df = df.drop_duplicates(subset=["group_id", "dataset", "subject_key"]).reset_index(
        drop=True
    )

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

        sizes = df_d.groupby("group_id")["subject_key"].nunique()
        keep_gids = set(sizes.index[sizes.values >= 2])
        df_dups = df_d[df_d["group_id"].isin(keep_gids)].copy()

        if df_dups.empty:
            continue

        df_dups = (
            df_dups[["group_id", "dataset", "subject_id"]]
            .drop_duplicates(subset=["group_id", "dataset", "subject_id"])
            .reset_index(drop=True)
        )

        gsize = df_dups.groupby("group_id").size()
        if (gsize < 2).any():
            raise RuntimeError(
                f"{ds}: found by_dataset group with fewer than 2 rows: "
                f"{gsize[gsize < 2].to_dict()}"
            )

        df_dups = _sort_if_not_empty(df_dups, list(sort_cols_detail))
        out_csv_ds = out_dir / f"{ds}.csv"
        _write_csv(df_dups, out_csv_ds)
        wrote_any = True

    return wrote_any


def report_by_category(
    in_csv: Union[str, Path],
    out_dir: Union[str, Path],
    out_stats_dir: Union[str, Path],
) -> bool:
    in_csv = Path(in_csv)
    out_dir = Path(out_dir)
    out_stats_dir = Path(out_stats_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_stats_dir.mkdir(parents=True, exist_ok=True)

    df0 = _read_base_csv(in_csv)
    df0 = df0[
        (df0["dataset"] != "") & (df0["subject_id"] != "") & (df0["subject_key"] != "")
    ].copy()

    if not df0.empty:
        df0 = df0.drop_duplicates(
            subset=["group_id", "dataset", "subject_key"]
        ).reset_index(drop=True)

    within_datasets_base_cols = ["dataset"]
    within_datasets_rows = []
    max_items_within_datasets = 0

    if not df0.empty:
        df = df0.copy()
        df = df.sort_values(
            ["group_id", "dataset", "subject_id"],
            kind="mergesort",
        )

        for (gid, ds), sub in df.groupby(["group_id", "dataset"], sort=False):
            subjects = _unique_nonempty_preserve_order(sub["subject_id"].tolist())
            if len(subjects) < 2:
                continue

            max_items_within_datasets = max(
                max_items_within_datasets,
                len(subjects),
            )
            within_datasets_rows.append(
                {
                    "dataset": ds,
                    "_subjects": subjects,
                }
            )

    within_datasets_cols = within_datasets_base_cols.copy()
    for i in range(1, max_items_within_datasets + 1):
        within_datasets_cols += [f"subject_id_{i}"]

    out_rows = []
    for r in within_datasets_rows:
        base = {"dataset": r["dataset"]}
        subjects = r["_subjects"]
        for i in range(max_items_within_datasets):
            subj = subjects[i] if i < len(subjects) else ""
            base[f"subject_id_{i+1}"] = subj
        out_rows.append(base)

    out_df = pd.DataFrame(out_rows, columns=within_datasets_cols)
    out_df = _sort_if_not_empty(out_df, ["dataset", "subject_id_1"])
    _write_csv(out_df, out_dir / "within_datasets.csv")
    _write_dataset_stats(out_df, out_stats_dir / "within_datasets.csv")

    print(
        f"[within_datasets] input rows: {len(df0):,}, "
        f"output groups: {len(out_df):,}, "
        f"max subjects per row: {max_items_within_datasets}"
    )
    print(f"[within_datasets] saved -> {out_dir / 'within_datasets.csv'}")

    return len(within_datasets_rows) > 0


def report_by_category_across_datasets(
    in_csv: Union[str, Path],
    out_dir: Union[str, Path],
    out_stats_dir: Union[str, Path],
) -> bool:
    in_csv = Path(in_csv)
    out_dir = Path(out_dir)
    out_stats_dir = Path(out_stats_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_stats_dir.mkdir(parents=True, exist_ok=True)

    df = _read_base_csv(in_csv)
    df = df[
        (df["dataset"] != "") & (df["subject_id"] != "") & (df["subject_key"] != "")
    ].copy()

    stats_csv = out_stats_dir / "across_datasets.csv"
    out_csv = out_dir / "across_datasets.csv"

    empty_cols = [
        "dataset_1",
        "subject_id_1",
    ]

    if df.empty:
        _empty_across_dataset_stats_csv(stats_csv)
        _write_csv(pd.DataFrame(columns=empty_cols), out_csv)
        return False

    df = df.drop_duplicates(subset=["group_id", "dataset", "subject_key"]).copy()

    ds_counts = df.groupby("group_id")["dataset"].nunique()
    cross_gids = set(ds_counts[ds_counts >= 2].index)

    df = df[df["group_id"].isin(cross_gids)].copy()

    if df.empty:
        _empty_across_dataset_stats_csv(stats_csv)
        _write_csv(pd.DataFrame(columns=empty_cols), out_csv)
        return False

    _write_across_dataset_set_stats(df, stats_csv)

    df = df.sort_values(
        ["group_id", "dataset", "subject_id"],
        kind="mergesort",
    )

    rows = []
    max_items = 0
    for gid, sub in df.groupby("group_id", sort=False):
        items = list(
            zip(
                sub["dataset"].tolist(),
                sub["subject_id"].tolist(),
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
        ]

    if not out_cols:
        out_cols = empty_cols

    out_rows = []
    for r in rows:
        base = {}
        items = r["_items"]
        for i in range(max_items):
            if i < len(items):
                ds, subj = items[i]
            else:
                ds = subj = ""
            base[f"dataset_{i+1}"] = ds
            base[f"subject_id_{i+1}"] = subj
        out_rows.append(base)

    out_df = pd.DataFrame(out_rows, columns=out_cols)
    out_df = _sort_if_not_empty(out_df, ["dataset_1", "subject_id_1"])
    _write_csv(out_df, out_csv)

    print(
        f"[across_datasets] input rows: {len(df):,}, "
        f"output groups: {len(out_df):,}, "
        f"max subjects per row: {max_items}"
    )
    print(f"[across_datasets] saved -> {out_csv}")

    return len(rows) > 0


def run_categorize_reports(
    in_csv: Union[str, Path],
    by_dataset_dir: Union[str, Path],
    by_category_dir: Union[str, Path],
    by_category_stats_dir: Union[str, Path],
) -> bool:
    wrote_by_dataset = report_by_dataset(
        in_csv=in_csv,
        out_dir=by_dataset_dir,
    )

    wrote_by_category = report_by_category(
        in_csv=in_csv,
        out_dir=by_category_dir,
        out_stats_dir=by_category_stats_dir,
    )

    wrote_across = report_by_category_across_datasets(
        in_csv=in_csv,
        out_dir=by_category_dir,
        out_stats_dir=by_category_stats_dir,
    )

    return wrote_by_dataset or wrote_by_category or wrote_across
