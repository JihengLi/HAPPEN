#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd


def clean_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null", "na", "n/a"}:
        return ""
    return s


SubjectKey = Tuple[str, str]  # (dataset, subject_id)


class UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[SubjectKey, SubjectKey] = {}
        self.rank: Dict[SubjectKey, int] = {}

    def add(self, x: SubjectKey) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: SubjectKey) -> SubjectKey:
        px = self.parent[x]
        if px != x:
            self.parent[x] = self.find(px)
        return self.parent[x]

    def union(self, a: SubjectKey, b: SubjectKey) -> None:
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


def subject_key(dataset: Any, subject_id: Any) -> SubjectKey:
    return (clean_str(dataset), clean_str(subject_id))


def make_subject_components_from_scan_groups(
    dup_df: pd.DataFrame,
) -> List[List[SubjectKey]]:
    required = ["group_id", "dataset", "subject_id", "session_id", "resolved_path"]
    for c in required:
        if c not in dup_df.columns:
            raise ValueError(f"duplicate_groups.csv missing required column: {c}")

    uf = UnionFind()

    for _, g in dup_df.groupby("group_id", sort=False):
        members = sorted(
            {
                subject_key(r["dataset"], r["subject_id"])
                for _, r in g.iterrows()
                if clean_str(r["dataset"]) and clean_str(r["subject_id"])
            }
        )
        if not members:
            continue

        for s in members:
            uf.add(s)

        pivot = members[0]
        for s in members[1:]:
            uf.union(pivot, s)

    comps: Dict[SubjectKey, List[SubjectKey]] = {}
    for s in uf.parent.keys():
        root = uf.find(s)
        comps.setdefault(root, []).append(s)

    out: List[List[SubjectKey]] = []
    for members in comps.values():
        out.append(sorted(set(members)))

    out.sort(key=lambda x: x[0])
    return out


def choose_keep_subject(
    members: List[SubjectKey],
    manifest_count: Dict[SubjectKey, int],
    prefer_datasets: Set[str],
) -> SubjectKey:
    infos = []
    for ds, sbj in members:
        infos.append(
            {
                "key": (ds, sbj),
                "dataset": ds,
                "subject_id": sbj,
                "rows": int(manifest_count.get((ds, sbj), 0)),
                "preferred": ds in prefer_datasets,
            }
        )

    preferred_infos = [x for x in infos if x["preferred"]]

    if preferred_infos:
        pool = preferred_infos
    else:
        pool = infos

    pool_sorted = sorted(
        pool,
        key=lambda x: (-int(x["rows"]), x["dataset"], x["subject_id"]),
    )
    return pool_sorted[0]["key"]


def main() -> None:
    ap = argparse.ArgumentParser(
        "Remove duplicate subjects from manifest based on duplicate_groups.csv."
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Input manifest CSV (must include dataset, subject_id, session_id, resolved_path)",
    )
    ap.add_argument(
        "--duplicate-groups",
        type=Path,
        required=True,
        help="Input duplicate_groups.csv (scan-level groups)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory",
    )
    ap.add_argument(
        "--prefer-datasets",
        nargs="*",
        default=[],
        help="Datasets to prefer keeping, e.g. --prefer-datasets ABCD ADNI",
    )
    args = ap.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    dup_groups_path = args.duplicate_groups.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not dup_groups_path.exists():
        raise FileNotFoundError(f"duplicate_groups not found: {dup_groups_path}")

    prefer_datasets = {clean_str(x) for x in args.prefer_datasets if clean_str(x)}

    manifest = pd.read_csv(manifest_path, dtype=str, na_filter=False)
    manifest.columns = [c.strip() for c in manifest.columns]

    dup_df = pd.read_csv(dup_groups_path, dtype=str, na_filter=False)
    dup_df.columns = [c.strip() for c in dup_df.columns]

    required_manifest = ["dataset", "subject_id", "session_id", "resolved_path"]
    for c in required_manifest:
        if c not in manifest.columns:
            raise ValueError(f"manifest missing required column: {c}")

    for c in required_manifest:
        manifest[c] = manifest[c].map(clean_str)

    for c in ["group_id", "dataset", "subject_id", "session_id", "resolved_path"]:
        dup_df[c] = dup_df[c].map(clean_str)

    manifest_count_series = manifest.groupby(
        ["dataset", "subject_id"], sort=False
    ).size()
    manifest_count: Dict[SubjectKey, int] = {
        (clean_str(ds), clean_str(sbj)): int(cnt)
        for (ds, sbj), cnt in manifest_count_series.items()
    }

    manifest_subjects: Set[SubjectKey] = set(manifest_count.keys())

    subject_components_all = make_subject_components_from_scan_groups(dup_df)

    subject_components_manifest: List[List[SubjectKey]] = []
    for members in subject_components_all:
        kept = sorted([m for m in members if m in manifest_subjects])
        if kept:
            subject_components_manifest.append(kept)

    subject_components_manifest = sorted(
        subject_components_manifest, key=lambda x: x[0]
    )

    subjects_to_drop: Set[SubjectKey] = set()

    num_components_total = 0
    num_components_with_conflict = 0

    for members in subject_components_manifest:
        num_components_total += 1

        if len(members) <= 1:
            continue

        num_components_with_conflict += 1
        keep_subject = choose_keep_subject(
            members=members,
            manifest_count=manifest_count,
            prefer_datasets=prefer_datasets,
        )

        for s in members:
            if s != keep_subject:
                subjects_to_drop.add(s)

    manifest_keys = list(zip(manifest["dataset"], manifest["subject_id"]))
    drop_mask = [k in subjects_to_drop for k in manifest_keys]
    keep_mask = [not x for x in drop_mask]

    manifest_removed = manifest.loc[drop_mask].copy()
    manifest_removed.reset_index(drop=True, inplace=True)

    manifest_dedup = manifest.loc[keep_mask].copy()
    manifest_dedup.reset_index(drop=True, inplace=True)

    out_manifest = out_dir / "pool_cleaned.csv"
    out_removed = out_dir / "removed_near_dup.csv"

    manifest_dedup.to_csv(out_manifest, index=False)
    manifest_removed.to_csv(out_removed, index=False)

    print("[DONE]")
    print(f"  manifest_in:                 {manifest_path}")
    print(f"  duplicate_groups_in:         {dup_groups_path}")
    print(f"  prefer_datasets:             {sorted(prefer_datasets)}")
    print(f"  manifest_out:                {out_manifest}")
    print(f"  manifest_removed_out:        {out_removed}")
    print(f"  subject_components_total:    {num_components_total}")
    print(f"  components_with_conflict:    {num_components_with_conflict}")
    print(f"  dropped_subjects:            {len(subjects_to_drop)}")
    print(f"  manifest_rows_before:        {len(manifest)}")
    print(f"  manifest_rows_removed:       {len(manifest_removed)}")
    print(f"  manifest_rows_after:         {len(manifest_dedup)}")


if __name__ == "__main__":
    main()
