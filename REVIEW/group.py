#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


class UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[Tuple[str, str, str, str], Tuple[str, str, str, str]] = {}
        self.rank: Dict[Tuple[str, str, str, str], int] = {}

    def add(self, x: Tuple[str, str, str, str]) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: Tuple[str, str, str, str]) -> Tuple[str, str, str, str]:
        px = self.parent[x]
        if px != x:
            self.parent[x] = self.find(px)
        return self.parent[x]

    def union(
        self,
        a: Tuple[str, str, str, str],
        b: Tuple[str, str, str, str],
    ) -> None:
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


def make_node(
    dataset: Any,
    subject_id: Any,
    session_id: Any,
    path: Any,
) -> Tuple[str, str, str, str]:
    return (
        clean_str(dataset),
        clean_str(subject_id),
        clean_str(session_id),
        clean_str(path),
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        "Convert duplicate pairs CSV into connected duplicate groups CSV."
    )
    ap.add_argument(
        "--in",
        dest="in_csv",
        type=Path,
        required=True,
        help="Input pairs CSV",
    )
    ap.add_argument(
        "--out",
        dest="out_csv",
        type=Path,
        required=True,
        help="Output groups CSV",
    )
    args = ap.parse_args()

    in_csv = args.in_csv.expanduser().resolve()
    out_csv = args.out_csv.expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not in_csv.exists():
        raise FileNotFoundError(f"input csv not found: {in_csv}")

    df = pd.read_csv(in_csv, dtype=str, na_filter=False)
    df.columns = [c.strip() for c in df.columns]

    required = [
        "dataset_1",
        "subject_id_1",
        "session_id_1",
        "path_1",
        "dataset_2",
        "subject_id_2",
        "session_id_2",
        "path_2",
        "QA_status",
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"input csv missing required column: {c}")

    uf = UnionFind()

    kept_rows = 0
    skipped_rows = 0

    for _, r in df.iterrows():
        qa = clean_str(r["QA_status"]).lower()
        if qa != "yes":
            skipped_rows += 1
            continue

        n1 = make_node(
            r["dataset_1"],
            r["subject_id_1"],
            r["session_id_1"],
            r["path_1"],
        )
        n2 = make_node(
            r["dataset_2"],
            r["subject_id_2"],
            r["session_id_2"],
            r["path_2"],
        )

        # Require dataset, subject_id, and path at minimum
        if not (n1[0] and n1[1] and n1[3] and n2[0] and n2[1] and n2[3]):
            skipped_rows += 1
            continue

        uf.add(n1)
        uf.add(n2)
        uf.union(n1, n2)
        kept_rows += 1

    # Collect connected components
    groups: Dict[Tuple[str, str, str, str], List[Tuple[str, str, str, str]]] = {}
    for node in uf.parent.keys():
        root = uf.find(node)
        groups.setdefault(root, []).append(node)

    # Make group ids deterministic:
    # sort members inside each component, then sort components by their first member
    components: List[List[Tuple[str, str, str, str]]] = []
    for members in groups.values():
        uniq_members = sorted(set(members))
        components.append(uniq_members)

    components.sort(key=lambda members: members[0])

    out_rows: List[Dict[str, str]] = []
    for i, members in enumerate(components, start=1):
        group_id = f"G{i:06d}"
        for ds, sbj, ses, path in members:
            out_rows.append(
                {
                    "group_id": group_id,
                    "dataset": ds,
                    "subject_id": sbj,
                    "session_id": ses,
                    "resolved_path": path,
                }
            )

    out_df = pd.DataFrame(
        out_rows,
        columns=["group_id", "dataset", "subject_id", "session_id", "resolved_path"],
    )

    out_df.to_csv(out_csv, index=False)

    print("[DONE]")
    print(f"  input_csv:      {in_csv}")
    print(f"  output_csv:     {out_csv}")
    print(f"  yes_rows_used:  {kept_rows}")
    print(f"  skipped_rows:   {skipped_rows}")
    print(f"  num_groups:     {len(components)}")
    print(f"  num_scans:      {len(out_df)}")


if __name__ == "__main__":
    main()
