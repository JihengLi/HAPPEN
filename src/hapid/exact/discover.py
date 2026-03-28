"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

#!/usr/bin/env python3
from __future__ import annotations

import subprocess, shutil
import pandas as pd

import nibabel as nib

from tqdm import tqdm
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed


def glob_bids_candidates(
    root: Path,
    modality: str,
    links_only: bool = False,
    exclude_dirnames: Iterable[str] = ("derivatives", ".git", "code", "sourcedata"),
) -> list[Path]:
    root = Path(root)

    exe = shutil.which("fdfind") or shutil.which("fd")
    if exe:
        print(f"[INFO] Using '{exe}' for fast file discovery.")
        pattern = rf"sub-.*/.*_{modality}\.nii(\.gz)?$"
        cmd = [exe, "-p", pattern, str(root)]
        for d in exclude_dirnames:
            cmd += ["-E", d]
        if links_only:
            cmd += ["-t", "l"]
        else:
            cmd += ["-t", "f", "-t", "l"]
        print(" ".join(cmd))
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        rels = [ln.strip() for ln in out.splitlines() if ln.strip()]
        paths = [root / r for r in rels]
        return sorted(set(paths))

    print(
        "[INFO] Using Python glob for file discovery "
        "(consider installing 'fd' or 'fdfind' for better performance)."
    )
    pat = f"**/sub-*/**/*_{modality}.nii*"
    cands = list(root.glob(pat))

    ex_dirs_cf = {d.casefold() for d in exclude_dirnames}

    def is_in_excluded_dir(p: Path) -> bool:
        return any(part.casefold() in ex_dirs_cf for part in p.parts)

    if links_only:
        cands = [p for p in cands if p.is_symlink()]
    cands = [p for p in cands if not is_in_excluded_dir(p)]
    return sorted(set(cands))


def parse_triplet(cand: str) -> tuple[str, str, str]:
    if not isinstance(cand, str) or "sub-" not in cand:
        return "", "", ""
    p = Path(cand)
    parts = [seg for seg in p.parts if seg]
    if not parts:
        return "", "", ""
    sub_idx = -1
    for i, seg in enumerate(parts):
        if seg.startswith("sub-"):
            sub_idx = i
            break
    if sub_idx == -1:
        return "", "", ""
    subject = parts[sub_idx]
    dataset = parts[sub_idx - 1] if sub_idx > 0 else ""
    session = ""
    if sub_idx + 1 < len(parts):
        next_seg = parts[sub_idx + 1]
        if next_seg.startswith("ses-"):
            session = next_seg
    return dataset, subject, session


def classify_candidates(
    candidates: list[Path],
    verify_readable: bool = False,
    max_workers: Optional[int] = None,
    use_threads: bool = True,
    exclude_dirnames: Iterable[str] = ("derivatives",),
) -> tuple[
    dict[str, str],
    dict[str, str],
    dict[str, str],
    dict[str, str],
    dict[str, str],
    dict[str, str],
]:
    ok: dict[str, str] = {}
    missing: dict[str, str] = {}
    permission_denied: dict[str, str] = {}
    not_nifti: dict[str, str] = {}
    unreadable: dict[str, str] = {}
    excluded_derivatives: dict[str, str] = {}

    EXCLUDED_DIRS = {s.casefold() for s in exclude_dirnames}

    def _is_in_excluded_dirs(
        p: Union[str, Path],
    ) -> bool:
        try:
            parts = Path(str(p)).parts
        except Exception:
            return False
        return any(part.casefold() in EXCLUDED_DIRS for part in parts)

    def _process_one(cand: Path):
        cstr = str(cand)
        if _is_in_excluded_dirs(cand):
            return ("excluded", cstr, "")
        try:
            resolved = cand.resolve(strict=True)
        except FileNotFoundError:
            return ("missing", cstr, "")
        except PermissionError:
            return ("denied", cstr, "")
        if not (
            (resolved.suffix == ".nii") or (resolved.suffixes[-2:] == [".nii", ".gz"])
        ):
            return ("not_nifti", cstr, str(resolved))
        if verify_readable:
            try:
                _ = nib.load(str(resolved))
            except Exception:
                return ("unreadable", cstr, str(resolved))
        return ("ok", cstr, str(resolved))

    Exec = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    with Exec(max_workers=max_workers) as ex:
        futures = [ex.submit(_process_one, p) for p in candidates]
        iterator = tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Classifying candidates",
            unit="file",
        )
        for fut in iterator:
            kind, a, b = fut.result()
            if kind == "ok":
                ok[b] = a
            elif kind == "missing":
                missing[a] = b
            elif kind == "denied":
                permission_denied[a] = b
            elif kind == "not_nifti":
                not_nifti[a] = b
            elif kind == "unreadable":
                unreadable[a] = b
            elif kind == "excluded":
                excluded_derivatives[a] = b
    return ok, missing, permission_denied, not_nifti, unreadable, excluded_derivatives


def write_mapping_to_csv(
    mapping: dict[str, str],
    out_path: Union[str, Path],
    meta_df: pd.DataFrame,
    left_name: str = "candidate",
    right_name: str = "resolved_path",
    place_keys: str = "left",
    sep: str = ",",
    sort_by: Tuple[str, ...] = (),
    drop_duplicates: bool = True,
    meta_join_on: str = "candidate",
    join_side: str = "right",
) -> None:
    out_path = Path(out_path)
    if place_keys == "left":
        rows = [{left_name: k, right_name: v} for k, v in mapping.items()]
    elif place_keys == "right":
        rows = [{left_name: v, right_name: k} for k, v in mapping.items()]
    else:
        raise ValueError('place_keys must be "left" or "right".')

    df = pd.DataFrame.from_records(rows)

    if df.empty:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False, sep=sep)
        return

    if drop_duplicates:
        df = df.drop_duplicates(subset=[left_name, right_name], keep="first")

    if meta_join_on not in meta_df.columns:
        raise ValueError(f"meta_df must contain column '{meta_join_on}' for joining.")

    if join_side == "right":
        join_col = right_name
    elif join_side == "left":
        join_col = left_name
    else:
        raise ValueError('join_side must be "left" or "right".')
    df = df.merge(
        meta_df,
        left_on=join_col,
        right_on=meta_join_on,
        how="left",
        suffixes=("", "_meta"),
    )
    if sort_by:
        df = df.sort_values(list(sort_by), kind="mergesort")
    core_cols = ["dataset", "subject_id", "session_id", "candidate", "resolved_path"]
    existing_core = [c for c in core_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in existing_core]
    df = df[existing_core + other_cols]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, sep=sep)


def read_paths_txt(txt_path: Union[str, Path]) -> List[Path]:
    txt = Path(txt_path).expanduser()
    if not txt.is_absolute():
        txt = Path.cwd() / txt
    txt = txt.resolve(strict=True)

    paths: List[Path] = []
    with open(txt, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            paths.append(Path(s).expanduser())
    return paths
