"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

#!/usr/bin/env python3
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

import nibabel as nib

from tqdm import tqdm
from pathlib import Path
from typing import Dict, Tuple, Union
from collections import deque

from concurrent.futures import ProcessPoolExecutor


def _write_empty_duplicates_csv(out_csv: Path, key_col: str) -> None:
    cols_out = [
        "group_id",
        key_col,
        "modality",
        "dataset",
        "subject_id",
        "session_id",
        "candidate",
        "resolved_path",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=cols_out).to_csv(out_csv, index=False)


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def hash_raw_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_voxelbyte(path: Path) -> dict:
    img = nib.load(str(path))
    can = nib.as_closest_canonical(img)
    hdr = can.header
    shape = tuple(int(x) for x in can.shape)
    try:
        zooms_full = tuple(float(x) for x in hdr.get_zooms()[: len(shape)])
    except Exception:
        zooms_full = tuple([np.nan] * len(shape))
    zooms_r = tuple(
        x if not (isinstance(x, float) and np.isfinite(x)) else round(x, 6)
        for x in zooms_full
    )

    h = hashlib.sha256()
    dataobj = can.dataobj
    try:
        if len(shape) == 4:
            for t in range(shape[3]):
                arr = np.asanyarray(dataobj[..., t])
                if not arr.flags.c_contiguous:
                    arr = np.ascontiguousarray(arr)
                h.update(memoryview(arr))
                del arr
        else:
            arr = np.asanyarray(dataobj)
            if not arr.flags.c_contiguous:
                arr = np.ascontiguousarray(arr)
            h.update(memoryview(arr))
            del arr
    finally:
        try:
            fm = getattr(img, "file_map", {})
            for k in ("image", "header"):
                f = fm.get(k)
                if f is not None and getattr(f, "fileobj", None):
                    f.fileobj.close()
        except Exception:
            pass
        del img, can, dataobj

    dtype_str = str(
        np.asanyarray(
            nib.load(str(path)).dataobj
            if len(shape) == 3
            else nib.load(str(path)).dataobj[..., 0]
        ).dtype
    )
    return {
        "hash": h.hexdigest(),
        "shape": shape,
        "dtype": dtype_str,
        "zooms": zooms_r,
        "geo_key": f"{shape[:3]}|{dtype_str}|{tuple(zooms_r[:3])}",
    }


def _process_one(path: str) -> dict:
    p = Path(path)
    rec: dict = {"resolved_path": str(p)}
    info = hash_voxelbyte(p)
    rec["can_hash"] = info["hash"]
    return rec


def run_hash_pipeline(
    files: list[Path],
    process_workers: int,
    out_path: Path,
    batch_size: int = 1000,
    inflight_factor: int = 2,
    write_header: bool = False,
) -> tuple[Path, list[dict]]:
    out_path = Path(out_path)
    first_write = bool(write_header)
    errors: list[dict] = []

    def _flush_batch(buf: list[dict]):
        nonlocal first_write
        if not buf:
            return
        pd.DataFrame(buf).to_csv(out_path, mode="a", index=False, header=first_write)
        first_write = False
        buf.clear()

    max_inflight = max(1, process_workers * inflight_factor)
    batch_buf: list[dict] = []
    inflight = deque()

    with (
        ProcessPoolExecutor(max_workers=process_workers) as ex,
        tqdm(total=len(files), desc="Hashing", unit="file") as pbar,
    ):
        it = iter(files)

        while len(inflight) < max_inflight:
            try:
                p = next(it)
            except StopIteration:
                break
            inflight.append((ex.submit(_process_one, str(p)), str(p)))

        while inflight:
            fut, p_str = inflight.popleft()
            try:
                rec = fut.result()
                batch_buf.append(rec)
            except Exception as e:
                errors.append(
                    {"resolved_path": p_str, "error": f"{type(e).__name__}: {e}"}
                )
            finally:
                pbar.update(1)
            if len(inflight) < max_inflight:
                try:
                    p = next(it)
                    inflight.append((ex.submit(_process_one, str(p)), str(p)))
                except StopIteration:
                    pass
            if len(batch_buf) >= batch_size:
                _flush_batch(batch_buf)
    _flush_batch(batch_buf)
    return out_path, errors


def run_hash_pipeline_chunked(
    files: list[Path],
    process_workers: int,
    out_path: Path,
    chunk_size: int = 8_000,
    batch_size: int = 1000,
    inflight_factor: int = 2,
    resume: bool = True,
    truncate: bool = False,
):
    out_path = Path(out_path)
    if truncate and out_path.exists():
        out_path.unlink()
    file_empty = (not out_path.exists()) or (out_path.stat().st_size == 0)
    already = set()
    if resume and out_path.exists():
        try:
            for chunk in pd.read_csv(
                out_path, usecols=["resolved_path"], chunksize=200_000, dtype=str
            ):
                already.update(x for x in chunk["resolved_path"].astype(str) if x)
        except Exception:
            pass

    all_errors: list[dict] = []
    total = len(files)
    done = 0

    def _chunked(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i : i + n]

    for ci, file_chunk in enumerate(_chunked(files, chunk_size), start=1):
        if resume and already:
            file_chunk = [p for p in file_chunk if str(p) not in already]
        if not file_chunk:
            print(f"[CHUNK {ci}] skip (all hashed).")
            continue

        print(
            f"[CHUNK {ci}] hashing {len(file_chunk):,} files "
            f"(progress {done:,}/{total:,}) -> {out_path.name}"
        )

        _, errs = run_hash_pipeline(
            files=file_chunk,
            process_workers=process_workers,
            out_path=out_path,
            batch_size=batch_size,
            inflight_factor=inflight_factor,
            write_header=file_empty,
        )
        all_errors.extend(errs)
        done += len(file_chunk)
        already.update(str(p) for p in file_chunk)
        file_empty = False

    return out_path, all_errors


def group_and_organize_duplicates(
    hash_csv: Union[str, Path],
    key_col: str,
    modality: str,
    map_csv: Union[str, Path],
    out_csv: Union[str, Path],
    chunksize: int = 200_000,
    final_sort_keys: Tuple[str, ...] = (
        "group_id",
        "dataset",
        "subject_id",
        "session_id",
    ),
) -> bool:
    hash_csv = Path(hash_csv)
    map_csv = Path(map_csv)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if out_csv.exists():
        out_csv.unlink()

    counts: Dict[str, int] = {}
    for chunk in pd.read_csv(hash_csv, usecols=[key_col], chunksize=chunksize):
        vc = chunk[key_col].value_counts(dropna=True)
        for k, c in vc.items():
            if pd.notna(k):
                counts[k] = counts.get(k, 0) + int(c)

    dup_keys = sorted([k for k, c in counts.items() if c > 1])
    if not dup_keys:
        _write_empty_duplicates_csv(out_csv, key_col)
        return False

    gid_map = {k: i + 1 for i, k in enumerate(dup_keys)}

    def _gid(k: str) -> str:
        return f"{key_col}:{gid_map[k]}"

    EXCLUDED_DIRS = {"derivatives"}

    def _is_in_excluded_dirs(p: Union[str, Path]) -> bool:
        try:
            parts = Path(str(p)).parts
        except Exception:
            return False
        return any(part.casefold() in EXCLUDED_DIRS for part in parts)

    m = pd.read_csv(map_csv, dtype=str)
    required_meta = [
        "resolved_path",
        "candidate",
        "dataset",
        "subject_id",
        "session_id",
    ]
    missing = [c for c in required_meta if c not in m.columns]
    if missing:
        raise ValueError(f"{map_csv} is missing required columns: {missing}")
    m = m[required_meta].copy()
    m = m.fillna("")

    first = True
    cols_out = [
        "group_id",
        key_col,
        "modality",
        "dataset",
        "subject_id",
        "session_id",
        "candidate",
        "resolved_path",
    ]

    wrote_any = False

    for chunk in pd.read_csv(
        hash_csv, usecols=["resolved_path", key_col], chunksize=chunksize, dtype=str
    ):
        sub = chunk[chunk[key_col].isin(dup_keys)].copy()
        if sub.empty:
            continue

        sub["group_id"] = sub[key_col].map(_gid)
        sub = sub.merge(
            m,
            on="resolved_path",
            how="left",
            suffixes=("", "_meta"),
        )

        for c in ["candidate", "dataset", "subject_id", "session_id"]:
            if c in sub.columns:
                sub[c] = sub[c].fillna("").astype(str)

        sub = sub[~sub["candidate"].apply(_is_in_excluded_dirs)]
        if sub.empty:
            continue

        sub["modality"] = modality
        sub = sub[cols_out]
        sub.to_csv(out_csv, mode="a", header=first, index=False)
        first = False
        wrote_any = True

    if not wrote_any:
        _write_empty_duplicates_csv(out_csv, key_col)
        return False

    if final_sort_keys:
        df = pd.read_csv(out_csv, dtype=str)
        df.sort_values(list(final_sort_keys), ascending=True, kind="mergesort").to_csv(
            out_csv, index=False
        )

    return True
