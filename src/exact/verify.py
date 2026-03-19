"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

#!/usr/bin/env python3
from __future__ import annotations

import os, re

import numpy as np
import pandas as pd

import nibabel as nib
from nibabel.orientations import io_orientation, aff2axcodes

from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from collections import defaultdict

from functools import lru_cache
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed


def files_identical_raw(
    p1: Union[str, Path], p2: Union[str, Path], chunk_mb: int = 4
) -> Tuple[bool, str]:
    p1, p2 = str(p1), str(p2)
    try:
        s1, s2 = os.path.getsize(p1), os.path.getsize(p2)
    except FileNotFoundError as e:
        return False, f"MISSING:{getattr(e, 'filename', '?')}"
    except PermissionError as e:
        return False, f"PERMISSION:{getattr(e, 'filename', '?')}"
    except Exception as e:
        return False, f"ERROR:{type(e).__name__}:{e}"
    if s1 != s2:
        return False, "DIFF_SIZE"
    bs = chunk_mb * 1024 * 1024
    try:
        with open(p1, "rb") as f1, open(p2, "rb") as f2:
            while True:
                b1 = f1.read(bs)
                b2 = f2.read(bs)
                if not b1 and not b2:
                    return True, "OK"
                if b1 != b2:
                    return False, "DIFF_CONTENT"
    except FileNotFoundError as e:
        return False, f"MISSING:{getattr(e, 'filename', '?')}"
    except PermissionError as e:
        return False, f"PERMISSION:{getattr(e, 'filename', '?')}"
    except Exception as e:
        return False, f"ERROR:{type(e).__name__}:{e}"


def all_identical_raw(
    paths: Iterable[Union[str, Path],], chunk_mb: int = 4
) -> Tuple[str, Optional[Tuple[str, str, str]]]:
    ps = [str(p) for p in paths if p]
    if len(ps) < 2:
        return "UNKNOWN", None
    anchor = ps[0]
    for other in ps[1:]:
        ok, reason = files_identical_raw(anchor, other, chunk_mb=chunk_mb)
        if not ok:
            return "FALSE", (anchor, other, reason)
    return "TRUE", None


def _canonical_meta_and_data(path: str):
    img = nib.load(path)
    can = nib.as_closest_canonical(img)

    data = np.asanyarray(can.dataobj)
    if not data.flags.c_contiguous:
        data = np.ascontiguousarray(data)

    shape = tuple(int(x) for x in data.shape)
    dtype_str = str(data.dtype)

    hdr = can.header
    try:
        zooms = tuple(float(x) for x in hdr.get_zooms()[: len(shape)])
    except Exception:
        zooms = tuple([np.nan] * len(shape))
    zooms_r = tuple(
        round(x, 6) if (isinstance(x, float) and np.isfinite(x)) else x for x in zooms
    )

    axcodes_canon = "".join(aff2axcodes(can.affine) or ())
    ornt = io_orientation(can.affine)

    slope, inter = img.header.get_slope_inter() or (None, None)
    qform = tuple(img.get_qform()) if img.get_qform() is not None else None
    sform = tuple(img.get_sform()) if img.get_sform() is not None else None
    qcode = int(img.header.get("qform_code", 0))
    scode = int(img.header.get("sform_code", 0))
    axcodes_orig = "".join(aff2axcodes(img.affine) or ())

    try:
        fm = getattr(img, "file_map", {})
        for k in ("image", "header"):
            f = fm.get(k)
            if f is not None and getattr(f, "fileobj", None):
                f.fileobj.close()
    except Exception:
        pass

    meta = {
        "shape": shape,
        "dtype": dtype_str,
        "zooms": zooms_r,
        "axcodes_canonical": axcodes_canon,
        "ornt": ornt.tolist() if hasattr(ornt, "tolist") else None,
        "slope": slope,
        "inter": inter,
        "qform_code": qcode,
        "sform_code": scode,
        "qform": qform,
        "sform": sform,
        "axcodes_original": axcodes_orig,
    }
    return meta, data


def headers_identical_strict(
    p1: Union[str, Path],
    p2: Union[str, Path],
) -> tuple[bool, str]:
    try:
        img1, img2 = nib.load(str(p1)), nib.load(str(p2))
    except FileNotFoundError as e:
        return False, f"MISSING:{getattr(e, 'filename', '?')}"
    except PermissionError as e:
        return False, f"PERMISSION:{getattr(e, 'filename', '?')}"
    except Exception as e:
        return False, f"LOAD_ERROR:{type(e).__name__}:{e}"

    try:
        b1 = bytes(img1.header.binaryblock)
        b2 = bytes(img2.header.binaryblock)
        if b1 != b2:
            return False, "HDR_BINARY_DIFF"

        def _ext_fp(hdr):
            try:
                exts = getattr(hdr, "extensions", [])
                fp = []
                for e in exts:
                    code = int(getattr(e, "ecode", getattr(e, "code", -1)))
                    size = int(getattr(e, "esize", 0))
                    fp.append((code, size))
                return tuple(fp)
            except Exception:
                return tuple()

        if _ext_fp(img1.header) != _ext_fp(img2.header):
            return False, "HDR_EXTENSIONS_DIFF"

        return True, "HDR_EQUAL"
    except Exception as e:
        return False, f"HDR_COMPARE_ERROR:{type(e).__name__}:{e}"


def compare_nifti(p1: str, p2: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"path1": p1, "path2": p2}
    raw_equal, raw_reason = files_identical_raw(p1, p2)
    out["raw_equal"] = raw_equal
    out["raw_reason"] = raw_reason
    if raw_equal:
        out["category"] = "SAME_BYTES"
        out["canonical_equal"] = True
        out["canonical_reason"] = "OK"
        return out
    try:
        meta1, arr1 = _canonical_meta_and_data(p1)
        meta2, arr2 = _canonical_meta_and_data(p2)
        hdr_equal, hdr_reason = headers_identical_strict(p1, p2)
        out["header_strict_equal"] = bool(hdr_equal)
        out["header_strict_reason"] = hdr_reason
    except FileNotFoundError as e:
        out.update(
            {
                "canonical_equal": False,
                "canonical_reason": f"MISSING:{getattr(e, 'filename', '?')}",
                "category": "ERROR_LOADING",
            }
        )
        return out
    except PermissionError as e:
        out.update(
            {
                "canonical_equal": False,
                "canonical_reason": f"PERMISSION:{getattr(e, 'filename', '?')}",
                "category": "ERROR_LOADING",
            }
        )
        return out
    except Exception as e:
        out.update(
            {
                "canonical_equal": False,
                "canonical_reason": f"LOAD_ERROR:{type(e).__name__}:{e}",
                "category": "ERROR_LOADING",
            }
        )
        return out

    def _equal_voxels(a: np.ndarray, b: np.ndarray) -> bool:
        def _eq(x, y):
            try:
                return np.array_equal(x, y, equal_nan=True)
            except TypeError:
                return np.array_equal(x, y)

        if a.shape == b.shape:
            return _eq(a, b)
        if a.ndim == 3 and b.ndim == 4 and b.shape[3] == 1:
            return _eq(a, b[..., 0])
        if b.ndim == 3 and a.ndim == 4 and a.shape[3] == 1:
            return _eq(a[..., 0], b)
        return False

    same_data = _equal_voxels(arr1, arr2)
    out["meta1"] = meta1
    out["meta2"] = meta2
    out["canonical_equal"] = bool(same_data)
    out["canonical_reason"] = "OK" if same_data else "DATA_DIFFER"

    if same_data:
        time_zooms_diff = (
            len(meta1["zooms"]) > 3 or len(meta2["zooms"]) > 3
        ) and tuple(meta1["zooms"][3:]) != tuple(meta2["zooms"][3:])
        affine_meta_diff = (
            (meta1["qform_code"] != meta2["qform_code"])
            or (meta1["sform_code"] != meta2["sform_code"])
            or (
                meta1.get("qform") is not None
                and meta2.get("qform") is not None
                and not np.allclose(
                    np.array(meta1["qform"]), np.array(meta2["qform"]), atol=1e-5
                )
            )
            or (
                meta1.get("qform") is None
                and meta2.get("qform") is None
                and meta1.get("sform") is not None
                and meta2.get("sform") is not None
                and not np.allclose(
                    np.array(meta1["sform"]), np.array(meta2["sform"]), atol=1e-5
                )
            )
        )
        orient_diff = (meta1["axcodes_original"] != meta2["axcodes_original"]) or (
            meta1["axcodes_canonical"] != meta2["axcodes_canonical"]
        )
        scaling_dtype_diff = (
            (meta1["dtype"] != meta2["dtype"])
            or (meta1["slope"] != meta2["slope"])
            or (meta1["inter"] != meta2["inter"])
        )
        shape_zooms_diff = tuple(meta1["shape"][:3]) != tuple(
            meta2["shape"][:3]
        ) or tuple(meta1["zooms"][:3]) != tuple(meta2["zooms"][:3])
        is_gz_pair_or_container = str(p1).endswith(".gz") or str(p2).endswith(".gz")

        flags = {
            "flag_orient_diff": orient_diff,
            "flag_scaling_or_dtype_diff": scaling_dtype_diff,
            "flag_shape_or_space_zooms_diff": shape_zooms_diff,
            "flag_time_zooms_diff": time_zooms_diff,
            "flag_affine_meta_diff": affine_meta_diff,
            "flag_header_other_diff": (
                not out["header_strict_equal"]
                and not orient_diff
                and not affine_meta_diff
                and not scaling_dtype_diff
                and not shape_zooms_diff
                and not time_zooms_diff
            ),
        }
        out["flags"] = flags

        if not out["header_strict_equal"]:
            out["category"] = "CAN_EQUAL_HEADER"
        elif is_gz_pair_or_container:
            out["category"] = "CAN_EQUAL_COMPRESSION_GZ"
        else:
            out["category"] = "CAN_EQUAL_OTHER"
    else:
        if raw_reason == "DIFF_SIZE":
            out["category"] = "DIFF_SIZE_AND_CAN_MISMATCH"
        elif raw_reason == "DIFF_CONTENT":
            out["category"] = "DIFF_BYTES_AND_CAN_MISMATCH"
        else:
            out["category"] = f"RAW_{raw_reason}_AND_CAN_MISMATCH"
    return out


def all_identical_canonical(
    paths: list[str],
) -> tuple[bool, Optional[Tuple[str, str, str]]]:
    paths = [p for p in paths if p]
    if len(paths) < 2:
        return False, None
    anchor = paths[0]
    for other in paths[1:]:
        res = compare_nifti(anchor, other)
        if not res.get("canonical_equal", False):
            return False, (anchor, other, str(res.get("canonical_reason", "")))
    return True, None


def _check_can_worker(paths_joined: str) -> str:
    paths: List[str] = [p for p in paths_joined.split("|::|") if p]
    if len(paths) < 2:
        return "UNKNOWN"
    ok, _ = all_identical_canonical(paths)
    return "TRUE" if ok else "FALSE"


def verify_dupe_table(
    in_csv: Union[str, Path],
    out_with_checks_csv: Union[str, Path],
    out_pairwise_issues_csv: Union[str, Path],
    exclude_categories: set[str] = {"SAME_BYTES"},
    thread_workers: Optional[int] = None,
    process_workers: Optional[int] = None,
) -> None:
    FLAGS_OUT = [
        "flag_orient_diff",
        "flag_scaling_or_dtype_diff",
        "flag_shape_or_space_zooms_diff",
        "flag_time_zooms_diff",
        "flag_affine_meta_diff",
        "flag_header_other_diff",
    ]

    def _iso(ts):
        if ts is None:
            return ""
        try:
            return datetime.fromtimestamp(ts).isoformat(sep=" ", timespec="seconds")
        except Exception:
            return ""

    def _stat_info(p: str):
        try:
            st = os.stat(p, follow_symlinks=True)
            return {
                "exists": True,
                "size": st.st_size,
                "mtime": st.st_mtime,
                "ctime": st.st_ctime,
            }
        except Exception:
            return {"exists": False, "size": None, "mtime": None, "ctime": None}

    def _sorted_dup_cols(cols, kind: str) -> list[str]:
        pat = re.compile(rf"^duplicate_(\d+)_({re.escape(kind)})$")
        pairs = []
        for c in cols:
            m = pat.match(c)
            if m:
                pairs.append((int(m.group(1)), c))
        return [c for _, c in sorted(pairs, key=lambda x: x[0])]

    def _meta_for_index(row: pd.Series, i: int) -> dict:
        di = row.get(f"dataset_{i}", "") or row.get("dataset", "")
        si = row.get(f"subject_id_{i}", "") or row.get("subject_id", "")
        se = row.get(f"session_id_{i}", "") or row.get("session_id", "")
        ci = row.get(f"duplicate_{i}_candidate", "")
        ri = row.get(f"duplicate_{i}_resolved_path", "")
        return {
            "dataset": str(di),
            "subject_id": str(si),
            "session_id": str(se),
            "candidate": str(ci),
            "resolved_path": str(ri),
        }

    in_csv = Path(in_csv)
    out_with_checks_csv = Path(out_with_checks_csv)
    out_pairwise_issues_csv = Path(out_pairwise_issues_csv)

    df = pd.read_csv(in_csv, dtype=str).fillna("")
    orig_cols = df.columns.tolist()

    rp_cols = _sorted_dup_cols(orig_cols, "resolved_path")
    if not rp_cols:
        df.to_csv(out_with_checks_csv, index=False)
        empty_cols = [
            "row_index",
            "pair_kind",
            "dataset_i",
            "subject_id_i",
            "session_id_i",
            "candidate_i",
            "resolved_path_i",
            "dataset_j",
            "subject_id_j",
            "session_id_j",
            "candidate_j",
            "resolved_path_j",
            "category",
            "raw_equal",
            "raw_reason",
            "canonical_equal",
            "canonical_reason",
            *FLAGS_OUT,
        ]
        pd.DataFrame(columns=empty_cols).to_csv(out_pairwise_issues_csv, index=False)
        return

    new_order = []
    for c in orig_cols:
        new_order.append(c)
        if c in rp_cols:
            base = c[: -len("_resolved_path")]
            mt, ct = f"{base}_mtime", f"{base}_ctime"
            if mt not in df.columns:
                df[mt] = ""
            if ct not in df.columns:
                df[ct] = ""
            new_order += [mt, ct]

    if "byte_identical_all" not in df.columns:
        df["byte_identical_all"] = ""
    if "canonical_identical_all" not in df.columns:
        df["canonical_identical_all"] = ""

    df["_paths_joined"] = ""
    for idx, row in df.iterrows():
        paths = []
        for rp_col in rp_cols:
            rp = str(row[rp_col]).strip()
            if not rp:
                continue
            info = _stat_info(rp)
            base = rp_col[: -len("_resolved_path")]
            df.at[idx, f"{base}_mtime"] = _iso(info["mtime"])
            df.at[idx, f"{base}_ctime"] = _iso(info["ctime"])
            if info["exists"]:
                paths.append(rp)
        df.at[idx, "_paths_joined"] = "|::|".join(paths)

    def _check_raw(paths_joined: str) -> str:
        paths = [p for p in paths_joined.split("|::|") if p]
        if len(paths) < 2:
            return "UNKNOWN"
        status, _ = all_identical_raw(paths)
        return status

    if thread_workers is None:
        thread_workers = min(16, os.cpu_count() or 4)

    with ThreadPoolExecutor(max_workers=thread_workers) as ex:
        fut2idx = {
            ex.submit(_check_raw, pj): idx for idx, pj in df["_paths_joined"].items()
        }
        for fut in as_completed(fut2idx):
            idx = fut2idx[fut]
            df.at[idx, "byte_identical_all"] = fut.result()

    need_can_idx = df.index[df["byte_identical_all"] == "FALSE"].tolist()
    df.loc[df["byte_identical_all"] == "TRUE", "canonical_identical_all"] = "TRUE"
    df.loc[df["byte_identical_all"] == "UNKNOWN", "canonical_identical_all"] = "UNKNOWN"

    if need_can_idx:
        if process_workers is None:
            process_workers = max(1, min((os.cpu_count() or 4) // 2, 8))
        mp_ctx = get_context("spawn")
        with ProcessPoolExecutor(max_workers=process_workers, mp_context=mp_ctx) as ex:
            fut2idx = {
                ex.submit(_check_can_worker, df.at[idx, "_paths_joined"]): idx
                for idx in need_can_idx
            }
            for fut in as_completed(fut2idx):
                idx = fut2idx[fut]
                df.at[idx, "canonical_identical_all"] = fut.result()

    lowname = in_csv.name.lower()
    if "within_sessions" in lowname:
        pair_kind = "within_sessions"
    elif "within_subjects" in lowname:
        pair_kind = "within_subjects"
    elif "within_datasets" in lowname:
        pair_kind = "within_datasets"
    elif "across_datasets" in lowname:
        pair_kind = "across_datasets"
    else:
        pair_kind = "unknown"

    @lru_cache(maxsize=200_000)
    def _compare_cached(p1: str, p2: str) -> dict:
        a, b = (p1, p2) if p1 <= p2 else (p2, p1)
        return compare_nifti(a, b)

    idx_nums = sorted(
        {
            int(m.group(1))
            for c in rp_cols
            for m in [re.search(r"duplicate_(\d+)_resolved_path$", c)]
            if m
        }
    )

    issue_rows = []
    row_flags_accum = defaultdict(lambda: {k: False for k in FLAGS_OUT})

    for ridx, row in df.iterrows():
        joined_set = {p for p in str(df.at[ridx, "_paths_joined"]).split("|::|") if p}
        if len(joined_set) < 2:
            continue
        if str(df.at[ridx, "byte_identical_all"]).upper() == "TRUE":
            continue

        present = []
        for i in idx_nums:
            rp = str(row.get(f"duplicate_{i}_resolved_path", "")).strip()
            if rp and (rp in joined_set):
                present.append(i)
        if len(present) < 2:
            continue

        anchor_i = present[0]
        meta_i = _meta_for_index(row, anchor_i)
        p_i = meta_i["resolved_path"]
        if not p_i:
            continue

        for j in present[1:]:
            meta_j = _meta_for_index(row, j)
            p_j = meta_j["resolved_path"]
            if not p_j:
                continue
            same_bytes, _ = files_identical_raw(p_i, p_j)
            if same_bytes:
                continue

            try:
                res = _compare_cached(p_i, p_j)
            except Exception as e:
                issue_rows.append(
                    {
                        "row_index": ridx,
                        "pair_kind": pair_kind,
                        "dataset_i": meta_i["dataset"],
                        "subject_id_i": meta_i["subject_id"],
                        "session_id_i": meta_i["session_id"],
                        "candidate_i": meta_i["candidate"],
                        "resolved_path_i": p_i,
                        "dataset_j": meta_j["dataset"],
                        "subject_id_j": meta_j["subject_id"],
                        "session_id_j": meta_j["session_id"],
                        "candidate_j": meta_j["candidate"],
                        "resolved_path_j": p_j,
                        "category": f"COMPARE_ERROR:{type(e).__name__}:{e}",
                        "raw_equal": "",
                        "raw_reason": "",
                        "canonical_equal": "",
                        "canonical_reason": "",
                        **{k: False for k in FLAGS_OUT},
                    }
                )
                continue

            flags = res.get("flags", {}) or {}
            for k in FLAGS_OUT:
                row_flags_accum[ridx][k] = bool(
                    row_flags_accum[ridx][k] or flags.get(k, False)
                )

            cat = str(res.get("category", ""))
            if cat in exclude_categories:
                continue

            issue_rows.append(
                {
                    "row_index": ridx,
                    "pair_kind": pair_kind,
                    "dataset_i": meta_i["dataset"],
                    "subject_id_i": meta_i["subject_id"],
                    "session_id_i": meta_i["session_id"],
                    "candidate_i": meta_i["candidate"],
                    "resolved_path_i": p_i,
                    "dataset_j": meta_j["dataset"],
                    "subject_id_j": meta_j["subject_id"],
                    "session_id_j": meta_j["session_id"],
                    "candidate_j": meta_j["candidate"],
                    "resolved_path_j": p_j,
                    "category": cat,
                    "raw_equal": res.get("raw_equal", ""),
                    "raw_reason": res.get("raw_reason", ""),
                    "canonical_equal": res.get("canonical_equal", ""),
                    "canonical_reason": res.get("canonical_reason", ""),
                    **{k: bool(flags.get(k, False)) for k in FLAGS_OUT},
                }
            )

    cols_pairwise = [
        "row_index",
        "pair_kind",
        "dataset_i",
        "subject_id_i",
        "session_id_i",
        "candidate_i",
        "resolved_path_i",
        "dataset_j",
        "subject_id_j",
        "session_id_j",
        "candidate_j",
        "resolved_path_j",
        "category",
        "raw_equal",
        "raw_reason",
        "canonical_equal",
        "canonical_reason",
        *FLAGS_OUT,
    ]
    if issue_rows:
        out_df = pd.DataFrame(issue_rows, columns=cols_pairwise)
        sort_cols = [
            c
            for c in [
                "pair_kind",
                "dataset_i",
                "subject_id_i",
                "session_id_i",
                "row_index",
            ]
            if c in out_df.columns
        ]
        if sort_cols:
            out_df = out_df.sort_values(sort_cols, kind="mergesort")
        out_df.to_csv(out_pairwise_issues_csv, index=False)
    else:
        pd.DataFrame(columns=cols_pairwise).to_csv(out_pairwise_issues_csv, index=False)

    final_cols = new_order + ["byte_identical_all", "canonical_identical_all"]
    df_out = df[final_cols].copy()

    for k in FLAGS_OUT:
        if k not in df_out.columns:
            df_out[k] = False
    for ridx, agg in row_flags_accum.items():
        for k, v in agg.items():
            df_out.at[ridx, k] = bool(v)

    df_out["canonical_identical_all"] = df_out["canonical_identical_all"].astype(str)
    false_mask = df_out["canonical_identical_all"].str.upper().eq("FALSE")
    if false_mask.any():
        out_false = out_with_checks_csv.with_name(
            out_with_checks_csv.name.replace("_with_checks", "_canonical_false")
        )
        df_out.loc[false_mask].to_csv(out_false, index=False)

    df_out.to_csv(out_with_checks_csv, index=False)


def verify_all_categories(
    in_dir: Union[str, Path],
    out_dir: Union[str, Path],
    thread_workers: Optional[int] = None,
    process_workers: Optional[int] = None,
    exclude_categories: set[str] = {"CAN_EQUAL_COMPRESSION_GZ", "SAME_BYTES"},
) -> None:
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = [
        "within_sessions",
        "within_subjects",
        "within_datasets",
        "across_datasets",
    ]

    for base in names:
        in_csv = in_dir / f"{base}.csv"
        if not in_csv.exists():
            print(f"[WARN] missing: {in_csv}")
            continue
        out_with = out_dir / f"{base}_with_checks.csv"
        out_pair = out_dir / f"{base}_pairwise_issues.csv"
        print(f"[VERIFY] {base} ...")
        verify_dupe_table(
            in_csv,
            out_with,
            out_pair,
            thread_workers=thread_workers,
            process_workers=process_workers,
            exclude_categories=exclude_categories,
        )
        print(f"[VERIFY] wrote:\n  - {out_with}\n  - {out_pair}")
