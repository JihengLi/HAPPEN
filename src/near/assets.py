"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

from __future__ import annotations

import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from ..utils import visualize_t1w
from .preprocessing import (
    run_n4,
    run_deepbet_mask,
    run_ants_reg_rigid,
    apply_label_affine,
    normalize_t1_minmax,
)


_SCAN_SLOT_RE = re.compile(r"^candidate_(\d+)_dataset$")
DEFAULT_FRACS: List[float] = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]
DEFAULT_PERC: Tuple[float, float] = (1.0, 99.0)


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


def atomic_write_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(out_path)


def _atomic_save_nifti(
    path: Path,
    data: np.ndarray,
    affine: np.ndarray,
    header: nib.Nifti1Header,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    name = path.name
    if name.endswith(".nii.gz"):
        tmp = path.with_name(name[:-7] + ".tmp.nii.gz")
    elif name.endswith(".nii"):
        tmp = path.with_name(name[:-4] + ".tmp.nii")
    else:
        tmp = path.with_name(name + ".tmp.nii.gz")

    hdr = header.copy()
    hdr.set_data_dtype(np.float32)

    nib.save(
        nib.Nifti1Image(data.astype(np.float32, copy=False), affine, hdr),
        str(tmp),
    )
    tmp.replace(path)


def _pair_key(scan_uid_a: str, scan_uid_b: str) -> str:
    a = str(scan_uid_a).strip()
    b = str(scan_uid_b).strip()
    x, y = sorted([a, b])
    return f"{x}__{y}"


def _norm_cache_path(review_dir: Path, scan_uid: str) -> Path:
    return review_dir / "cache" / "norm" / f"{scan_uid}.nii.gz"


def _work_dir(review_dir: Path, scan_uid: str) -> Path:
    return review_dir / "cache" / "work" / scan_uid


def to_uint8(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = a.astype(np.float32, copy=False)
        vmax = float(a.max()) if a.size else 1.0
        vmin = float(a.min()) if a.size else 0.0
        if vmax <= 1.0 and vmin >= 0.0:
            a = a * 255.0
        else:
            a = (a - vmin) / (vmax - vmin + 1e-6) * 255.0
        a = np.clip(a, 0, 255).astype(np.uint8)

    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    elif a.ndim == 3 and a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    elif a.ndim == 3 and a.shape[-1] == 4:
        a = a[..., :3]
    return a


def split_mosaic_to_three(img: np.ndarray) -> List[np.ndarray]:
    img = to_uint8(img)
    h, w = img.shape[:2]
    if w >= h:
        tile_w = w // 3
        tiles = [img[:, i * tile_w : (i + 1) * tile_w, ...] for i in range(3)]
    else:
        tile_h = h // 3
        tiles = [img[i * tile_h : (i + 1) * tile_h, :, ...] for i in range(3)]

    min_w = min(t.shape[1] for t in tiles)
    min_h = min(t.shape[0] for t in tiles)
    return [t[:min_h, :min_w] for t in tiles]


def make_column_from_mosaic(img: np.ndarray) -> np.ndarray:
    tiles = split_mosaic_to_three(img)
    min_w = min(t.shape[1] for t in tiles)
    tiles = [t[:, :min_w] for t in tiles]
    return np.vstack(tiles)


def hstack_with_padding(
    columns: List[np.ndarray],
    pad_value: int = 0,
    gap: int = 0,
) -> np.ndarray:
    cols_u8 = []
    for c in columns:
        cc = to_uint8(c)
        if cc.ndim == 2:
            cc = np.stack([cc, cc, cc], axis=-1)
        cols_u8.append(cc)

    heights = [c.shape[0] for c in cols_u8]
    widths = [c.shape[1] for c in cols_u8]
    H = max(heights)
    W = max(widths)

    out = []
    for i, cc in enumerate(cols_u8):
        if cc.shape[1] < W:
            pad_w = np.full(
                (cc.shape[0], W - cc.shape[1], 3),
                pad_value,
                dtype=cc.dtype,
            )
            cc = np.hstack([cc, pad_w])
        elif cc.shape[1] > W:
            cc = cc[:, :W]

        if cc.shape[0] < H:
            pad_h = np.full((H - cc.shape[0], W, 3), pad_value, dtype=cc.dtype)
            cc = np.vstack([cc, pad_h])
        elif cc.shape[0] > H:
            cc = cc[:H, :]

        out.append(cc)

        if gap > 0 and i < len(cols_u8) - 1:
            out.append(np.full((H, gap, 3), pad_value, dtype=np.uint8))

    return np.hstack(out)


def _blank_col(h: int = 256 * 3, w: int = 256, v: int = 230) -> np.ndarray:
    return np.ones((h, w, 3), dtype=np.uint8) * v


def _load_nifti_shape_3d(nifti_path: Path) -> Tuple[int, int, int]:
    img = nib.load(str(nifti_path), mmap=True)
    img_ras = nib.as_closest_canonical(img)
    sx, sy, sz = img_ras.shape[:3]
    return int(sx), int(sy), int(sz)


def _frac_to_index(frac: float, L: int) -> int:
    if L <= 1:
        return 0
    idx = int(round(frac * (L - 1)))
    return max(0, min(L - 1, idx))


def build_percent_groups(nifti_path: Path, fracs: Sequence[float]) -> List[dict]:
    sx, sy, sz = _load_nifti_shape_3d(nifti_path)
    groups: List[dict] = []
    for f in fracs:
        groups.append(
            dict(
                sagittal_slices=_frac_to_index(float(f), sx),
                coronal_slices=_frac_to_index(float(f), sy),
                axial_slices=_frac_to_index(float(f), sz),
            )
        )
    return groups


def _safe_columns_for_groups(nifti_path: Path, groups: List[dict]) -> List[np.ndarray]:
    cols: List[np.ndarray] = []
    for g in groups:
        try:
            mosaic = visualize_t1w(
                t1_file=str(nifti_path),
                sagittal_slices=g["sagittal_slices"],
                coronal_slices=g["coronal_slices"],
                axial_slices=g["axial_slices"],
                perc=DEFAULT_PERC,
                save_path=None,
                show_img=False,
            )
            cols.append(make_column_from_mosaic(mosaic))
        except Exception:
            cols.append(_blank_col())
    return cols


def _difference_tile(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = to_uint8(a).astype(np.float32)
    bb = to_uint8(b).astype(np.float32)
    diff = np.abs(aa - bb).mean(axis=-1)
    diff = np.clip(diff, 0, 255).astype(np.uint8)
    return to_uint8(diff)


def _checkerboard_tile(a: np.ndarray, b: np.ndarray, tile: int = 16) -> np.ndarray:
    aa = to_uint8(a)
    bb = to_uint8(b)

    h = min(aa.shape[0], bb.shape[0])
    w = min(aa.shape[1], bb.shape[1])

    aa = aa[:h, :w]
    bb = bb[:h, :w]

    yy = np.arange(h)[:, None]
    xx = np.arange(w)[None, :]
    mask = (yy // tile + xx // tile) % 2 == 0

    out = np.empty_like(aa)
    out[mask] = aa[mask]
    out[~mask] = bb[~mask]
    return out


def _pair_column_from_mosaics(
    img_a: np.ndarray,
    img_b: np.ndarray,
    kind: str,
) -> np.ndarray:
    tiles_a = split_mosaic_to_three(img_a)
    tiles_b = split_mosaic_to_three(img_b)

    out_tiles = []
    for ta, tb in zip(tiles_a, tiles_b):
        h = min(ta.shape[0], tb.shape[0])
        w = min(ta.shape[1], tb.shape[1])
        ta = ta[:h, :w]
        tb = tb[:h, :w]

        if kind == "diff":
            out_tiles.append(_difference_tile(ta, tb))
        elif kind == "checkerboard":
            out_tiles.append(_checkerboard_tile(ta, tb))
        else:
            raise ValueError(f"unknown pair kind: {kind}")

    min_w = min(t.shape[1] for t in out_tiles)
    out_tiles = [t[:, :min_w] for t in out_tiles]
    return np.vstack(out_tiles)


def _safe_pair_columns_for_groups(
    nifti_path_a: Path,
    nifti_path_b: Path,
    groups: List[dict],
    kind: str,
) -> List[np.ndarray]:
    cols: List[np.ndarray] = []
    for g in groups:
        try:
            img_a = visualize_t1w(
                t1_file=str(nifti_path_a),
                sagittal_slices=g["sagittal_slices"],
                coronal_slices=g["coronal_slices"],
                axial_slices=g["axial_slices"],
                perc=DEFAULT_PERC,
                save_path=None,
                show_img=False,
            )
            img_b = visualize_t1w(
                t1_file=str(nifti_path_b),
                sagittal_slices=g["sagittal_slices"],
                coronal_slices=g["coronal_slices"],
                axial_slices=g["axial_slices"],
                perc=DEFAULT_PERC,
                save_path=None,
                show_img=False,
            )
            cols.append(_pair_column_from_mosaics(img_a, img_b, kind=kind))
        except Exception:
            cols.append(_blank_col())
    return cols


def render_scan_png(
    norm_path: Path,
    out_path: Path,
    overwrite: bool,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        return

    if not norm_path.exists():
        raise FileNotFoundError(f"norm_path not found: {norm_path}")

    groups = build_percent_groups(norm_path, fracs=DEFAULT_FRACS)
    cols = _safe_columns_for_groups(norm_path, groups=groups)
    final_img = hstack_with_padding(cols, pad_value=0, gap=0)
    Image.fromarray(to_uint8(final_img)).save(out_path)


def render_pair_png(
    norm_path_a: Path,
    norm_path_b: Path,
    out_path: Path,
    kind: str,
    overwrite: bool,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        return

    if not norm_path_a.exists():
        raise FileNotFoundError(f"norm_path_a not found: {norm_path_a}")
    if not norm_path_b.exists():
        raise FileNotFoundError(f"norm_path_b not found: {norm_path_b}")

    groups = build_percent_groups(norm_path_a, fracs=DEFAULT_FRACS)
    cols = _safe_pair_columns_for_groups(
        norm_path_a,
        norm_path_b,
        groups=groups,
        kind=kind,
    )
    final_img = hstack_with_padding(cols, pad_value=0, gap=0)
    Image.fromarray(to_uint8(final_img)).save(out_path)


def _read_manifest(
    manifest_csv: Union[str, Path],
) -> pd.DataFrame:
    manifest_csv = Path(manifest_csv)

    required_cols = [
        "dataset",
        "subject_id",
        "session_id",
        "resolved_path",
        "scan_uid",
    ]

    df = pd.read_csv(manifest_csv, dtype=str).fillna("")
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{manifest_csv} is missing required columns: {missing}")

    for c in required_cols:
        df[c] = df[c].fillna("").astype(str)

    if df["scan_uid"].duplicated().any():
        dups = df.loc[df["scan_uid"].duplicated(), "scan_uid"].tolist()[:10]
        raise ValueError(f"{manifest_csv} contains duplicated scan_uid values: {dups}")

    return df


def _manifest_meta_map(
    manifest_df: pd.DataFrame,
) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for _, r in manifest_df.iterrows():
        uid = str(r["scan_uid"])
        out[uid] = {
            "dataset": str(r["dataset"]),
            "subject_id": str(r["subject_id"]),
            "session_id": str(r["session_id"]),
            "resolved_path": str(r["resolved_path"]),
        }
    return out


def _candidate_slots_from_cols(cols: Sequence[str]) -> List[int]:
    slots = []
    for c in cols:
        m = _SCAN_SLOT_RE.match(str(c).strip())
        if m:
            slots.append(int(m.group(1)))
    return sorted(set(slots))


def _read_scan_level_review(
    scan_level_review_csv: Union[str, Path],
) -> pd.DataFrame:
    scan_level_review_csv = Path(scan_level_review_csv)
    df = pd.read_csv(scan_level_review_csv, dtype=str).fillna("")
    required_cols = [
        "query_dataset",
        "query_subject_id",
        "query_session_id",
        "query_scan_uid",
        "query_png_path",
        "n_candidates",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{scan_level_review_csv} is missing required columns: {missing}"
        )
    return df


def _read_subject_level_review(
    subject_level_review_csv: Union[str, Path],
) -> pd.DataFrame:
    subject_level_review_csv = Path(subject_level_review_csv)
    df = pd.read_csv(subject_level_review_csv, dtype=str).fillna("")
    required_cols = [
        "group_id",
        "group_size",
        "query_dataset",
        "query_subject_id",
        "n_candidates",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{subject_level_review_csv} is missing required columns: {missing}"
        )
    return df


def _register_unique_path(
    mapping: Dict[str, str],
    key: str,
    rel_path: str,
    kind: str,
) -> None:
    k = str(key).strip()
    p = str(rel_path).strip()
    if not k:
        return
    if k in mapping and mapping[k] != p:
        raise ValueError(f"inconsistent {kind} path for key={k}: {mapping[k]} vs {p}")
    mapping[k] = p


def collect_scan_requests(
    scan_level_review_csv: Union[str, Path],
    subject_level_review_csv: Union[str, Path],
) -> Dict[str, str]:
    scan_df = _read_scan_level_review(scan_level_review_csv)
    subj_df = _read_subject_level_review(subject_level_review_csv)

    scan_uid_to_png: Dict[str, str] = {}

    scan_slots = _candidate_slots_from_cols(scan_df.columns.tolist())
    for _, r in scan_df.iterrows():
        _register_unique_path(
            scan_uid_to_png,
            str(r["query_scan_uid"]),
            str(r["query_png_path"]),
            kind="scan png",
        )
        for k in scan_slots:
            ds = clean_str(r.get(f"candidate_{k}_dataset", ""))
            if not ds:
                continue
            _register_unique_path(
                scan_uid_to_png,
                str(r.get(f"candidate_{k}_scan_uid", "")),
                str(r.get(f"candidate_{k}_png_path", "")),
                kind="scan png",
            )

    subj_slots = _candidate_slots_from_cols(subj_df.columns.tolist())
    for _, r in subj_df.iterrows():
        for k in subj_slots:
            ds = clean_str(r.get(f"candidate_{k}_dataset", ""))
            if not ds:
                continue
            _register_unique_path(
                scan_uid_to_png,
                str(r.get(f"candidate_{k}_query_exemplar_scan_uid", "")),
                str(r.get(f"candidate_{k}_query_exemplar_png_path", "")),
                kind="scan png",
            )
            _register_unique_path(
                scan_uid_to_png,
                str(r.get(f"candidate_{k}_candidate_exemplar_scan_uid", "")),
                str(r.get(f"candidate_{k}_candidate_exemplar_png_path", "")),
                kind="scan png",
            )

    return scan_uid_to_png


def collect_pair_requests(
    scan_level_review_csv: Union[str, Path],
    subject_level_review_csv: Union[str, Path],
) -> Dict[str, Dict[str, str]]:
    scan_df = _read_scan_level_review(scan_level_review_csv)
    subj_df = _read_subject_level_review(subject_level_review_csv)

    pair_map: Dict[str, Dict[str, str]] = {}

    scan_slots = _candidate_slots_from_cols(scan_df.columns.tolist())
    for _, r in scan_df.iterrows():
        q_uid = str(r["query_scan_uid"]).strip()
        for k in scan_slots:
            ds = clean_str(r.get(f"candidate_{k}_dataset", ""))
            if not ds:
                continue
            c_uid = str(r.get(f"candidate_{k}_scan_uid", "")).strip()
            if not c_uid:
                continue
            pkey = _pair_key(q_uid, c_uid)
            diff_path = str(r.get(f"candidate_{k}_diff_path", "")).strip()
            cb_path = str(r.get(f"candidate_{k}_checkerboard_path", "")).strip()
            rec = {
                "scan_uid_a": min(q_uid, c_uid),
                "scan_uid_b": max(q_uid, c_uid),
                "diff_path": diff_path,
                "checkerboard_path": cb_path,
            }
            if pkey in pair_map:
                if pair_map[pkey] != rec:
                    raise ValueError(f"inconsistent pair asset paths for {pkey}")
            else:
                pair_map[pkey] = rec

    subj_slots = _candidate_slots_from_cols(subj_df.columns.tolist())
    for _, r in subj_df.iterrows():
        for k in subj_slots:
            ds = clean_str(r.get(f"candidate_{k}_dataset", ""))
            if not ds:
                continue
            q_uid = str(r.get(f"candidate_{k}_query_exemplar_scan_uid", "")).strip()
            c_uid = str(r.get(f"candidate_{k}_candidate_exemplar_scan_uid", "")).strip()
            if not q_uid or not c_uid:
                continue
            pkey = _pair_key(q_uid, c_uid)
            diff_path = str(r.get(f"candidate_{k}_diff_path", "")).strip()
            cb_path = str(r.get(f"candidate_{k}_checkerboard_path", "")).strip()
            rec = {
                "scan_uid_a": min(q_uid, c_uid),
                "scan_uid_b": max(q_uid, c_uid),
                "diff_path": diff_path,
                "checkerboard_path": cb_path,
            }
            if pkey in pair_map:
                if pair_map[pkey] != rec:
                    raise ValueError(f"inconsistent pair asset paths for {pkey}")
            else:
                pair_map[pkey] = rec

    return pair_map


def _build_review_norm_cache(
    resolved_path: Path,
    atlas_image: Path,
    out_norm_path: Path,
    work_dir: Path,
    winsor: Tuple[float, float],
    hist_matching: bool,
    repro: bool,
    delete_extras: bool,
) -> Path:
    if out_norm_path.exists():
        return out_norm_path

    work_dir.mkdir(parents=True, exist_ok=True)

    n4_path = work_dir / "n4.nii.gz"
    masked_brain_path = work_dir / "masked_brain.nii.gz"
    skull_mask_native_path = work_dir / "skull_mask_native.nii.gz"
    reg_path = work_dir / "reg_rigid.nii.gz"
    mat_path = work_dir / "rigid.mat"
    skull_mask_reg_path = work_dir / "skull_mask_reg.nii.gz"
    n4_brain_path = work_dir / "n4_brain.nii.gz"

    try:
        run_n4(resolved_path, n4_path)

        run_deepbet_mask(
            in_path=n4_path,
            out_brain_path=masked_brain_path,
            out_mask_path=skull_mask_native_path,
        )

        run_ants_reg_rigid(
            fixed=atlas_image,
            moving=masked_brain_path,
            out_warped=reg_path,
            out_mat=mat_path,
            hist_matching=hist_matching,
            repro=repro,
            delete_extras=delete_extras,
        )

        apply_label_affine(
            moving_label=skull_mask_native_path,
            fixed_img=atlas_image,
            out_label=skull_mask_reg_path,
            affine_mat=mat_path,
        )

        run_n4(reg_path, n4_brain_path)

        reg_img = nib.load(str(n4_brain_path))
        vol = reg_img.get_fdata(dtype=np.float32)

        mask_img = nib.load(str(skull_mask_reg_path))
        brain_mask = mask_img.get_fdata() > 0

        if brain_mask.shape != vol.shape:
            raise ValueError(
                f"brain mask shape {brain_mask.shape} != registered image shape {vol.shape}"
            )

        vol_norm = normalize_t1_minmax(
            vol=vol,
            brain_mask=brain_mask,
            winsor=winsor,
        )

        _atomic_save_nifti(
            out_norm_path,
            vol_norm,
            reg_img.affine,
            reg_img.header,
        )
        return out_norm_path

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def ensure_scan_assets(
    review_dir: Union[str, Path],
    manifest_csv: Union[str, Path],
    atlas_image: Union[str, Path],
    scan_level_review_csv: Union[str, Path],
    subject_level_review_csv: Union[str, Path],
    workers: int = 24,
    overwrite: bool = False,
    winsor: Tuple[float, float] = (1.0, 99.0),
    hist_matching: bool = False,
    repro: bool = False,
    delete_extras: bool = True,
) -> Tuple[Path, Path]:
    review_dir = Path(review_dir).expanduser().resolve()
    manifest_csv = Path(manifest_csv).expanduser().resolve()
    atlas_image = Path(atlas_image).expanduser().resolve()

    if not atlas_image.exists():
        raise FileNotFoundError(f"atlas image not found: {atlas_image}")

    manifest_df = _read_manifest(manifest_csv)
    meta_map = _manifest_meta_map(manifest_df)

    scan_requests = collect_scan_requests(
        scan_level_review_csv=scan_level_review_csv,
        subject_level_review_csv=subject_level_review_csv,
    )

    missing_uids = sorted([uid for uid in scan_requests if uid not in meta_map])
    if missing_uids:
        raise ValueError(
            f"scan_uids referenced in review tables not found in manifest: {missing_uids[:10]}"
        )

    fail_csv = review_dir / "scan_asset_failures.csv"

    recs: List[Dict[str, str]] = []
    for scan_uid, rel_png in sorted(scan_requests.items()):
        meta = meta_map[scan_uid]
        recs.append(
            {
                "scan_uid": scan_uid,
                "resolved_path": meta["resolved_path"],
                "png_path": str((review_dir / rel_png).resolve()),
                "norm_path": str(_norm_cache_path(review_dir, scan_uid)),
                "work_dir": str(_work_dir(review_dir, scan_uid)),
            }
        )

    failures_all: List[Dict[str, str]] = []
    ok_total = 0

    def _worker(rec: Dict[str, str]) -> Tuple[int, List[Dict[str, str]]]:
        failures: List[Dict[str, str]] = []
        ok = 0
        try:
            norm_path = _build_review_norm_cache(
                resolved_path=Path(rec["resolved_path"]),
                atlas_image=atlas_image,
                out_norm_path=Path(rec["norm_path"]),
                work_dir=Path(rec["work_dir"]),
                winsor=winsor,
                hist_matching=hist_matching,
                repro=repro,
                delete_extras=delete_extras,
            )
            render_scan_png(
                norm_path=norm_path,
                out_path=Path(rec["png_path"]),
                overwrite=overwrite,
            )
            ok = 1
        except Exception as e:
            failures.append(
                {
                    "scan_uid": rec["scan_uid"],
                    "resolved_path": rec["resolved_path"],
                    "png_path": rec["png_path"],
                    "error": str(e),
                }
            )
        return ok, failures

    with ProcessPoolExecutor(max_workers=int(workers)) as ex:
        futs = [ex.submit(_worker, rec) for rec in recs]
        for fut in tqdm(
            as_completed(futs),
            total=len(futs),
            desc="review_scan_assets",
            unit="scan",
        ):
            ok, fails = fut.result()
            ok_total += int(ok)
            if fails:
                failures_all.extend(fails)

    pd.DataFrame(failures_all).to_csv(fail_csv, index=False)

    print("[REVIEW ASSETS] scan assets done")
    print(f"[REVIEW ASSETS] scans ok: {ok_total}/{len(recs)}")
    print(f"[REVIEW ASSETS] scan failures: {len(failures_all)} -> {fail_csv}")

    return review_dir / "assets" / "png", fail_csv


def ensure_pair_assets(
    review_dir: Union[str, Path],
    manifest_csv: Union[str, Path],
    atlas_image: Union[str, Path],
    scan_uid_a: str,
    scan_uid_b: str,
    diff_rel_path: Optional[str] = None,
    checkerboard_rel_path: Optional[str] = None,
    overwrite: bool = False,
    winsor: Tuple[float, float] = (1.0, 99.0),
    hist_matching: bool = False,
    repro: bool = False,
    delete_extras: bool = True,
) -> Tuple[Path, Path]:
    review_dir = Path(review_dir).expanduser().resolve()
    manifest_csv = Path(manifest_csv).expanduser().resolve()
    atlas_image = Path(atlas_image).expanduser().resolve()

    if not atlas_image.exists():
        raise FileNotFoundError(f"atlas image not found: {atlas_image}")

    manifest_df = _read_manifest(manifest_csv)
    meta_map = _manifest_meta_map(manifest_df)

    a = str(scan_uid_a).strip()
    b = str(scan_uid_b).strip()
    if a not in meta_map:
        raise ValueError(f"scan_uid not found in manifest: {a}")
    if b not in meta_map:
        raise ValueError(f"scan_uid not found in manifest: {b}")

    if diff_rel_path is None:
        diff_rel_path = (Path("assets") / "diff" / f"{_pair_key(a, b)}.png").as_posix()
    if checkerboard_rel_path is None:
        checkerboard_rel_path = (
            Path("assets") / "checkerboard" / f"{_pair_key(a, b)}.png"
        ).as_posix()

    meta_a = meta_map[a]
    meta_b = meta_map[b]

    norm_a = _build_review_norm_cache(
        resolved_path=Path(meta_a["resolved_path"]),
        atlas_image=atlas_image,
        out_norm_path=_norm_cache_path(review_dir, a),
        work_dir=_work_dir(review_dir, a),
        winsor=winsor,
        hist_matching=hist_matching,
        repro=repro,
        delete_extras=delete_extras,
    )
    norm_b = _build_review_norm_cache(
        resolved_path=Path(meta_b["resolved_path"]),
        atlas_image=atlas_image,
        out_norm_path=_norm_cache_path(review_dir, b),
        work_dir=_work_dir(review_dir, b),
        winsor=winsor,
        hist_matching=hist_matching,
        repro=repro,
        delete_extras=delete_extras,
    )

    diff_out = (review_dir / diff_rel_path).resolve()
    cb_out = (review_dir / checkerboard_rel_path).resolve()

    render_pair_png(
        norm_path_a=norm_a,
        norm_path_b=norm_b,
        out_path=diff_out,
        kind="diff",
        overwrite=overwrite,
    )
    render_pair_png(
        norm_path_a=norm_a,
        norm_path_b=norm_b,
        out_path=cb_out,
        kind="checkerboard",
        overwrite=overwrite,
    )

    return diff_out, cb_out


def precompute_pair_assets(
    review_dir: Union[str, Path],
    manifest_csv: Union[str, Path],
    atlas_image: Union[str, Path],
    scan_level_review_csv: Union[str, Path],
    subject_level_review_csv: Union[str, Path],
    workers: int = 24,
    overwrite: bool = False,
    winsor: Tuple[float, float] = (1.0, 99.0),
    hist_matching: bool = False,
    repro: bool = False,
    delete_extras: bool = True,
) -> Tuple[Path, Path]:
    review_dir = Path(review_dir).expanduser().resolve()
    manifest_csv = Path(manifest_csv).expanduser().resolve()
    atlas_image = Path(atlas_image).expanduser().resolve()

    manifest_df = _read_manifest(manifest_csv)
    meta_map = _manifest_meta_map(manifest_df)

    pair_requests = collect_pair_requests(
        scan_level_review_csv=scan_level_review_csv,
        subject_level_review_csv=subject_level_review_csv,
    )

    missing_uids = sorted(
        [
            uid
            for rec in pair_requests.values()
            for uid in (rec["scan_uid_a"], rec["scan_uid_b"])
            if uid not in meta_map
        ]
    )
    if missing_uids:
        raise ValueError(
            f"scan_uids referenced in pair review tables not found in manifest: {missing_uids[:10]}"
        )

    fail_csv = review_dir / "pair_asset_failures.csv"

    recs: List[Dict[str, str]] = []
    for pkey, rec in sorted(pair_requests.items()):
        a = rec["scan_uid_a"]
        b = rec["scan_uid_b"]
        recs.append(
            {
                "pair_key": pkey,
                "scan_uid_a": a,
                "scan_uid_b": b,
                "resolved_path_a": meta_map[a]["resolved_path"],
                "resolved_path_b": meta_map[b]["resolved_path"],
                "norm_path_a": str(_norm_cache_path(review_dir, a)),
                "norm_path_b": str(_norm_cache_path(review_dir, b)),
                "work_dir_a": str(_work_dir(review_dir, a)),
                "work_dir_b": str(_work_dir(review_dir, b)),
                "diff_path": str((review_dir / rec["diff_path"]).resolve()),
                "checkerboard_path": str(
                    (review_dir / rec["checkerboard_path"]).resolve()
                ),
            }
        )

    failures_all: List[Dict[str, str]] = []
    ok_total = 0

    def _worker(rec: Dict[str, str]) -> Tuple[int, List[Dict[str, str]]]:
        failures: List[Dict[str, str]] = []
        ok = 0
        try:
            norm_a = _build_review_norm_cache(
                resolved_path=Path(rec["resolved_path_a"]),
                atlas_image=atlas_image,
                out_norm_path=Path(rec["norm_path_a"]),
                work_dir=Path(rec["work_dir_a"]),
                winsor=winsor,
                hist_matching=hist_matching,
                repro=repro,
                delete_extras=delete_extras,
            )
            norm_b = _build_review_norm_cache(
                resolved_path=Path(rec["resolved_path_b"]),
                atlas_image=atlas_image,
                out_norm_path=Path(rec["norm_path_b"]),
                work_dir=Path(rec["work_dir_b"]),
                winsor=winsor,
                hist_matching=hist_matching,
                repro=repro,
                delete_extras=delete_extras,
            )

            render_pair_png(
                norm_path_a=norm_a,
                norm_path_b=norm_b,
                out_path=Path(rec["diff_path"]),
                kind="diff",
                overwrite=overwrite,
            )
            render_pair_png(
                norm_path_a=norm_a,
                norm_path_b=norm_b,
                out_path=Path(rec["checkerboard_path"]),
                kind="checkerboard",
                overwrite=overwrite,
            )
            ok = 1
        except Exception as e:
            failures.append(
                {
                    "pair_key": rec["pair_key"],
                    "scan_uid_a": rec["scan_uid_a"],
                    "scan_uid_b": rec["scan_uid_b"],
                    "error": str(e),
                }
            )
        return ok, failures

    with ProcessPoolExecutor(max_workers=int(workers)) as ex:
        futs = [ex.submit(_worker, rec) for rec in recs]
        for fut in tqdm(
            as_completed(futs),
            total=len(futs),
            desc="review_pair_assets",
            unit="pair",
        ):
            ok, fails = fut.result()
            ok_total += int(ok)
            if fails:
                failures_all.extend(fails)

    pd.DataFrame(failures_all).to_csv(fail_csv, index=False)

    print("[REVIEW ASSETS] pair assets done")
    print(f"[REVIEW ASSETS] pairs ok: {ok_total}/{len(recs)}")
    print(f"[REVIEW ASSETS] pair failures: {len(failures_all)} -> {fail_csv}")

    return review_dir / "assets" / "diff", review_dir / "assets" / "checkerboard"


def build_review_assets(
    review_dir: Union[str, Path],
    manifest_csv: Union[str, Path],
    scan_level_review_csv: Union[str, Path],
    subject_level_review_csv: Union[str, Path],
    mode: str = "lazy",
    workers: int = 24,
    overwrite: bool = False,
    winsor: Tuple[float, float] = (1.0, 99.0),
    hist_matching: bool = False,
    repro: bool = False,
    delete_extras: bool = True,
) -> None:
    atlas_image = (
        Path(__file__).resolve().parents[2]
        / "resources"
        / "mni_1mm3_t1_brain_atlas.nii.gz"
    )
    mode = str(mode).strip().lower()
    if mode not in {"lazy", "precompute"}:
        raise ValueError("mode must be 'lazy' or 'precompute'")

    ensure_scan_assets(
        review_dir=review_dir,
        manifest_csv=manifest_csv,
        atlas_image=atlas_image,
        scan_level_review_csv=scan_level_review_csv,
        subject_level_review_csv=subject_level_review_csv,
        workers=workers,
        overwrite=overwrite,
        winsor=winsor,
        hist_matching=hist_matching,
        repro=repro,
        delete_extras=delete_extras,
    )

    if mode == "precompute":
        precompute_pair_assets(
            review_dir=review_dir,
            manifest_csv=manifest_csv,
            atlas_image=atlas_image,
            scan_level_review_csv=scan_level_review_csv,
            subject_level_review_csv=subject_level_review_csv,
            workers=workers,
            overwrite=overwrite,
            winsor=winsor,
            hist_matching=hist_matching,
            repro=repro,
            delete_extras=delete_extras,
        )

    print(f"[REVIEW ASSETS] mode={mode} done")
