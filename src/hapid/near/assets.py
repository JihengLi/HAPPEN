"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

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
    apply_label_affine,
    normalize_t1_minmax,
    run_ants_reg_rigid,
    run_deepbet_mask,
    run_n4,
)


DEFAULT_FRACS: List[float] = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]
DEFAULT_PERC: Tuple[float, float] = (1.0, 99.0)
DEFAULT_DIFF_MAX: float = 64.0

SCAN_FAIL_COLS = ["scan_uid", "resolved_path", "png_path", "error"]
PAIR_FAIL_COLS = ["pair_key", "scan_uid_a", "scan_uid_b", "error"]


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


def _atomic_save_nifti(
    path: Path,
    data: np.ndarray,
    affine: np.ndarray,
    header: nib.Nifti1Header,
) -> None:
    path = Path(path).expanduser().resolve()
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


def _write_empty_failure_csv(path: Path, columns: Sequence[str]) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=list(columns)).to_csv(path, index=False)


def clear_review_cache(review_dir: Union[str, Path]) -> None:
    cache_dir = Path(review_dir).expanduser().resolve() / "cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def _pair_key(scan_uid_a: str, scan_uid_b: str) -> str:
    a = str(scan_uid_a).strip()
    b = str(scan_uid_b).strip()
    x, y = sorted([a, b])
    return f"{x}__{y}"


def _norm_cache_path(review_dir: Path, scan_uid: str) -> Path:
    return (
        Path(review_dir).expanduser().resolve()
        / "cache"
        / "norm"
        / f"{scan_uid}.nii.gz"
    )


def _work_dir(review_dir: Path, scan_uid: str) -> Path:
    return Path(review_dir).expanduser().resolve() / "cache" / "work" / scan_uid


def _scan_png_path(
    review_dir: Path,
    dataset: str,
    subject_id: str,
    session_id: str,
    scan_uid: str,
) -> Path:
    base = (
        Path(review_dir).expanduser().resolve()
        / "assets"
        / "png"
        / str(dataset).strip()
        / str(subject_id).strip()
    )
    ses = str(session_id).strip()
    if ses:
        base = base / ses
    return (base / f"{str(scan_uid).strip()}.png").resolve()


def _pair_diff_path(
    review_dir: Path,
    scan_uid_a: str,
    scan_uid_b: str,
) -> Path:
    return (
        Path(review_dir).expanduser().resolve()
        / "assets"
        / "diff"
        / f"{_pair_key(scan_uid_a, scan_uid_b)}.png"
    ).resolve()


def _pair_checkerboard_path(
    review_dir: Path,
    scan_uid_a: str,
    scan_uid_b: str,
) -> Path:
    return (
        Path(review_dir).expanduser().resolve()
        / "assets"
        / "checkerboard"
        / f"{_pair_key(scan_uid_a, scan_uid_b)}.png"
    ).resolve()


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
    cols_u8: List[np.ndarray] = []
    for c in columns:
        cc = to_uint8(c)
        if cc.ndim == 2:
            cc = np.stack([cc, cc, cc], axis=-1)
        cols_u8.append(cc)

    if not cols_u8:
        raise ValueError("columns must not be empty")

    heights = [c.shape[0] for c in cols_u8]
    widths = [c.shape[1] for c in cols_u8]
    H = max(heights)
    W = max(widths)

    out: List[np.ndarray] = []
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


def _load_nifti_shape_3d(nifti_path: Path) -> Tuple[int, int, int]:
    img = nib.load(str(nifti_path), mmap=True)
    img_ras = nib.as_closest_canonical(img)
    sx, sy, sz = img_ras.shape[:3]
    return int(sx), int(sy), int(sz)


def _frac_to_index(frac: float, length: int) -> int:
    if length <= 1:
        return 0
    idx = int(round(frac * (length - 1)))
    return max(0, min(length - 1, idx))


def build_percent_groups(nifti_path: Path, fracs: Sequence[float]) -> List[dict]:
    sx, sy, sz = _load_nifti_shape_3d(nifti_path)
    groups: List[dict] = []

    for f in fracs:
        groups.append(
            {
                "sagittal_slices": _frac_to_index(float(f), sx),
                "coronal_slices": _frac_to_index(float(f), sy),
                "axial_slices": _frac_to_index(float(f), sz),
            }
        )

    return groups


def _columns_for_groups(nifti_path: Path, groups: List[dict]) -> List[np.ndarray]:
    cols: List[np.ndarray] = []

    for g in groups:
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

    return cols


def _difference_tile(
    query_img: np.ndarray,
    candidate_img: np.ndarray,
    diff_max: float = DEFAULT_DIFF_MAX,
) -> np.ndarray:
    qq = to_uint8(query_img).astype(np.float32)
    cc = to_uint8(candidate_img).astype(np.float32)

    h = min(qq.shape[0], cc.shape[0])
    w = min(qq.shape[1], cc.shape[1])
    qq = qq[:h, :w]
    cc = cc[:h, :w]

    qv = qq[..., 0]
    cv = cc[..., 0]

    d = qv - cv
    t = np.clip(d / float(diff_max), -1.0, 1.0)

    out = np.empty((h, w, 3), dtype=np.float32)

    pos = t >= 0
    neg = ~pos

    out[..., 0] = 255.0
    out[..., 1] = 255.0
    out[..., 2] = 255.0

    out[pos, 1] = 255.0 * (1.0 - t[pos])
    out[pos, 2] = 255.0 * (1.0 - t[pos])

    u = 1.0 + t[neg]
    out[neg, 0] = 255.0 * u
    out[neg, 1] = 255.0 * u
    out[neg, 2] = 255.0

    return np.clip(out, 0, 255).astype(np.uint8)


def _checkerboard_tile(a: np.ndarray, b: np.ndarray, tile: int = 32) -> np.ndarray:
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
    diff_max: float = DEFAULT_DIFF_MAX,
) -> np.ndarray:
    tiles_a = split_mosaic_to_three(img_a)
    tiles_b = split_mosaic_to_three(img_b)

    out_tiles: List[np.ndarray] = []
    for ta, tb in zip(tiles_a, tiles_b):
        h = min(ta.shape[0], tb.shape[0])
        w = min(ta.shape[1], tb.shape[1])
        ta = ta[:h, :w]
        tb = tb[:h, :w]

        if kind == "diff":
            out_tiles.append(_difference_tile(ta, tb, diff_max=diff_max))
        elif kind == "checkerboard":
            out_tiles.append(_checkerboard_tile(ta, tb))
        else:
            raise ValueError(f"unknown pair kind: {kind}")

    min_w = min(t.shape[1] for t in out_tiles)
    out_tiles = [t[:, :min_w] for t in out_tiles]
    return np.vstack(out_tiles)


def _pair_columns_for_groups(
    nifti_path_a: Path,
    nifti_path_b: Path,
    groups: List[dict],
    kind: str,
    diff_max: float = DEFAULT_DIFF_MAX,
) -> List[np.ndarray]:
    cols: List[np.ndarray] = []

    for g in groups:
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
        cols.append(
            _pair_column_from_mosaics(
                img_a,
                img_b,
                kind=kind,
                diff_max=diff_max,
            )
        )

    return cols


def render_scan_png(
    norm_path: Path,
    out_path: Path,
    overwrite: bool,
) -> None:
    norm_path = Path(norm_path).expanduser().resolve()
    out_path = Path(out_path).expanduser().resolve()

    if not norm_path.exists():
        raise FileNotFoundError(f"norm_path not found: {norm_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        return

    groups = build_percent_groups(norm_path, fracs=DEFAULT_FRACS)
    cols = _columns_for_groups(norm_path, groups=groups)
    final_img = hstack_with_padding(cols, pad_value=0, gap=0)
    Image.fromarray(to_uint8(final_img)).save(out_path)


def render_pair_png(
    norm_path_a: Path,
    norm_path_b: Path,
    out_path: Path,
    kind: str,
    overwrite: bool,
    diff_max: float = DEFAULT_DIFF_MAX,
) -> None:
    norm_path_a = Path(norm_path_a).expanduser().resolve()
    norm_path_b = Path(norm_path_b).expanduser().resolve()
    out_path = Path(out_path).expanduser().resolve()

    if not norm_path_a.exists():
        raise FileNotFoundError(f"norm_path_a not found: {norm_path_a}")
    if not norm_path_b.exists():
        raise FileNotFoundError(f"norm_path_b not found: {norm_path_b}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        return

    groups = build_percent_groups(norm_path_a, fracs=DEFAULT_FRACS)
    cols = _pair_columns_for_groups(
        norm_path_a,
        norm_path_b,
        groups=groups,
        kind=kind,
        diff_max=diff_max,
    )
    final_img = hstack_with_padding(cols, pad_value=0, gap=0)
    Image.fromarray(to_uint8(final_img)).save(out_path)


def _read_manifest(manifest_csv: Union[str, Path]) -> pd.DataFrame:
    manifest_csv = Path(manifest_csv).expanduser().resolve()

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


def _manifest_meta_map(manifest_df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
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
    slots: List[int] = []
    for c in cols:
        cc = str(c).strip()
        if cc.startswith("candidate_") and cc.endswith("_dataset"):
            parts = cc.split("_")
            if len(parts) >= 3 and parts[1].isdigit():
                slots.append(int(parts[1]))
    return sorted(set(slots))


def _read_review_candidates(review_candidates_csv: Union[str, Path]) -> pd.DataFrame:
    review_candidates_csv = Path(review_candidates_csv).expanduser().resolve()
    df = pd.read_csv(review_candidates_csv, dtype=str).fillna("")
    df.columns = [str(c).strip() for c in df.columns]

    required_cols = [
        "query_dataset",
        "query_subject_id",
        "query_session_id",
        "query_scan_uid",
        "query_src_path",
        "n_candidates",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{review_candidates_csv} is missing required columns: {missing}"
        )

    return df


def _register_unique_scan_request(
    mapping: Dict[str, Dict[str, str]],
    scan_uid: str,
    dataset: str,
    subject_id: str,
    session_id: str,
    src_path: str,
) -> None:
    uid = clean_str(scan_uid)
    if not uid:
        return

    rec = {
        "dataset": clean_str(dataset),
        "subject_id": clean_str(subject_id),
        "session_id": clean_str(session_id),
        "src_path": clean_str(src_path),
    }

    if not rec["dataset"] or not rec["subject_id"] or not rec["src_path"]:
        raise ValueError(f"incomplete scan request for scan_uid={uid}: {rec}")

    if uid in mapping and mapping[uid] != rec:
        raise ValueError(
            f"inconsistent scan metadata for scan_uid={uid}: {mapping[uid]} vs {rec}"
        )

    mapping[uid] = rec


def collect_scan_requests(
    review_candidates_csv: Union[str, Path],
) -> Dict[str, Dict[str, str]]:
    df = _read_review_candidates(review_candidates_csv)
    slots = _candidate_slots_from_cols(df.columns.tolist())

    scan_uid_to_meta: Dict[str, Dict[str, str]] = {}

    for _, r in df.iterrows():
        _register_unique_scan_request(
            scan_uid_to_meta,
            scan_uid=str(r["query_scan_uid"]),
            dataset=str(r["query_dataset"]),
            subject_id=str(r["query_subject_id"]),
            session_id=str(r["query_session_id"]),
            src_path=str(r["query_src_path"]),
        )

        for k in slots:
            ds = clean_str(r.get(f"candidate_{k}_dataset", ""))
            if not ds:
                continue

            _register_unique_scan_request(
                scan_uid_to_meta,
                scan_uid=str(r.get(f"candidate_{k}_scan_uid", "")),
                dataset=str(r.get(f"candidate_{k}_dataset", "")),
                subject_id=str(r.get(f"candidate_{k}_subject_id", "")),
                session_id=str(r.get(f"candidate_{k}_session_id", "")),
                src_path=str(r.get(f"candidate_{k}_src_path", "")),
            )

    return scan_uid_to_meta


def collect_pair_requests(
    review_candidates_csv: Union[str, Path],
) -> Dict[str, Dict[str, str]]:
    df = _read_review_candidates(review_candidates_csv)
    slots = _candidate_slots_from_cols(df.columns.tolist())

    pair_map: Dict[str, Dict[str, str]] = {}

    for _, r in df.iterrows():
        query_scan_uid = clean_str(r.get("query_scan_uid", ""))
        if not query_scan_uid:
            continue

        for k in slots:
            cand_dataset = clean_str(r.get(f"candidate_{k}_dataset", ""))
            if not cand_dataset:
                continue

            cand_scan_uid = clean_str(r.get(f"candidate_{k}_scan_uid", ""))
            if not cand_scan_uid:
                raise ValueError(
                    f"missing candidate_{k}_scan_uid for query_scan_uid={query_scan_uid}"
                )

            pkey = _pair_key(query_scan_uid, cand_scan_uid)
            rec = {
                "scan_uid_a": min(query_scan_uid, cand_scan_uid),
                "scan_uid_b": max(query_scan_uid, cand_scan_uid),
            }

            if pkey in pair_map and pair_map[pkey] != rec:
                raise ValueError(f"inconsistent pair registration for {pkey}")

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
    overwrite: bool = False,
) -> Path:
    resolved_path = Path(resolved_path).expanduser().resolve()
    atlas_image = Path(atlas_image).expanduser().resolve()
    out_norm_path = Path(out_norm_path).expanduser().resolve()
    work_dir = Path(work_dir).expanduser().resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(f"input image not found: {resolved_path}")
    if not atlas_image.exists():
        raise FileNotFoundError(f"atlas image not found: {atlas_image}")

    if out_norm_path.exists() and not overwrite:
        return out_norm_path

    if overwrite and out_norm_path.exists():
        out_norm_path.unlink()

    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_norm_path.parent.mkdir(parents=True, exist_ok=True)

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


def _scan_asset_worker(
    rec: Dict[str, str],
    atlas_image: str,
    overwrite: bool,
    winsor: Tuple[float, float],
    hist_matching: bool,
    repro: bool,
    delete_extras: bool,
) -> int:
    norm_path = _build_review_norm_cache(
        resolved_path=Path(rec["resolved_path"]),
        atlas_image=Path(atlas_image),
        out_norm_path=Path(rec["norm_path"]),
        work_dir=Path(rec["work_dir"]),
        winsor=winsor,
        hist_matching=hist_matching,
        repro=repro,
        delete_extras=delete_extras,
        overwrite=overwrite,
    )

    render_scan_png(
        norm_path=norm_path,
        out_path=Path(rec["png_path"]),
        overwrite=overwrite,
    )
    return 1


def _norm_cache_worker(
    rec: Dict[str, str],
    atlas_image: str,
    overwrite: bool,
    winsor: Tuple[float, float],
    hist_matching: bool,
    repro: bool,
    delete_extras: bool,
) -> int:
    _build_review_norm_cache(
        resolved_path=Path(rec["resolved_path"]),
        atlas_image=Path(atlas_image),
        out_norm_path=Path(rec["norm_path"]),
        work_dir=Path(rec["work_dir"]),
        winsor=winsor,
        hist_matching=hist_matching,
        repro=repro,
        delete_extras=delete_extras,
        overwrite=overwrite,
    )
    return 1


def _pair_asset_worker(
    rec: Dict[str, str],
    overwrite: bool,
    diff_max: float,
) -> int:
    norm_path_a = Path(rec["norm_path_a"]).expanduser().resolve()
    norm_path_b = Path(rec["norm_path_b"]).expanduser().resolve()

    if not norm_path_a.exists():
        raise FileNotFoundError(f"norm_path_a not found: {norm_path_a}")
    if not norm_path_b.exists():
        raise FileNotFoundError(f"norm_path_b not found: {norm_path_b}")

    render_pair_png(
        norm_path_a=norm_path_a,
        norm_path_b=norm_path_b,
        out_path=Path(rec["diff_path"]),
        kind="diff",
        overwrite=overwrite,
        diff_max=diff_max,
    )
    render_pair_png(
        norm_path_a=norm_path_a,
        norm_path_b=norm_path_b,
        out_path=Path(rec["checkerboard_path"]),
        kind="checkerboard",
        overwrite=overwrite,
        diff_max=diff_max,
    )
    return 1


def _run_parallel_futures(
    futures: List[Any],
    desc: str,
    unit: str,
) -> int:
    ok_total = 0
    for fut in tqdm(as_completed(futures), total=len(futures), desc=desc, unit=unit):
        ok_total += int(fut.result())
    return ok_total


def _precompute_pair_norm_caches(
    review_dir: Path,
    atlas_image: Path,
    meta_map: Dict[str, Dict[str, str]],
    pair_requests: Dict[str, Dict[str, str]],
    workers: int,
    overwrite: bool,
    winsor: Tuple[float, float],
    hist_matching: bool,
    repro: bool,
    delete_extras: bool,
) -> int:
    needed_scan_uids = sorted(
        {rec["scan_uid_a"] for rec in pair_requests.values()}
        | {rec["scan_uid_b"] for rec in pair_requests.values()}
    )

    recs: List[Dict[str, str]] = []
    for scan_uid in needed_scan_uids:
        meta = meta_map[scan_uid]
        recs.append(
            {
                "scan_uid": scan_uid,
                "resolved_path": meta["resolved_path"],
                "norm_path": str(_norm_cache_path(review_dir, scan_uid)),
                "work_dir": str(_work_dir(review_dir, scan_uid)),
            }
        )

    if not recs:
        return 0

    with ProcessPoolExecutor(
        max_workers=int(workers),
        mp_context=get_context("spawn"),
    ) as ex:
        futures = [
            ex.submit(
                _norm_cache_worker,
                rec,
                str(atlas_image),
                bool(overwrite),
                winsor,
                bool(hist_matching),
                bool(repro),
                bool(delete_extras),
            )
            for rec in recs
        ]
        return _run_parallel_futures(
            futures,
            desc="review_pair_norm_cache",
            unit="scan",
        )


def ensure_scan_assets(
    review_dir: Union[str, Path],
    manifest_csv: Union[str, Path],
    atlas_image: Union[str, Path],
    review_candidates_csv: Union[str, Path],
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
    manifest_meta = _manifest_meta_map(manifest_df)
    requested_meta = collect_scan_requests(review_candidates_csv)

    missing_uids = sorted(uid for uid in requested_meta if uid not in manifest_meta)
    if missing_uids:
        raise ValueError(
            "scan_uids referenced in review_candidates.csv not found in manifest: "
            f"{missing_uids[:10]}"
        )

    recs: List[Dict[str, str]] = []
    for scan_uid, req in sorted(requested_meta.items()):
        man = manifest_meta[scan_uid]

        if req["dataset"] != man["dataset"]:
            raise ValueError(
                f"dataset mismatch for scan_uid={scan_uid}: "
                f"review={req['dataset']} vs manifest={man['dataset']}"
            )
        if req["subject_id"] != man["subject_id"]:
            raise ValueError(
                f"subject_id mismatch for scan_uid={scan_uid}: "
                f"review={req['subject_id']} vs manifest={man['subject_id']}"
            )
        if req["session_id"] != man["session_id"]:
            raise ValueError(
                f"session_id mismatch for scan_uid={scan_uid}: "
                f"review={req['session_id']} vs manifest={man['session_id']}"
            )
        if req["src_path"] and req["src_path"] != man["resolved_path"]:
            raise ValueError(
                f"src_path mismatch for scan_uid={scan_uid}: "
                f"review={req['src_path']} vs manifest={man['resolved_path']}"
            )

        recs.append(
            {
                "scan_uid": scan_uid,
                "resolved_path": man["resolved_path"],
                "png_path": str(
                    _scan_png_path(
                        review_dir=review_dir,
                        dataset=man["dataset"],
                        subject_id=man["subject_id"],
                        session_id=man["session_id"],
                        scan_uid=scan_uid,
                    )
                ),
                "norm_path": str(_norm_cache_path(review_dir, scan_uid)),
                "work_dir": str(_work_dir(review_dir, scan_uid)),
            }
        )

    png_dir = review_dir / "assets" / "png"
    png_dir.mkdir(parents=True, exist_ok=True)

    if recs:
        with ProcessPoolExecutor(
            max_workers=int(workers),
            mp_context=get_context("spawn"),
        ) as ex:
            futures = [
                ex.submit(
                    _scan_asset_worker,
                    rec,
                    str(atlas_image),
                    bool(overwrite),
                    winsor,
                    bool(hist_matching),
                    bool(repro),
                    bool(delete_extras),
                )
                for rec in recs
            ]
            ok_total = _run_parallel_futures(
                futures,
                desc="review_scan_assets",
                unit="scan",
            )
    else:
        ok_total = 0

    fail_csv = review_dir / "scan_asset_failures.csv"
    _write_empty_failure_csv(fail_csv, SCAN_FAIL_COLS)

    print("[REVIEW ASSETS] scan assets done")
    print(f"[REVIEW ASSETS] scans ok: {ok_total}/{len(recs)}")
    print(f"[REVIEW ASSETS] scan failures: 0 -> {fail_csv}")

    return png_dir, fail_csv


def ensure_pair_assets(
    review_dir: Union[str, Path],
    manifest_csv: Union[str, Path],
    atlas_image: Union[str, Path],
    scan_uid_a: str,
    scan_uid_b: str,
    overwrite: bool = False,
    winsor: Tuple[float, float] = (1.0, 99.0),
    hist_matching: bool = False,
    repro: bool = False,
    delete_extras: bool = True,
    diff_max: float = DEFAULT_DIFF_MAX,
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
        overwrite=overwrite,
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
        overwrite=overwrite,
    )

    diff_out = _pair_diff_path(review_dir, a, b)
    cb_out = _pair_checkerboard_path(review_dir, a, b)

    render_pair_png(
        norm_path_a=norm_a,
        norm_path_b=norm_b,
        out_path=diff_out,
        kind="diff",
        overwrite=overwrite,
        diff_max=diff_max,
    )
    render_pair_png(
        norm_path_a=norm_a,
        norm_path_b=norm_b,
        out_path=cb_out,
        kind="checkerboard",
        overwrite=overwrite,
        diff_max=diff_max,
    )

    return diff_out, cb_out


def precompute_pair_assets(
    review_dir: Union[str, Path],
    manifest_csv: Union[str, Path],
    atlas_image: Union[str, Path],
    review_candidates_csv: Union[str, Path],
    workers: int = 24,
    overwrite: bool = False,
    winsor: Tuple[float, float] = (1.0, 99.0),
    hist_matching: bool = False,
    repro: bool = False,
    delete_extras: bool = True,
    diff_max: float = DEFAULT_DIFF_MAX,
) -> Tuple[Path, Path]:
    review_dir = Path(review_dir).expanduser().resolve()
    manifest_csv = Path(manifest_csv).expanduser().resolve()
    atlas_image = Path(atlas_image).expanduser().resolve()

    if not atlas_image.exists():
        raise FileNotFoundError(f"atlas image not found: {atlas_image}")

    manifest_df = _read_manifest(manifest_csv)
    meta_map = _manifest_meta_map(manifest_df)
    pair_requests = collect_pair_requests(review_candidates_csv)

    missing_uids = sorted(
        {
            uid
            for rec in pair_requests.values()
            for uid in (rec["scan_uid_a"], rec["scan_uid_b"])
            if uid not in meta_map
        }
    )
    if missing_uids:
        raise ValueError(
            "scan_uids referenced in review_candidates.csv not found in manifest: "
            f"{missing_uids[:10]}"
        )

    norm_ok_total = _precompute_pair_norm_caches(
        review_dir=review_dir,
        atlas_image=atlas_image,
        meta_map=meta_map,
        pair_requests=pair_requests,
        workers=workers,
        overwrite=overwrite,
        winsor=winsor,
        hist_matching=hist_matching,
        repro=repro,
        delete_extras=delete_extras,
    )

    recs: List[Dict[str, str]] = []
    for pkey, rec in sorted(pair_requests.items()):
        a = rec["scan_uid_a"]
        b = rec["scan_uid_b"]
        recs.append(
            {
                "pair_key": pkey,
                "scan_uid_a": a,
                "scan_uid_b": b,
                "norm_path_a": str(_norm_cache_path(review_dir, a)),
                "norm_path_b": str(_norm_cache_path(review_dir, b)),
                "diff_path": str(_pair_diff_path(review_dir, a, b)),
                "checkerboard_path": str(_pair_checkerboard_path(review_dir, a, b)),
            }
        )

    diff_dir = review_dir / "assets" / "diff"
    checkerboard_dir = review_dir / "assets" / "checkerboard"
    diff_dir.mkdir(parents=True, exist_ok=True)
    checkerboard_dir.mkdir(parents=True, exist_ok=True)

    if recs:
        with ProcessPoolExecutor(
            max_workers=int(workers),
            mp_context=get_context("spawn"),
        ) as ex:
            futures = [
                ex.submit(
                    _pair_asset_worker,
                    rec,
                    bool(overwrite),
                    float(diff_max),
                )
                for rec in recs
            ]
            ok_total = _run_parallel_futures(
                futures,
                desc="review_pair_assets",
                unit="pair",
            )
    else:
        ok_total = 0

    fail_csv = review_dir / "pair_asset_failures.csv"
    _write_empty_failure_csv(fail_csv, PAIR_FAIL_COLS)

    print("[REVIEW ASSETS] pair norm caches done")
    print(f"[REVIEW ASSETS] pair norm caches ok: {norm_ok_total}")
    print("[REVIEW ASSETS] pair assets done")
    print(f"[REVIEW ASSETS] pairs ok: {ok_total}/{len(recs)}")
    print(f"[REVIEW ASSETS] pair failures: 0 -> {fail_csv}")

    return diff_dir, checkerboard_dir


def build_review_assets(
    review_dir: Union[str, Path],
    manifest_csv: Union[str, Path],
    review_candidates_csv: Union[str, Path],
    mode: str = "lazy",
    workers: int = 24,
    overwrite: bool = False,
    winsor: Tuple[float, float] = (1.0, 99.0),
    hist_matching: bool = False,
    repro: bool = False,
    delete_extras: bool = True,
    diff_max: float = DEFAULT_DIFF_MAX,
    cleanup_cache: bool = True,
) -> None:
    review_dir = Path(review_dir).expanduser().resolve()

    atlas_image = (
        Path(__file__).resolve().parents[3]
        / "resources"
        / "mni_1mm3_t1_brain_atlas.nii.gz"
    )

    mode = str(mode).strip().lower()
    if mode not in {"lazy", "precompute"}:
        raise ValueError("mode must be 'lazy' or 'precompute'")

    try:
        ensure_scan_assets(
            review_dir=review_dir,
            manifest_csv=manifest_csv,
            atlas_image=atlas_image,
            review_candidates_csv=review_candidates_csv,
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
                review_candidates_csv=review_candidates_csv,
                workers=workers,
                overwrite=overwrite,
                winsor=winsor,
                hist_matching=hist_matching,
                repro=repro,
                delete_extras=delete_extras,
                diff_max=diff_max,
            )
        else:
            diff_dir = review_dir / "assets" / "diff"
            checkerboard_dir = review_dir / "assets" / "checkerboard"
            diff_dir.mkdir(parents=True, exist_ok=True)
            checkerboard_dir.mkdir(parents=True, exist_ok=True)

            pair_fail_csv = review_dir / "pair_asset_failures.csv"
            _write_empty_failure_csv(pair_fail_csv, PAIR_FAIL_COLS)

            print("[REVIEW ASSETS] pair assets skipped (mode=lazy)")
            print(f"[REVIEW ASSETS] pair failures: 0 -> {pair_fail_csv}")

        print(f"[REVIEW ASSETS] mode={mode} done")

    finally:
        if cleanup_cache:
            clear_review_cache(review_dir)
            print(f"[REVIEW ASSETS] cache cleared: {review_dir / 'cache'}")
