"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

from __future__ import annotations

import shutil
import subprocess

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np

from deepbet import run_bet


def _run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout}")


def run_n4(in_path: Path, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return out_path

    _run(
        [
            "N4BiasFieldCorrection",
            "-d",
            "3",
            "-i",
            str(in_path),
            "-o",
            str(out_path),
        ]
    )
    return out_path


def run_deepbet_mask(
    in_path: Path,
    out_brain_path: Path,
    out_mask_path: Path,
) -> Tuple[Path, Path]:
    out_brain_path.parent.mkdir(parents=True, exist_ok=True)
    out_mask_path.parent.mkdir(parents=True, exist_ok=True)

    if out_brain_path.exists() and out_mask_path.exists():
        return out_brain_path, out_mask_path

    run_bet([str(in_path)], [str(out_brain_path)], [str(out_mask_path)])
    return out_brain_path, out_mask_path


def run_ants_reg_rigid(
    fixed: Path,
    moving: Path,
    out_warped: Path,
    out_mat: Path,
    hist_matching: bool = False,
    repro: bool = False,
    delete_extras: bool = True,
) -> Tuple[Path, Path]:
    out_warped.parent.mkdir(parents=True, exist_ok=True)
    out_mat.parent.mkdir(parents=True, exist_ok=True)

    if out_warped.exists() and out_mat.exists():
        return out_warped, out_mat

    out_prefix = out_warped.with_name("ants_rigid_")
    out_prefix_str = str(out_prefix)

    cmd = [
        "antsRegistrationSyN.sh",
        "-d",
        "3",
        "-f",
        str(fixed),
        "-m",
        str(moving),
        "-o",
        out_prefix_str,
        "-t",
        "r",
    ]
    if hist_matching:
        cmd += ["-j", "1"]
    if repro:
        cmd += ["-e", "114514", "-n", "1"]

    _run(cmd)

    produced = Path(out_prefix_str + "Warped.nii.gz")
    inv_warped = Path(out_prefix_str + "InverseWarped.nii.gz")
    mat = Path(out_prefix_str + "0GenericAffine.mat")

    if not produced.exists():
        raise FileNotFoundError(f"antsRegistrationSyN.sh did not produce: {produced}")
    if not mat.exists():
        raise FileNotFoundError(f"antsRegistrationSyN.sh did not produce: {mat}")

    produced.rename(out_warped)
    mat.rename(out_mat)

    if delete_extras:
        try:
            if inv_warped.exists():
                inv_warped.unlink()
        except Exception:
            pass

    return out_warped, out_mat


def apply_label_affine(
    moving_label: Path,
    fixed_img: Path,
    out_label: Path,
    affine_mat: Path,
) -> Path:
    out_label.parent.mkdir(parents=True, exist_ok=True)
    if out_label.exists():
        return out_label

    cmd = [
        "antsApplyTransforms",
        "-d",
        "3",
        "-i",
        str(moving_label),
        "-r",
        str(fixed_img),
        "-n",
        "NearestNeighbor",
        "-o",
        str(out_label),
        "-t",
        str(affine_mat),
    ]
    _run(cmd)

    if not out_label.exists():
        raise FileNotFoundError(f"antsApplyTransforms did not produce: {out_label}")

    return out_label


def normalize_t1_minmax(
    vol: np.ndarray,
    brain_mask: Optional[np.ndarray] = None,
    winsor: Tuple[float, float] = (1.0, 99.0),
    eps: float = 1e-6,
) -> np.ndarray:
    v = vol.astype(np.float32, copy=False)

    if brain_mask is None:
        mask = (v > 0).astype(np.uint8)
    else:
        mask = (brain_mask > 0).astype(np.uint8)

    vals = v[mask > 0].astype(np.float64)
    if vals.size == 0:
        return v.copy()

    p1, p99 = np.percentile(vals, winsor)
    if (not np.isfinite(p1)) or (not np.isfinite(p99)) or ((p99 - p1) <= eps):
        p1 = float(np.min(vals))
        p99 = float(np.max(vals))

    v_w = v.copy()
    if (p99 - p1) > eps:
        v_w[mask > 0] = np.clip(v_w[mask > 0], p1, p99)

    vals_w = v_w[mask > 0].astype(np.float64)
    vmin = float(np.min(vals_w))
    vmax = float(np.max(vals_w))
    scale = max(vmax - vmin, eps)

    out = np.zeros_like(v_w, dtype=np.float32)
    out[mask > 0] = ((v_w[mask > 0] - vmin) / scale).astype(np.float32, copy=False)
    out[mask > 0] = np.clip(out[mask > 0], 0.0, 1.0)

    return out


def _norm_reload(
    vol_norm: np.ndarray,
    affine: np.ndarray,
    header: nib.Nifti1Header,
) -> np.ndarray:
    hdr = header.copy()
    hdr.set_data_dtype(np.float32)

    img = nib.Nifti1Image(
        vol_norm.astype(np.float32, copy=False),
        affine,
        hdr,
    )
    img = nib.as_closest_canonical(img)

    arr = img.get_fdata(dtype=np.float32)
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.ndim != 3:
        raise ValueError(f"expected 3D nifti after squeeze, got shape={arr.shape}")

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(arr.astype(np.float32, copy=False))


def _center_index_from_nonzero(arr: np.ndarray, axis: int) -> int:
    nz = np.nonzero(arr)
    if len(nz[0]) == 0:
        return arr.shape[axis] // 2

    ax_vals = nz[axis].astype(np.int64)
    return int(np.clip(np.median(ax_vals), 0, arr.shape[axis] - 1))


def _crop_pad_center(vol: np.ndarray, target: Tuple[int, int, int]) -> np.ndarray:
    assert vol.ndim == 3

    Dx, Dy, Dz = vol.shape
    Tx, Ty, Tz = map(int, target)

    out = np.zeros((Tx, Ty, Tz), dtype=vol.dtype)

    sx = max((Dx - Tx) // 2, 0)
    ex = sx + min(Dx, Tx)
    sy = max((Dy - Ty) // 2, 0)
    ey = sy + min(Dy, Ty)
    sz = max((Dz - Tz) // 2, 0)
    ez = sz + min(Dz, Tz)

    tx = max((Tx - Dx) // 2, 0)
    tex = tx + min(Dx, Tx)
    ty = max((Ty - Dy) // 2, 0)
    tey = ty + min(Dy, Ty)
    tz = max((Tz - Dz) // 2, 0)
    tez = tz + min(Dz, Tz)

    out[tx:tex, ty:tey, tz:tez] = vol[sx:ex, sy:ey, sz:ez]
    return out


def _extract_2p5d(
    vol: np.ndarray,
    k: int,
    axis: int = 2,
    stride: int = 1,
    center_override: Optional[int] = None,
) -> np.ndarray:
    assert vol.ndim == 3

    k = int(k)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    stride = max(int(stride), 1)
    D = int(vol.shape[axis])
    k = min(k, D)

    if center_override is not None:
        c = int(np.clip(int(center_override), 0, D - 1))
    else:
        c = _center_index_from_nonzero(vol, axis)

    if k % 2 == 0:
        left = k // 2 - 1
        right = k // 2
    else:
        left = right = k // 2

    start = c - left * stride
    end = c + right * stride
    idxs = np.arange(start, end + 1, stride, dtype=int)

    if idxs.size < k:
        need = k - idxs.size
        extra = np.arange(end + stride, end + stride * (need + 1), stride, dtype=int)
        idxs = np.concatenate([idxs, extra])
    elif idxs.size > k:
        idxs = idxs[:k]

    idxs = np.clip(idxs, 0, D - 1)

    if axis == 0:
        stack = vol[idxs, :, :]
    elif axis == 1:
        stack = vol[:, idxs, :].transpose(1, 0, 2)
    else:
        stack = vol[:, :, idxs].transpose(2, 0, 1)

    if stack.shape[0] != k:
        raise RuntimeError(f"slice count mismatch: got {stack.shape[0]} vs k={k}")

    return np.ascontiguousarray(stack.astype(np.float32, copy=False))


def _make_work_paths(work_dir: Path) -> Dict[str, Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    return {
        "n4_path": work_dir / "n4.nii.gz",
        "masked_brain_path": work_dir / "masked_brain.nii.gz",
        "skull_mask_native_path": work_dir / "skull_mask_native.nii.gz",
        "reg_path": work_dir / "reg_rigid.nii.gz",
        "mat_path": work_dir / "rigid.mat",
        "skull_mask_reg_path": work_dir / "skull_mask_reg.nii.gz",
        "n4_brain_path": work_dir / "n4_brain.nii.gz",
    }


def preprocess_resolved_path_2p5d(
    resolved_path: Union[str, Path],
    work_dir: Union[str, Path],
    crop_shape: Tuple[int, int, int] = (160, 192, 160),
    axial_axis: int = 2,
    slices_2p5d: int = 32,
    slice_stride: int = 1,
    winsor: Tuple[float, float] = (1.0, 99.0),
    hist_matching: bool = False,
    repro: bool = False,
    delete_extras: bool = True,
    cleanup: bool = True,
) -> np.ndarray:
    resolved_path = Path(resolved_path).expanduser().resolve()
    atlas_image = (
        Path(__file__).resolve().parents[3]
        / "resources"
        / "mni_1mm3_t1_brain_atlas.nii.gz"
    )
    work_dir = Path(work_dir).expanduser().resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(f"input image not found: {resolved_path}")

    paths = _make_work_paths(work_dir)

    try:
        run_n4(resolved_path, paths["n4_path"])

        run_deepbet_mask(
            in_path=paths["n4_path"],
            out_brain_path=paths["masked_brain_path"],
            out_mask_path=paths["skull_mask_native_path"],
        )

        run_ants_reg_rigid(
            fixed=atlas_image,
            moving=paths["masked_brain_path"],
            out_warped=paths["reg_path"],
            out_mat=paths["mat_path"],
            hist_matching=hist_matching,
            repro=repro,
            delete_extras=delete_extras,
        )

        apply_label_affine(
            moving_label=paths["skull_mask_native_path"],
            fixed_img=atlas_image,
            out_label=paths["skull_mask_reg_path"],
            affine_mat=paths["mat_path"],
        )

        run_n4(paths["reg_path"], paths["n4_brain_path"])

        reg_img = nib.load(str(paths["n4_brain_path"]))
        vol = reg_img.get_fdata(dtype=np.float32)

        mask_img = nib.load(str(paths["skull_mask_reg_path"]))
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
        vol_norm = _norm_reload(
            vol_norm=vol_norm,
            affine=reg_img.affine,
            header=reg_img.header,
        )

        vol_norm = _crop_pad_center(vol_norm, crop_shape)
        c = _center_index_from_nonzero(vol_norm, axial_axis)

        stack = _extract_2p5d(
            vol=vol_norm,
            k=slices_2p5d,
            axis=axial_axis,
            stride=slice_stride,
            center_override=c,
        )

        # -> (1, K, H, W)
        return np.ascontiguousarray(stack[None, ...], dtype=np.float32)

    finally:
        if cleanup:
            shutil.rmtree(work_dir, ignore_errors=True)
