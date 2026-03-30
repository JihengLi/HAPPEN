"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

#!/usr/bin/env python3
from __future__ import annotations

import nibabel as nib
import numpy as np
import warnings

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.backends.backend_agg import FigureCanvasAgg

from pathlib import Path
from collections.abc import Sequence
from typing import List, Literal, Union, Optional

from .bids_path_finder import find_slant_addr_subjectid, find_t1w_addr_subjectid

lut_addr = "labels/slant.label"


def load_lut(lut_path: str, bg_transparent: bool = True):
    idx_list, rgba_list = [], []
    with open(lut_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            idx = int(parts[0])
            r, g, b = map(int, parts[1:4])
            a = float(parts[4])
            idx_list.append(idx)
            rgba_list.append((r / 255.0, g / 255.0, b / 255.0, a))

    max_idx = max(idx_list)
    full_rgba = [(0, 0, 0, 0)] * (max_idx + 1)
    for i, idx in enumerate(idx_list):
        full_rgba[idx] = rgba_list[i]

    if bg_transparent and 0 <= max_idx:
        full_rgba[0] = (0, 0, 0, 0)

    cmap = ListedColormap(full_rgba, name="slant_lut")
    bounds = np.arange(max_idx + 2) - 0.5
    norm = BoundaryNorm(boundaries=bounds, ncolors=cmap.N)
    return cmap, norm


def _normalize_slices(
    val: int | str | Sequence[int | str], size: int
) -> tuple[int, ...]:
    if isinstance(val, (str, int)):
        val = (val,)
    idxs = []
    for v in val:
        idx = size // 2 if v == "mid" else int(v)
        if not 0 <= idx < size:
            raise ValueError(f"Slice index {idx} out of bounds (0‥{size-1})")
        idxs.append(idx)
    return tuple(idxs)


def _keep_roi(arr: np.ndarray, roi: List) -> np.ndarray:
    mask = np.isin(arr, roi)
    out = np.where(mask, arr, 0)
    return out


def _load_as_ras(path, dtype=np.float32):
    img = nib.load(path, mmap=True)
    img_ras = nib.as_closest_canonical(img)
    data = np.asarray(img_ras.dataobj, dtype=dtype)
    zooms = img_ras.header.get_zooms()[:3]
    return data, *zooms, img_ras.affine


def _fig_to_array(fig, close: bool = True) -> np.ndarray:
    if fig.canvas is None or not isinstance(fig.canvas, FigureCanvasAgg):
        FigureCanvasAgg(fig)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    if close:
        plt.close(fig)
    return buf


def visualize_slant(
    seg_file: Union[str, Path],
    lut_file: Union[str, Path],
    sagittal_slices: Union[int, str, Sequence[Union[int, str]]] = "mid",
    coronal_slices: Union[int, str, Sequence[Union[int, str]]] = "mid",
    axial_slices: Union[int, str, Sequence[Union[int, str]]] = "mid",
    keep_roi_list: Optional[List] = None,
    auto_slice: bool = False,
    t1_file: Optional[Union[str, Path]] = None,
    alpha_seg: float = 0.6,
    save_path: Optional[Path] = None,
    show_img: bool = True,
) -> np.ndarray:
    seg_data, seg_sx, seg_sy, seg_sz, _ = _load_as_ras(seg_file)
    cmap, norm = load_lut(lut_file, bg_transparent=t1_file != None)

    if t1_file:
        t1_data, *_ = _load_as_ras(t1_file)
        if t1_data.shape != seg_data.shape:
            raise ValueError("T1 shape ≠ seg shape")
    else:
        t1_data = None
        alpha_seg = 1.0

    if auto_slice:
        if keep_roi_list is None:
            raise ValueError("auto_slice=True requires keep_roi_list to be provided")
        if sagittal_slices != "mid" or coronal_slices != "mid" or axial_slices != "mid":
            warnings.warn("auto_slice=True: provided slices ignored.", UserWarning)

        def _best_slice_along_axis(axis):
            """Return a dict {roi: best_idx} and a fallback idx (max area)."""
            best_for_roi = {}
            max_area_idx, max_area = -1, -1

            for i in range(seg_data.shape[axis]):
                if axis == 0:
                    slc = seg_data[i, :, :]
                elif axis == 1:
                    slc = seg_data[:, i, :]
                else:
                    slc = seg_data[:, :, i]

                roi_counts = {
                    roi: np.count_nonzero(slc == roi) for roi in keep_roi_list
                }
                total = sum(roi_counts.values())

                if total > max_area:
                    max_area, max_area_idx = total, i

                for roi, cnt in roi_counts.items():
                    if cnt == 0:
                        continue
                    if roi not in best_for_roi or cnt > best_for_roi[roi][1]:
                        best_for_roi[roi] = (i, cnt)

            best_for_roi = {roi: idx_cnt[0] for roi, idx_cnt in best_for_roi.items()}
            return best_for_roi, max_area_idx

        best_x_roi, fallback_x = _best_slice_along_axis(axis=0)
        best_y_roi, fallback_y = _best_slice_along_axis(axis=1)
        best_z_roi, fallback_z = _best_slice_along_axis(axis=2)

        def _pick(best_map, fallback):
            if len(best_map) == len(keep_roi_list):
                idx = max(
                    best_map.values(),
                    key=lambda i: (
                        np.count_nonzero(
                            np.isin(
                                (
                                    seg_data[i, :, :]
                                    if best_map is best_x_roi
                                    else (
                                        seg_data[:, i, :]
                                        if best_map is best_y_roi
                                        else seg_data[:, :, i]
                                    )
                                ),
                                keep_roi_list,
                            )
                        )
                    ),
                )
            elif best_map:
                counts = {
                    idx: np.count_nonzero(
                        np.isin(
                            (
                                seg_data[idx, :, :]
                                if best_map is best_x_roi
                                else (
                                    seg_data[:, idx, :]
                                    if best_map is best_y_roi
                                    else seg_data[:, :, idx]
                                )
                            ),
                            keep_roi_list,
                        )
                    )
                    for idx in best_map.values()
                }
                idx = max(counts, key=counts.get)
            else:
                idx = fallback
            return int(idx)

        best_x = _pick(best_x_roi, fallback_x)
        best_y = _pick(best_y_roi, fallback_y)
        best_z = _pick(best_z_roi, fallback_z)

        sagittal_slices = [best_x]
        coronal_slices = [best_y]
        axial_slices = [best_z]

    x_idxs = _normalize_slices(sagittal_slices, seg_data.shape[0])
    y_idxs = _normalize_slices(coronal_slices, seg_data.shape[1])
    z_idxs = _normalize_slices(axial_slices, seg_data.shape[2])

    combos = [(x, y, z) for x in x_idxs for y in y_idxs for z in z_idxs]
    n_rows = len(combos)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    axes = np.asarray(axes).reshape(n_rows, 3)

    for row_idx, (x, y, z) in enumerate(combos):
        slices_seg = (
            np.rot90(seg_data[x, :, :]),
            np.rot90(seg_data[:, y, :]),
            np.rot90(seg_data[:, :, z]),
        )
        if keep_roi_list:
            slices_seg = tuple(_keep_roi(slc, keep_roi_list) for slc in slices_seg)
        if t1_data is not None:
            slices_t1 = (
                np.rot90(t1_data[x, :, :]),
                np.rot90(t1_data[:, y, :]),
                np.rot90(t1_data[:, :, z]),
            )
        titles = (
            f"Sagittal X={x}",
            f"Coronal  Y={y}",
            f"Axial    Z={z}",
        )
        for col_idx, (seg_slc, title) in enumerate(zip(slices_seg, titles)):
            ax = axes[row_idx, col_idx]
            if t1_data is not None:
                ax.imshow(slices_t1[col_idx], cmap="gray", interpolation="nearest")
            ax.imshow(
                seg_slc, cmap=cmap, norm=norm, interpolation="nearest", alpha=alpha_seg
            )
            if col_idx == 0:
                aspect = seg_sz / seg_sy
            elif col_idx == 1:
                aspect = seg_sz / seg_sx
            else:
                aspect = seg_sy / seg_sx
            ax.set_aspect(aspect)
            ax.set_title(title, fontsize=9)
            ax.axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show_img:
        plt.show()
    return _fig_to_array(fig)


def visualize_slant_compare(
    seg_file_a: Union[str, Path],
    seg_file_b: Union[str, Path],
    t1_file_a: Optional[Union[str, Path]] = None,
    t1_file_b: Optional[Union[str, Path]] = None,
    sagittal_slices: Union[int, str, Sequence[Union[int, str]]] = "mid",
    coronal_slices: Union[int, str, Sequence[Union[int, str]]] = "mid",
    axial_slices: Union[int, str, Sequence[Union[int, str]]] = "mid",
    keep_roi_list: Optional[List] = None,
    auto_slice: bool = False,
    alpha_seg: float = 0.6,
    save_path: Optional[Path] = None,
    show_img: bool = True,
) -> np.ndarray:
    segA_data, segA_sx, segA_sy, segA_sz, _ = _load_as_ras(seg_file_a)
    segB_data, segB_sx, segB_sy, segB_sz, _ = _load_as_ras(seg_file_b)

    if segA_data.shape != segB_data.shape:
        raise ValueError("Segmentation volume shapes differ.")
    if not np.allclose((segA_sx, segA_sy, segA_sz), (segB_sx, segB_sy, segB_sz)):
        raise ValueError("Voxel spacing mismatch between A and B.")
    if (t1_file_a is None) != (t1_file_b is None):
        raise ValueError("Provide both T1 files or neither.")

    if t1_file_a and t1_file_b:
        t1A_data, *_ = _load_as_ras(t1_file_a)
        t1B_data, *_ = _load_as_ras(t1_file_b)
        if t1A_data.shape != segA_data.shape or t1B_data.shape != segA_data.shape:
            raise ValueError("T1 shape mismatch")
        t1_data = (t1A_data.astype(np.float32) + t1B_data.astype(np.float32)) / 2
    else:
        t1_data = None

    if keep_roi_list is None:
        raise ValueError("keep_roi_list must be set")
    maskA = np.isin(segA_data, keep_roi_list)
    maskB = np.isin(segB_data, keep_roi_list)

    diff_map = np.zeros(segA_data.shape, dtype=np.uint8)
    diff_map[maskA & maskB] = 1
    diff_map[maskA & ~maskB] = 2
    diff_map[~maskA & maskB] = 3

    if auto_slice:
        best_x = np.argmax(diff_map.sum(axis=(1, 2)))
        best_y = np.argmax(diff_map.sum(axis=(0, 2)))
        best_z = np.argmax(diff_map.sum(axis=(0, 1)))
        sagittal_slices, coronal_slices, axial_slices = [best_x], [best_y], [best_z]

    x_idx = _normalize_slices(sagittal_slices, segA_data.shape[0])
    y_idx = _normalize_slices(coronal_slices, segA_data.shape[1])
    z_idx = _normalize_slices(axial_slices, segA_data.shape[2])

    combos = [(x, y, z) for x in x_idx for y in y_idx for z in z_idx]
    fig, axes = plt.subplots(
        len(combos), 3, figsize=(15, 5 * len(combos)), squeeze=False
    )
    axes = np.asarray(axes)

    cmap = ListedColormap(
        [
            (0, 0, 0, 0),
            (77 / 255, 175 / 255, 74 / 255, alpha_seg),  # Common: green
            (215 / 255, 25 / 255, 28 / 255, alpha_seg),  # Only A: red
            (255 / 255, 215 / 255, 0 / 255, alpha_seg),  # Only B: yellow
        ]
    )

    orient_titles = ("Sagittal", "Coronal", "Axial")
    for row, (x, y, z) in enumerate(combos):
        diff_slices = (
            np.rot90(diff_map[x, :, :]),
            np.rot90(diff_map[:, y, :]),
            np.rot90(diff_map[:, :, z]),
        )
        if t1_data is not None:
            t1_slices = (
                np.rot90(t1_data[x, :, :]),
                np.rot90(t1_data[:, y, :]),
                np.rot90(t1_data[:, :, z]),
            )
        for col, (diff_slc, title_base) in enumerate(zip(diff_slices, orient_titles)):
            ax = axes[row, col]
            if t1_data is not None:
                ax.imshow(t1_slices[col], cmap="gray", interpolation="nearest")
            ax.imshow(diff_slc, cmap=cmap, interpolation="nearest", vmin=0, vmax=3)
            if col == 0:
                aspect = segA_sz / segA_sy
            elif col == 1:
                aspect = segA_sz / segA_sx
            else:
                aspect = segA_sy / segA_sx
            ax.set_aspect(aspect)
            ax.set_title(
                f"{title_base} ({['X','Y','Z'][col]}={[x,y,z][col]})", fontsize=9
            )
            ax.axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show_img:
        plt.show()
    return _fig_to_array(fig)


def visualize_slant_subjectid(
    subjectid: str,
    root: Union[str, Path],
    lut_file: Union[str, Path] = lut_addr,
    sagittal_slices: Union[int, str, Sequence[Union[int, str]]] = "mid",
    coronal_slices: Union[int, str, Sequence[Union[int, str]]] = "mid",
    axial_slices: Union[int, str, Sequence[Union[int, str]]] = "mid",
    keep_roi_list: Optional[List] = None,
    auto_slice: bool = False,
    bg_t1_file: bool = False,
    alpha_seg: float = 0.6,
    save_path: Optional[Path] = None,
    show_img: bool = True,
) -> np.ndarray:
    seg_file = find_slant_addr_subjectid(subjectid, Path(root))
    t1_file = None
    if bg_t1_file:
        t1_file = find_t1w_addr_subjectid(subjectid, Path(root).parent)
    return visualize_slant(
        seg_file,
        lut_file,
        sagittal_slices,
        coronal_slices,
        axial_slices,
        keep_roi_list,
        auto_slice,
        t1_file,
        alpha_seg,
        save_path,
        show_img,
    )


def visualize_slant_compare_subjectid(
    subjectid: str,
    root: Union[str, Path],
    sagittal_slices: Union[int, str, Sequence[Union[int, str]]] = "mid",
    coronal_slices: Union[int, str, Sequence[Union[int, str]]] = "mid",
    axial_slices: Union[int, str, Sequence[Union[int, str]]] = "mid",
    keep_roi_list: Optional[List] = None,
    auto_slice: bool = False,
    bg_t1_file: bool = False,
    alpha_seg: float = 0.6,
    save_path: Optional[Path] = None,
    show_img: bool = True,
) -> np.ndarray:
    seg_file_a = find_slant_addr_subjectid(subjectid + "_ses-00", Path(root))
    seg_file_b = find_slant_addr_subjectid(subjectid + "_ses-12", Path(root))

    t1_file_a, t1_file_b = None, None
    if bg_t1_file:
        t1_file_a = find_t1w_addr_subjectid(subjectid + "_ses-00", Path(root).parent)
        t1_file_b = find_t1w_addr_subjectid(subjectid + "_ses-12", Path(root).parent)

    return visualize_slant_compare(
        seg_file_a,
        seg_file_b,
        t1_file_a,
        t1_file_b,
        sagittal_slices,
        coronal_slices,
        axial_slices,
        keep_roi_list,
        auto_slice,
        alpha_seg,
        save_path,
        show_img,
    )


def visualize_t1w(
    t1_file: Union[str, Path],
    sagittal_slices: Union[int, str, Sequence[Union[int, str]]] = "mid",
    coronal_slices: Union[int, str, Sequence[Union[int, str]]] = "mid",
    axial_slices: Union[int, str, Sequence[Union[int, str]]] = "mid",
    perc: tuple[float, float] = (1, 99),
    save_path: Optional[Path] = None,
    show_img: bool = True,
) -> np.ndarray:
    t1_file = Path(t1_file)
    if not t1_file.exists():
        raise FileNotFoundError(t1_file)

    data_ras, sx, sy, sz, _ = _load_as_ras(t1_file)

    x_idxs = _normalize_slices(sagittal_slices, data_ras.shape[0])
    y_idxs = _normalize_slices(coronal_slices, data_ras.shape[1])
    z_idxs = _normalize_slices(axial_slices, data_ras.shape[2])

    combos = [(x, y, z) for x in x_idxs for y in y_idxs for z in z_idxs]
    n_rows = len(combos)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    axes = np.asarray(axes).reshape(n_rows, 3)
    vmin, vmax = np.percentile(data_ras, perc)

    for row_idx, (x, y, z) in enumerate(combos):
        slices = (
            np.rot90(data_ras[x, :, :]),
            np.rot90(data_ras[:, y, :]),
            np.rot90(data_ras[:, :, z]),
        )
        titles = (
            f"Sagittal X={x}",
            f"Coronal  Y={y}",
            f"Axial    Z={z}",
        )
        for col_idx, (slc, title) in enumerate(zip(slices, titles)):
            ax = axes[row_idx, col_idx]
            ax.imshow(slc, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
            if col_idx == 0:
                aspect = sz / sy
            elif col_idx == 1:
                aspect = sz / sx
            else:
                aspect = sy / sx
            ax.set_aspect(aspect)
            ax.set_title(title, fontsize=9)
            ax.axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show_img:
        plt.show()
    return _fig_to_array(fig)


def visualize_t1w_compare(
    t1_base: Union[str, Path],
    t1_fup: Union[str, Path],
    sagittal_slices: Union[int, str] = "mid",
    coronal_slices: Union[int, str] = "mid",
    axial_slices: Union[int, str] = "mid",
    perc: tuple[float, float] = (1, 99),
    diff_mode: Literal["diff", "pct", "zscore"] = "diff",
    robust_pct: float = 99,
    grid_step: int = 20,
    save_path: Optional[Path] = None,
    show_img: bool = True,
) -> np.ndarray:
    data0, sx, sy, sz, _ = _load_as_ras(Path(t1_base))
    data1, sx1, sy1, sz1, _ = _load_as_ras(Path(t1_fup))
    if data0.shape != data1.shape:
        raise ValueError("baseline & follow-up shapes differ")
    if not np.allclose((sx, sy, sz), (sx1, sy1, sz1), atol=1e-6):
        raise ValueError("voxel spacing mismatch")

    ix = _normalize_slices(sagittal_slices, data0.shape[0])[0]
    iy = _normalize_slices(coronal_slices, data0.shape[1])[0]
    iz = _normalize_slices(axial_slices, data0.shape[2])[0]

    sag0, sag1 = np.rot90(data0[ix, :, :]), np.rot90(data1[ix, :, :])
    cor0, cor1 = np.rot90(data0[:, iy, :]), np.rot90(data1[:, iy, :])
    axi0, axi1 = np.rot90(data0[:, :, iz]), np.rot90(data1[:, :, iz])

    def _calc_diff(a, b):
        if diff_mode == "diff":
            return b - a
        if diff_mode == "pct":
            denom = np.where(np.abs(a) < 1e-4, 1e-4, a)
            return (b - a) / denom
        if diff_mode == "zscore":
            z_a = (a - np.nanmean(a)) / (np.nanstd(a) + 1e-6)
            z_b = (b - np.nanmean(b)) / (np.nanstd(b) + 1e-6)
            return z_b - z_a
        raise ValueError(diff_mode)

    sag_d, cor_d, axi_d = (
        _calc_diff(sag0, sag1),
        _calc_diff(cor0, cor1),
        _calc_diff(axi0, axi1),
    )

    vmin_g, vmax_g = np.nanpercentile(
        np.concatenate([data0.ravel(), data1.ravel()]), perc
    )
    vmax_d = np.nanpercentile(
        np.abs(np.concatenate([sag_d.ravel(), cor_d.ravel(), axi_d.ravel()])),
        robust_pct,
    )

    fig, axes = plt.subplots(3, 3, figsize=(12, 12), dpi=500)
    view_labels = ["Sagittal", "Coronal", "Axial"]
    panels = [(sag0, sag1, sag_d), (cor0, cor1, cor_d), (axi0, axi1, axi_d)]

    for row_idx, (im0, im1, imd) in enumerate(panels):
        for col_idx, (img, title_suffix) in enumerate(
            [(im0, "Subject 1"), (im1, "Subject 2"), (imd, f"Δ ({diff_mode})")]
        ):
            ax = axes[row_idx, col_idx]
            if col_idx < 2:
                ax.imshow(
                    img, cmap="gray", vmin=vmin_g, vmax=vmax_g, interpolation="nearest"
                )
                ny, nx = img.shape
                ax.set_xticks(np.arange(0, nx, grid_step))
                ax.set_yticks(np.arange(0, ny, grid_step))
                ax.grid(True, color="yellow", linestyle="--", linewidth=0.5, zorder=1)
            else:
                ax.imshow(
                    img,
                    cmap="RdBu_r",
                    vmin=-vmax_d,
                    vmax=+vmax_d,
                    interpolation="nearest",
                )
                ax.set_xticks([])
                ax.set_yticks([])

            if row_idx == 0:
                aspect = sz / sy
            elif row_idx == 1:
                aspect = sz / sx
            else:
                aspect = sy / sx
            ax.set_aspect(aspect)

            ax.set_title(f"{view_labels[row_idx]} {title_suffix}")
            ax.tick_params(length=0, labelleft=False, labelbottom=False)

    plt.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    mappable = axes[0, 2].images[0]
    fig.colorbar(mappable, cax=cbar_ax).set_label(
        {
            "diff": "Intensity difference",
            "pct": "Percent change",
            "zscore": "Z-score difference",
        }[diff_mode]
    )

    if save_path:
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
    if show_img:
        plt.show()
    else:
        plt.close(fig)
    return _fig_to_array(fig)


def visualize_t1w_subjectid(
    subjectid: str,
    root: Union[str, Path],
    sagittal_slices: Union[int, str, Sequence[Union[int, str]]] = "mid",
    coronal_slices: Union[int, str, Sequence[Union[int, str]]] = "mid",
    axial_slices: Union[int, str, Sequence[Union[int, str]]] = "mid",
    perc: tuple[float, float] = (1, 99),
    save_path: Optional[Path] = None,
    show_img: bool = True,
) -> np.ndarray:
    t1_file = find_t1w_addr_subjectid(subjectid, Path(root).parent)
    return visualize_t1w(
        t1_file,
        sagittal_slices,
        coronal_slices,
        axial_slices,
        perc,
        save_path,
        show_img,
    )


def visualize_t1w_compare_subjectid(
    subjectid: str,
    root: Union[str, Path],
    sagittal_slices: Union[int, str, Sequence[Union[int, str]]] = "mid",
    coronal_slices: Union[int, str, Sequence[Union[int, str]]] = "mid",
    axial_slices: Union[int, str, Sequence[Union[int, str]]] = "mid",
    perc: tuple[float, float] = (1, 99),
    diff_mode: Literal["diff", "pct", "zscore"] = "diff",
    robust_pct: float = 99,
    grid_step: int = 20,
    save_path: Optional[Path] = None,
    show_img: bool = True,
) -> np.ndarray:
    t1_file_a = find_t1w_addr_subjectid(subjectid + "_ses-00", Path(root).parent)
    t1_file_b = find_t1w_addr_subjectid(subjectid + "_ses-12", Path(root).parent)
    return visualize_t1w_compare(
        t1_file_a,
        t1_file_b,
        sagittal_slices,
        coronal_slices,
        axial_slices,
        perc,
        diff_mode,
        robust_pct,
        grid_step,
        save_path,
        show_img,
    )


def visualize_ct(
    ct_file: Union[str, Path],
    sagittal_slices: Union[int, str, Sequence[Union[int, str]]] = "mid",
    coronal_slices: Union[int, str, Sequence[Union[int, str]]] = "mid",
    axial_slices: Union[int, str, Sequence[Union[int, str]]] = "mid",
    window: tuple[int, int] = (-1024, 1024),
    save_path: Union[str, Path] | None = None,
    show_img: bool = True,
) -> np.ndarray:
    ct_file = Path(ct_file)
    if not ct_file.exists():
        raise FileNotFoundError(f"CT file not found: {ct_file}")

    data_ras, sx, sy, sz, _ = _load_as_ras(ct_file)

    x_idxs = _normalize_slices(sagittal_slices, data_ras.shape[0])
    y_idxs = _normalize_slices(coronal_slices, data_ras.shape[1])
    z_idxs = _normalize_slices(axial_slices, data_ras.shape[2])
    combos = [(x, y, z) for x in x_idxs for y in y_idxs for z in z_idxs]
    n = len(combos)

    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
    axes = np.asarray(axes).reshape(n, 3)
    vmin, vmax = window

    orient_titles = ("Sagittal", "Coronal", "Axial")
    for i, (x, y, z) in enumerate(combos):
        sl_sag = np.rot90(data_ras[x, :, :])
        sl_cor = np.rot90(data_ras[:, y, :])
        sl_axi = np.rot90(data_ras[:, :, z])
        slices = (sl_sag, sl_cor, sl_axi)

        for j, slc in enumerate(slices):
            ax = axes[i, j]
            ax.imshow(slc, cmap="bone", vmin=vmin, vmax=vmax, interpolation="nearest")
            if j == 0:
                aspect = sz / sy
            elif j == 1:
                aspect = sz / sx
            else:
                aspect = sy / sx

            ax.set_aspect(aspect)
            ax.set_title(
                f"{orient_titles[j]} ({['X','Y','Z'][j]}={[x,y,z][j]})", fontsize=9
            )
            ax.axis("off")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show_img:
        plt.show()
    return _fig_to_array(fig)
