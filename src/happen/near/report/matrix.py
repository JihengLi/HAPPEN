"""
Author: Jiheng Li
Email: jiheng.li.1@Vanderbilt.Edu
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.offline import plot as plot_html

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from multiprocessing import get_context


REQUIRED_COLS = [
    "group_id",
    "dataset",
    "subject_id",
    "subject_key",
]


def build_sorted_subject_index(
    duplicates_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    subjects = (
        duplicates_df[["dataset", "subject_id", "subject_key"]]
        .drop_duplicates(subset=["subject_key"])
        .copy()
    )

    dup_count_per_ds = (
        duplicates_df.groupby("dataset")["subject_key"].nunique().astype(int)
    )

    all_ds = subjects["dataset"].astype(str).unique()
    for ds in all_ds:
        if ds not in dup_count_per_ds.index:
            dup_count_per_ds.loc[ds] = 0

    ds_order_df = (
        dup_count_per_ds.rename("dup_count")
        .reset_index()
        .assign(dataset_key=lambda d: d["dataset"].astype(str).str.lower())
        .sort_values(["dup_count", "dataset_key"], ascending=[True, True])
        .reset_index(drop=True)
    )
    dataset_rank_map = {ds: rank for rank, ds in enumerate(ds_order_df["dataset"])}

    subjects["dataset_rank"] = subjects["dataset"].map(dataset_rank_map).astype(int)

    def natural_int(s: str, default: int = 10**9) -> int:
        if not isinstance(s, str):
            return default
        m = re.search(r"\d+", s)
        return int(m.group()) if m else default

    subjects["subject_num"] = subjects["subject_id"].astype(str).map(natural_int)

    subjects = subjects.sort_values(
        by=[
            "dataset_rank",
            "subject_num",
            "subject_id",
            "subject_key",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    subjects["subject_idx"] = np.arange(len(subjects), dtype=int)
    idx_of = dict(zip(subjects["subject_key"], subjects["subject_idx"]))
    return subjects, idx_of


def _collect_points_from_group_arrays(
    subject_idx: np.ndarray,
    ds_code: np.ndarray,
    ds_str: np.ndarray,
    sub_str: np.ndarray,
) -> Dict[str, Dict[str, List[Any]]]:
    out: Dict[str, Dict[str, List[Any]]] = {
        "within_dataset": {"x": [], "y": [], "custom": []},
        "across_datasets": {"x": [], "y": [], "custom": []},
    }

    k = subject_idx.shape[0]
    if k < 2:
        return out

    I, J = np.triu_indices(k, 1)

    ia, ib = subject_idx[I], subject_idx[J]
    ia, ib = np.minimum(ia, ib), np.maximum(ia, ib)

    same_ds = ds_code[I] == ds_code[J]

    masks = {
        "within_dataset": same_ds,
        "across_datasets": ~same_ds,
    }

    for cat, m in masks.items():
        if not np.any(m):
            continue

        xs = ia[m].tolist()
        ys = ib[m].tolist()

        out[cat]["x"].extend(xs)
        out[cat]["y"].extend(ys)

        dsa = ds_str[I][m]
        suba = sub_str[I][m]
        dsb = ds_str[J][m]
        subb = sub_str[J][m]

        cat_label = cat.replace("_", " ")

        out[cat]["custom"].extend(
            [
                (
                    int(x),
                    int(y),
                    str(cat_label),
                    str(dsa_i),
                    str(suba_i),
                    str(dsb_i),
                    str(subb_i),
                )
                for x, y, dsa_i, suba_i, dsb_i, subb_i in zip(
                    xs,
                    ys,
                    dsa.tolist(),
                    suba.tolist(),
                    dsb.tolist(),
                    subb.tolist(),
                )
            ]
        )

    return out


def _worker_process_chunk(
    chunk: List[Dict[str, Any]],
) -> Dict[str, Dict[str, List[Any]]]:
    out: Dict[str, Dict[str, List[Any]]] = {
        "within_dataset": {"x": [], "y": [], "custom": []},
        "across_datasets": {"x": [], "y": [], "custom": []},
    }

    for item in chunk:
        res = _collect_points_from_group_arrays(
            item["subject_idx"],
            item["ds_code"],
            item["ds_str"],
            item["sub_str"],
        )
        for cat in out.keys():
            out[cat]["x"].extend(res[cat]["x"])
            out[cat]["y"].extend(res[cat]["y"])
            out[cat]["custom"].extend(res[cat]["custom"])

    return out


def collect_duplicate_points_parallel(
    duplicates_df: pd.DataFrame,
    subjects: pd.DataFrame,
    n_procs: int,
) -> Dict[str, Dict[str, List[Any]]]:
    duplicates_df = duplicates_df.copy()

    missing = [c for c in REQUIRED_COLS if c not in duplicates_df.columns]
    if missing:
        raise ValueError(f"duplicates_df missing required columns: {missing}")

    duplicates_df = duplicates_df[
        ["group_id", "dataset", "subject_id", "subject_key"]
    ].copy()

    duplicates_df = duplicates_df.drop_duplicates(
        subset=["group_id", "subject_key"]
    ).reset_index(drop=True)

    duplicates_df = duplicates_df.merge(
        subjects[["subject_key", "subject_idx"]],
        on="subject_key",
        how="left",
        validate="m:1",
    )

    duplicates_df["ds_code"] = pd.factorize(duplicates_df["dataset"].astype(str))[
        0
    ].astype(np.int32)

    if n_procs <= 1:
        out: Dict[str, Dict[str, List[Any]]] = {
            "within_dataset": {"x": [], "y": [], "custom": []},
            "across_datasets": {"x": [], "y": [], "custom": []},
        }

        for _, g in duplicates_df.groupby("group_id", sort=False):
            pack = {
                "subject_idx": g["subject_idx"].to_numpy(np.int64, copy=False),
                "ds_code": g["ds_code"].to_numpy(np.int32, copy=False),
                "ds_str": g["dataset"]
                .astype("string")
                .fillna("")
                .to_numpy(object, copy=False),
                "sub_str": g["subject_id"]
                .astype("string")
                .fillna("")
                .to_numpy(object, copy=False),
            }
            res = _collect_points_from_group_arrays(**pack)
            for cat in out.keys():
                out[cat]["x"].extend(res[cat]["x"])
                out[cat]["y"].extend(res[cat]["y"])
                out[cat]["custom"].extend(res[cat]["custom"])
        return out

    group_list: List[Tuple[Any, int]] = []
    sizes: List[int] = []

    for group_id, g in duplicates_df.groupby("group_id", sort=False):
        k = int(g.shape[0])
        pairs = k * (k - 1) // 2
        if pairs <= 0:
            continue
        group_list.append((group_id, k))
        sizes.append(pairs)

    order = np.argsort(np.array(sizes))[::-1]
    bins: List[List[Any]] = [[] for _ in range(n_procs)]
    bin_loads = [0] * n_procs

    duplicates_df = duplicates_df.reset_index(drop=True)
    group_indices: Dict[Any, np.ndarray] = {
        ch: idx.values
        for ch, idx in duplicates_df.groupby("group_id", sort=False).groups.items()
    }

    for idx in order:
        group_id, _ = group_list[idx]
        pairs = sizes[idx]
        b = int(np.argmin(bin_loads))
        bin_loads[b] += pairs

        rows = group_indices[group_id]
        g = duplicates_df.loc[rows]

        bins[b].append(
            {
                "subject_idx": g["subject_idx"].to_numpy(np.int64, copy=False),
                "ds_code": g["ds_code"].to_numpy(np.int32, copy=False),
                "ds_str": g["dataset"]
                .astype("string")
                .fillna("")
                .to_numpy(object, copy=False),
                "sub_str": g["subject_id"]
                .astype("string")
                .fillna("")
                .to_numpy(object, copy=False),
            }
        )

    with get_context("spawn").Pool(processes=n_procs) as pool:
        results = pool.map(_worker_process_chunk, bins)

    out: Dict[str, Dict[str, List[Any]]] = {
        "within_dataset": {"x": [], "y": [], "custom": []},
        "across_datasets": {"x": [], "y": [], "custom": []},
    }

    for res in results:
        for cat in out.keys():
            out[cat]["x"].extend(res[cat]["x"])
            out[cat]["y"].extend(res[cat]["y"])
            out[cat]["custom"].extend(res[cat]["custom"])

    return out


def compute_blocks(subjects: pd.DataFrame, by: str) -> list[dict[str, Any]]:
    vals = subjects[by].astype(str).tolist()
    blocks: list[dict[str, Any]] = []
    if not vals:
        return blocks

    start = 0
    last = vals[0]
    for i, val in enumerate(vals[1:], start=1):
        if val != last:
            blocks.append({"start": start, "end": i - 1, "value": last})
            start = i
            last = val
    blocks.append({"start": start, "end": len(vals) - 1, "value": last})
    return blocks


def add_ruler_shapes_and_labels(
    fig: go.Figure,
    blocks: list[dict[str, Any]],
    y0: float,
    y1: float,
    fillcolor: str,
    text_color: str,
    show_text: bool,
    min_block_len_for_text: int,
):
    for blk in blocks:
        x0 = blk["start"] - 0.5
        x1 = blk["end"] + 0.5
        fig.add_shape(
            type="rect",
            xref="x",
            x0=x0,
            x1=x1,
            yref="paper",
            y0=y0,
            y1=y1,
            line=dict(color=fillcolor, width=0.5),
            fillcolor=fillcolor,
            opacity=0.18,
            layer="above",
        )
        if show_text and (blk["end"] - blk["start"] + 1) >= min_block_len_for_text:
            cx = (blk["start"] + blk["end"]) / 2
            fig.add_annotation(
                xref="x",
                x=cx,
                yref="paper",
                y=(y0 + y1) / 2,
                text=str(blk["value"]),
                showarrow=False,
                font=dict(size=11, color=text_color),
                align="center",
                yanchor="middle",
            )


def add_divider_lines(
    fig: go.Figure,
    blocks: list[dict[str, Any]],
    n: int,
    color: str,
    width: float,
):
    for blk in blocks:
        pos = blk["start"] - 0.5
        fig.add_shape(
            type="line",
            xref="x",
            yref="y",
            x0=-0.5,
            x1=n - 0.5,
            y0=pos,
            y1=pos,
            line=dict(color=color, width=width),
            layer="below",
        )
        fig.add_shape(
            type="line",
            xref="x",
            yref="y",
            x0=pos,
            x1=pos,
            y0=-0.5,
            y1=n - 0.5,
            line=dict(color=color, width=width),
            layer="below",
        )


def draw_horizontal_dataset_brackets(
    fig,
    ds_blocks,
    y_pad_units=0.0,
    tick_len_units=0.8,
    line_color="#425a78",
    text_color="#1f334a",
):
    y = -0.5 + y_pad_units
    y_tick0 = y - tick_len_units / 2
    y_tick1 = y + tick_len_units / 2
    for blk in ds_blocks:
        x0 = blk["start"] - 0.5
        x1 = blk["end"] + 0.5
        fig.add_shape(
            type="line",
            xref="x",
            yref="y",
            x0=x0,
            x1=x1,
            y0=y,
            y1=y,
            line=dict(color=line_color, width=1.2),
            layer="above",
        )
        fig.add_shape(
            type="line",
            xref="x",
            yref="y",
            x0=x0,
            x1=x0,
            y0=y_tick0,
            y1=y_tick1,
            line=dict(color=line_color, width=1.2),
            layer="above",
        )
        fig.add_shape(
            type="line",
            xref="x",
            yref="y",
            x0=x1,
            x1=x1,
            y0=y_tick0,
            y1=y_tick1,
            line=dict(color=line_color, width=1.2),
            layer="above",
        )
        cx = (x0 + x1) / 2
        fig.add_annotation(
            xref="x",
            yref="y",
            x=cx,
            y=y - tick_len_units * 0.55,
            text=str(blk["value"]),
            showarrow=False,
            font=dict(size=11, color=text_color),
            textangle=-90,
            yanchor="bottom",
            align="center",
            bgcolor="rgba(255,255,255,0.6)",
        )


def draw_vertical_dataset_brackets(
    fig,
    ds_blocks,
    x_pad_units=0.0,
    tick_len_units=0.8,
    line_color="#425a78",
    text_color="#1f334a",
):
    x = -0.5 + x_pad_units
    x_tick0 = x - tick_len_units / 2
    x_tick1 = x + tick_len_units / 2
    for blk in ds_blocks:
        y0 = blk["start"] - 0.5
        y1 = blk["end"] + 0.5
        fig.add_shape(
            type="line",
            xref="x",
            yref="y",
            x0=x,
            x1=x,
            y0=y0,
            y1=y1,
            line=dict(color=line_color, width=1.2),
            layer="above",
        )
        fig.add_shape(
            type="line",
            xref="x",
            yref="y",
            x0=x_tick0,
            x1=x_tick1,
            y0=y0,
            y1=y0,
            line=dict(color=line_color, width=1.2),
            layer="above",
        )
        fig.add_shape(
            type="line",
            xref="x",
            yref="y",
            x0=x_tick0,
            x1=x_tick1,
            y0=y1,
            y1=y1,
            line=dict(color=line_color, width=1.2),
            layer="above",
        )
        cy = (y0 + y1) / 2
        fig.add_annotation(
            xref="x",
            yref="y",
            x=x - tick_len_units * 0.55,
            y=cy,
            text=str(blk["value"]),
            showarrow=False,
            font=dict(size=11, color=text_color),
            xanchor="right",
            align="center",
            bgcolor="rgba(255,255,255,0.6)",
        )


def build_figure(
    subjects: pd.DataFrame,
    modality: str,
    buckets: dict[str, dict[str, list[Any]]],
    marker_size_px: int = 5,
    opacity: float = 1.0,
) -> go.Figure:
    n = len(subjects)
    fig = go.Figure()

    hovertemplate = (
        "<b>Category: %{customdata[2]}</b><br>"
        "<span style='font-family:monospace'>A: Dataset = %{customdata[3]}, Subject ID = %{customdata[4]}<br>"
        "<span style='font-family:monospace'>B: Dataset = %{customdata[5]}, Subject ID = %{customdata[6]}<br>"
        "<extra></extra>"
    )

    COLOR_MAP = {
        "across_datasets": "#d62728",
        "within_dataset": "#ffdd00",
    }

    for cat in [
        "across_datasets",
        "within_dataset",
    ]:
        data = buckets[cat]
        if not data["x"]:
            continue

        trace_kwargs = dict(
            x=data["x"],
            y=data["y"],
            mode="markers",
            name=cat,
            marker=dict(color=COLOR_MAP[cat], size=marker_size_px),
            opacity=opacity,
        )
        trace_kwargs.update(customdata=data["custom"], hovertemplate=hovertemplate)
        fig.add_trace(go.Scattergl(**trace_kwargs))

    fig.update_xaxes(
        range=[-0.5, n - 0.5],
        showgrid=False,
        zeroline=False,
        showticklabels=False,
    )
    fig.update_yaxes(
        range=[-0.5, n - 0.5],
        autorange="reversed",
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        scaleanchor="x",
        scaleratio=1,
    )

    add_divider_lines(
        fig,
        compute_blocks(subjects, "dataset"),
        n,
        color="#cccccc",
        width=0.6,
    )
    ds_blocks = compute_blocks(subjects, "dataset")
    draw_horizontal_dataset_brackets(
        fig,
        ds_blocks,
        y_pad_units=0.0,
        tick_len_units=0.8,
    )
    draw_vertical_dataset_brackets(
        fig,
        ds_blocks,
        x_pad_units=0.0,
        tick_len_units=0.8,
    )

    fig.add_shape(
        type="rect",
        xref="x",
        yref="y",
        x0=-0.5,
        x1=n - 0.5,
        y0=-0.5,
        y1=n - 0.5,
        line=dict(color="#bbbbbb", width=1.0),
        layer="below",
    )
    fig.update_layout(
        autosize=True,
        title=f"Subjects×Subjects Duplicate Map — modality={modality} (only lower-triangle)",
        legend=dict(orientation="h", x=0, y=1.12, bgcolor="rgba(255,255,255,0.7)"),
        margin=dict(l=10, r=10, t=80, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="closest",
        dragmode="pan",
        uirevision=True,
    )
    return fig


def visualize_file_matrix(
    duplicates_csv: Union[str, Path],
    out_dir: Union[str, Path],
    modality: str,
    process_workers: Optional[int] = None,
    marker_size_px: int = 5,
    opacity: float = 1.0,
):
    if process_workers is None:
        process_workers = max(1, min((os.cpu_count() or 4) // 2, 8))

    duplicates_df = pd.read_csv(duplicates_csv, dtype=str)

    missing = [c for c in REQUIRED_COLS if c not in duplicates_df.columns]
    if missing:
        raise ValueError(f"{duplicates_csv} missing required columns: {missing}")

    duplicates_df = duplicates_df[REQUIRED_COLS].copy()
    for c in REQUIRED_COLS:
        duplicates_df[c] = duplicates_df[c].fillna("").astype(str)

    duplicates_df = duplicates_df[
        (duplicates_df["group_id"] != "")
        & (duplicates_df["dataset"] != "")
        & (duplicates_df["subject_id"] != "")
        & (duplicates_df["subject_key"] != "")
    ].copy()

    subjects, _ = build_sorted_subject_index(duplicates_df)
    buckets = collect_duplicate_points_parallel(
        duplicates_df,
        subjects,
        n_procs=process_workers,
    )

    fig = build_figure(subjects, modality, buckets, marker_size_px, opacity)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_html = out_dir / f"files_matrix_{modality}.html"

    cfg = dict(scrollZoom=True, displaylogo=False)
    plot_html(
        fig,
        filename=str(out_html),
        auto_open=False,
        include_plotlyjs="cdn",
        config=cfg,
    )
