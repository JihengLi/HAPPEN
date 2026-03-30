"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

#!/usr/bin/env python3
from __future__ import annotations

import os, re

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.offline import plot as plot_html

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from multiprocessing import get_context


def build_sorted_file_index(
    duplicates_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    files = (
        duplicates_df[
            ["dataset", "subject_id", "session_id", "candidate", "resolved_path"]
        ]
        .drop_duplicates(subset=["candidate"])
        .copy()
    )
    dup_count_per_ds = (
        duplicates_df.groupby(["dataset", "can_hash"])["candidate"]
        .nunique()
        .groupby("dataset")
        .sum()
        .astype(int)
    )
    all_ds = files["dataset"].astype(str).unique()
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

    files["dataset_rank"] = files["dataset"].map(dataset_rank_map).astype(int)
    files["subject_key"] = files["subject_id"].astype(str)
    files["session_key"] = files["session_id"].astype(str)

    def natural_int(s: str, default: int = 10**9) -> int:
        if not isinstance(s, str):
            return default
        m = re.search(r"\d+", s)
        return int(m.group()) if m else default

    files["subject_num"] = files["subject_key"].map(natural_int)
    files["session_num"] = files["session_key"].map(natural_int)

    files = files.sort_values(
        by=[
            "dataset_rank",
            "subject_num",
            "subject_key",
            "session_num",
            "session_key",
            "candidate",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    files["file_idx"] = np.arange(len(files), dtype=int)
    idx_of = dict(zip(files["candidate"], files["file_idx"]))
    return files, idx_of


def _collect_points_from_group_arrays(
    file_idx: np.ndarray,
    ds_code: np.ndarray,
    sub_code: np.ndarray,
    ses_code: np.ndarray,
    ses_missing: np.ndarray,
    cand: np.ndarray,
    rpath: np.ndarray,
    ds_str: np.ndarray,
    sub_str: np.ndarray,
    ses_str: np.ndarray,
) -> Dict[str, Dict[str, List[Any]]]:
    out: Dict[str, Dict[str, List[Any]]] = {
        "within_session": {"x": [], "y": [], "custom": []},
        "across_sessions": {"x": [], "y": [], "custom": []},
        "across_subjects": {"x": [], "y": [], "custom": []},
        "across_datasets": {"x": [], "y": [], "custom": []},
    }
    k = file_idx.shape[0]
    if k < 2:
        return out

    I, J = np.triu_indices(k, 1)

    ia, ib = file_idx[I], file_idx[J]
    ia, ib = np.minimum(ia, ib), np.maximum(ia, ib)
    same_ds = ds_code[I] == ds_code[J]
    same_sub = sub_code[I] == sub_code[J]
    same_ses = ses_code[I] == ses_code[J]
    either_missing = ses_missing[I] | ses_missing[J]

    masks = {
        "within_session": same_ds & same_sub & (same_ses | either_missing),
        "across_sessions": same_ds & same_sub & (~same_ses) & (~either_missing),
        "across_subjects": same_ds & ~same_sub,
        "across_datasets": ~same_ds,
    }

    for cat, m in masks.items():
        if not np.any(m):
            continue
        xs = ia[m].tolist()
        ys = ib[m].tolist()
        out[cat]["x"].extend(xs)
        out[cat]["y"].extend(ys)
        ca = cand[I][m]
        cb = cand[J][m]
        ra = rpath[I][m]
        rb = rpath[J][m]
        dsa = ds_str[I][m]
        suba = sub_str[I][m]
        sesa = ses_str[I][m]
        dsb = ds_str[J][m]
        subb = sub_str[J][m]
        sesb = ses_str[J][m]
        cat_label = cat.replace("_", " ")
        out[cat]["custom"].extend(
            [
                (
                    int(x),
                    int(y),
                    str(cat_label),
                    str(dsa_i),
                    str(suba_i),
                    str(sesa_i),
                    str(dsb_i),
                    str(subb_i),
                    str(sesb_i),
                    str(a),
                    str(b),
                    str(ra),
                    str(rb),
                )
                for x, y, dsa_i, suba_i, sesa_i, dsb_i, subb_i, sesb_i, a, b, ra, rb in zip(
                    xs,
                    ys,
                    dsa.tolist(),
                    suba.tolist(),
                    sesa.tolist(),
                    dsb.tolist(),
                    subb.tolist(),
                    sesb.tolist(),
                    ca.tolist(),
                    cb.tolist(),
                    ra.tolist(),
                    rb.tolist(),
                )
            ]
        )
    return out


def _worker_process_chunk(
    chunk: List[Dict[str, Any]],
) -> Dict[str, Dict[str, List[Any]]]:
    out: Dict[str, Dict[str, List[Any]]] = {
        "within_session": {"x": [], "y": [], "custom": []},
        "across_sessions": {"x": [], "y": [], "custom": []},
        "across_subjects": {"x": [], "y": [], "custom": []},
        "across_datasets": {"x": [], "y": [], "custom": []},
    }
    for item in chunk:
        res = _collect_points_from_group_arrays(
            item["file_idx"],
            item["ds_code"],
            item["sub_code"],
            item["ses_code"],
            item["ses_missing"],
            item["cand"],
            item["rpath"],
            item["ds_str"],
            item["sub_str"],
            item["ses_str"],
        )
        for cat in out.keys():
            out[cat]["x"].extend(res[cat]["x"])
            out[cat]["y"].extend(res[cat]["y"])
            out[cat]["custom"].extend(res[cat]["custom"])
    return out


def collect_duplicate_points_parallel(
    duplicates_df: pd.DataFrame, files: pd.DataFrame, n_procs: int
) -> Dict[str, Dict[str, List[Any]]]:
    duplicates_df = duplicates_df.copy()
    duplicates_df = duplicates_df.merge(
        files[["candidate", "file_idx"]], on="candidate", how="left", validate="m:1"
    )
    duplicates_df["ds_code"] = pd.factorize(duplicates_df["dataset"])[0].astype(
        np.int32
    )
    duplicates_df["sub_code"] = pd.factorize(duplicates_df["subject_id"])[0].astype(
        np.int32
    )
    duplicates_df["ses_code"] = pd.factorize(duplicates_df["session_id"])[0].astype(
        np.int32
    )
    duplicates_df["ses_missing"] = duplicates_df["session_id"].isna() | (
        duplicates_df["session_id"].astype(str).str.strip() == ""
    )

    if n_procs <= 1:
        out: Dict[str, Dict[str, List[Any]]] = {
            "within_session": {"x": [], "y": [], "custom": []},
            "across_sessions": {"x": [], "y": [], "custom": []},
            "across_subjects": {"x": [], "y": [], "custom": []},
            "across_datasets": {"x": [], "y": [], "custom": []},
        }
        for can_hash, g in duplicates_df.groupby("can_hash", sort=False):
            pack = {
                "file_idx": g["file_idx"].to_numpy(np.int64, copy=False),
                "ds_code": g["ds_code"].to_numpy(np.int32, copy=False),
                "sub_code": g["sub_code"].to_numpy(np.int32, copy=False),
                "ses_code": g["ses_code"].to_numpy(np.int32, copy=False),
                "ses_missing": g["ses_missing"].to_numpy(np.bool_, copy=False),
                "cand": g["candidate"].to_numpy(object, copy=False),
                "rpath": g["resolved_path"]
                .astype("string")
                .fillna("")
                .to_numpy(object, copy=False),
                "ds_str": g["dataset"]
                .astype("string")
                .fillna("")
                .to_numpy(object, copy=False),
                "sub_str": g["subject_id"]
                .astype("string")
                .fillna("")
                .to_numpy(object, copy=False),
                "ses_str": (
                    g["session_id"]
                    .astype("string")
                    .fillna("")
                    .str.strip()
                    .mask(lambda s: s == "", "nan")
                ).to_numpy(object, copy=False),
            }
            res = _collect_points_from_group_arrays(**pack)
            for cat in out.keys():
                out[cat]["x"].extend(res[cat]["x"])
                out[cat]["y"].extend(res[cat]["y"])
                out[cat]["custom"].extend(res[cat]["custom"])
        return out
    group_list: List[Tuple[Any, int]] = []
    sizes: List[int] = []
    for can_hash, g in duplicates_df.groupby("can_hash", sort=False):
        k = int(g.shape[0])
        pairs = k * (k - 1) // 2
        if pairs <= 0:
            continue
        group_list.append((can_hash, k))
        sizes.append(pairs)

    order = np.argsort(np.array(sizes))[::-1]
    bins: List[List[Any]] = [[] for _ in range(n_procs)]
    bin_loads = [0] * n_procs

    duplicates_df = duplicates_df.reset_index(drop=True)
    group_indices: Dict[Any, np.ndarray] = {
        ch: idx.values
        for ch, idx in duplicates_df.groupby("can_hash", sort=False).groups.items()
    }

    for idx in order:
        can_hash, k = group_list[idx]
        pairs = sizes[idx]
        b = int(np.argmin(bin_loads))
        bin_loads[b] += pairs
        rows = group_indices[can_hash]
        g = duplicates_df.loc[rows]
        bins[b].append(
            {
                "file_idx": g["file_idx"].to_numpy(np.int64, copy=False),
                "ds_code": g["ds_code"].to_numpy(np.int32, copy=False),
                "sub_code": g["sub_code"].to_numpy(np.int32, copy=False),
                "ses_code": g["ses_code"].to_numpy(np.int32, copy=False),
                "ses_missing": g["ses_missing"].to_numpy(np.bool_, copy=False),
                "cand": g["candidate"].to_numpy(object, copy=False),
                "rpath": g["resolved_path"]
                .astype("string")
                .fillna("")
                .to_numpy(object, copy=False),
                "ds_str": g["dataset"]
                .astype("string")
                .fillna("")
                .to_numpy(object, copy=False),
                "sub_str": g["subject_id"]
                .astype("string")
                .fillna("")
                .to_numpy(object, copy=False),
                "ses_str": (
                    g["session_id"]
                    .astype("string")
                    .fillna("")
                    .str.strip()
                    .mask(lambda s: s == "", "nan")
                ).to_numpy(object, copy=False),
            }
        )

    with get_context("spawn").Pool(processes=n_procs) as pool:
        results = pool.map(_worker_process_chunk, bins)

    out: Dict[str, Dict[str, List[Any]]] = {
        "within_session": {"x": [], "y": [], "custom": []},
        "across_sessions": {"x": [], "y": [], "custom": []},
        "across_subjects": {"x": [], "y": [], "custom": []},
        "across_datasets": {"x": [], "y": [], "custom": []},
    }
    for res in results:
        for cat in out.keys():
            out[cat]["x"].extend(res[cat]["x"])
            out[cat]["y"].extend(res[cat]["y"])
            out[cat]["custom"].extend(res[cat]["custom"])
    return out


def compute_blocks(files: pd.DataFrame, by: str) -> list[dict[str, Any]]:
    vals = files[by].astype(str).tolist()
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
    fig: go.Figure, blocks: list[dict[str, Any]], n: int, color: str, width: float
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
    files: pd.DataFrame,
    modality: str,
    buckets: dict[str, dict[str, list[Any]]],
    marker_size_px: int = 5,
    opacity: float = 1.0,
) -> go.Figure:
    n = len(files)
    fig = go.Figure()

    hovertemplate = (
        "<b>Category: %{customdata[2]}</b><br>"
        "<span style='font-family:monospace'>A: Dataset = %{customdata[3]}, Subject ID = %{customdata[4]}, Session ID = %{customdata[5]}<br>"
        "<span style='font-family:monospace'>B: Dataset = %{customdata[6]}, Subject ID = %{customdata[7]}, Session ID = %{customdata[8]}<br>"
        "<span style='font-family:monospace'>A: %{customdata[9]}</span><br>"
        "<span style='font-family:monospace'>B: %{customdata[10]}</span><br>"
        # "<span style='font-family:monospace'>A: %{customdata[11]}</span><br>"
        # "<span style='font-family:monospace'>B: %{customdata[12]}</span><br>"
        "<extra></extra>"
    )

    COLOR_MAP = {
        "across_datasets": "#d62728",
        "across_subjects": "#ffdd00",
        "across_sessions": "#2ca02c",
        "within_session": "#000000",
    }

    for cat in [
        "across_datasets",
        "across_subjects",
        "across_sessions",
        "within_session",
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
        fig, compute_blocks(files, "dataset"), n, color="#cccccc", width=0.6
    )
    ds_blocks = compute_blocks(files, "dataset")
    draw_horizontal_dataset_brackets(
        fig, ds_blocks, y_pad_units=0.0, tick_len_units=0.8
    )
    draw_vertical_dataset_brackets(fig, ds_blocks, x_pad_units=0.0, tick_len_units=0.8)

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
        title=f"Files×Files Duplicate Map — modality={modality} (only lower-triangle)",
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
    duplicates_df = pd.read_csv(duplicates_csv)
    files, _ = build_sorted_file_index(duplicates_df)
    buckets = collect_duplicate_points_parallel(
        duplicates_df, files, n_procs=process_workers
    )
    fig = build_figure(files, modality, buckets, marker_size_px, opacity)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_html = out_dir / f"files_matrix_{modality}.html"
    cfg = dict(scrollZoom=True, displaylogo=False)
    plot_html(
        fig, filename=str(out_html), auto_open=False, include_plotlyjs="cdn", config=cfg
    )
