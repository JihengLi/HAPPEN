"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

from .assets import build_review_assets
from .embedding import run_embedding_inference
from .format import build_review_table
from .grouping import run_grouping
from .preparation import prepare_no_exact_dup
from .retrieval import run_faiss_retrieval
from ..utils.runtime_profile import RuntimeProfiler


@dataclass
class NearStageArtifacts:
    near_dir: Path

    no_exact_dup_csv: Path
    exact_removed_csv: Path

    embeddings_npy: Path
    embedding_manifest_csv: Path
    embedding_failures_csv: Path

    raw_neighbors_csv: Path
    scan_candidates_csv: Path
    subject_edges_csv: Path
    subject_groups_csv: Path

    review_dir: Optional[Path]
    review_candidates_csv: Optional[Path]
    png_dir: Optional[Path]
    diff_dir: Optional[Path]
    checkerboard_dir: Optional[Path]
    scan_asset_failures_csv: Optional[Path]
    pair_asset_failures_csv: Optional[Path]


def run_near_stage(
    out_dir: Union[str, Path],
    preprocess_workers: int = 1,
    batch_size: int = 16,
    device: str = "auto",
    use_amp: bool = False,
    normalize: bool = True,
    overwrite_embeddings: bool = False,
    fail_fast: bool = False,
    crop_shape: Tuple[int, int, int] = (160, 192, 160),
    axial_axis: int = 2,
    slices_2p5d: int = 32,
    slice_stride: int = 1,
    winsor: Tuple[float, float] = (1.0, 99.0),
    hist_matching: bool = False,
    repro: bool = False,
    delete_extras: bool = True,
    inflight_factor: int = 2,
    topk: int = 500,
    retrieval_batch_size: int = 2048,
    use_ivf: bool = True,
    nlist: int = 4096,
    nprobe: int = 64,
    train_size: int = 200000,
    seed: int = 0,
    overwrite_retrieval: bool = False,
    min_similarity: float = 0.92,
    review_mode: str = "off",  # "off" | "lazy" | "precompute"
    review_workers: int = 24,
    overwrite_review_assets: bool = False,
    max_scan_candidates_per_query: Optional[int] = None,
    profiler: Optional[RuntimeProfiler] = None,
) -> NearStageArtifacts:
    out_dir = Path(out_dir).expanduser().resolve()
    exact_dir = out_dir / "exact"
    near_dir = out_dir / "near"
    review_dir = near_dir / "review"

    near_dir.mkdir(parents=True, exist_ok=True)

    valid_csv = exact_dir / "valid.csv"
    exact_duplicates_csv = exact_dir / "duplicates.csv"
    hash_errors_csv = exact_dir / "hash_errors.csv"

    no_exact_dup_csv = near_dir / "no_exact_dup.csv"
    exact_removed_csv = near_dir / "exact_removed.csv"

    embeddings_npy = near_dir / "embeddings.npy"
    embedding_manifest_csv = near_dir / "embedding_manifest.csv"
    embedding_failures_csv = near_dir / "embedding_failures.csv"

    raw_neighbors_csv = near_dir / "raw_neighbors.csv"
    scan_candidates_csv = near_dir / "scan_candidates.csv"
    subject_edges_csv = near_dir / "subject_edges.csv"
    subject_groups_csv = near_dir / "subject_groups.csv"

    review_candidates_csv: Optional[Path] = None
    png_dir: Optional[Path] = None
    diff_dir: Optional[Path] = None
    checkerboard_dir: Optional[Path] = None
    scan_asset_failures_csv: Optional[Path] = None
    pair_asset_failures_csv: Optional[Path] = None

    review_mode = str(review_mode).strip().lower()
    if review_mode not in {"off", "lazy", "precompute"}:
        raise ValueError("review_mode must be 'off', 'lazy', or 'precompute'")

    remove_exact_ctx = (
        profiler.stage(
            "near_remove_exact",
            valid_csv=str(valid_csv),
            exact_duplicates_csv=str(exact_duplicates_csv),
            hash_errors_csv=str(hash_errors_csv),
        )
        if profiler is not None
        else nullcontext()
    )

    with remove_exact_ctx:
        print("[NEAR] Step 1/6: prepare near input")
        prepare_no_exact_dup(
            valid_csv=valid_csv,
            exact_duplicates_csv=exact_duplicates_csv,
            out_keep_csv=no_exact_dup_csv,
            out_removed_csv=exact_removed_csv,
            hash_errors_csv=hash_errors_csv,
        )

    embedding_ctx = (
        profiler.stage(
            "near_embedding",
            preprocess_workers=preprocess_workers,
            batch_size=batch_size,
            device=device,
            use_amp=use_amp,
            normalize=normalize,
            overwrite_embeddings=overwrite_embeddings,
            fail_fast=fail_fast,
            crop_shape=str(tuple(crop_shape)),
            axial_axis=axial_axis,
            slices_2p5d=slices_2p5d,
            slice_stride=slice_stride,
            winsor=str(tuple(winsor)),
            hist_matching=hist_matching,
            repro=repro,
            delete_extras=delete_extras,
            inflight_factor=inflight_factor,
        )
        if profiler is not None
        else nullcontext()
    )

    with embedding_ctx:
        print("[NEAR] Step 2/6: embedding inference")
        run_embedding_inference(
            in_csv=no_exact_dup_csv,
            out_dir=near_dir,
            preprocess_workers=preprocess_workers,
            batch_size=batch_size,
            device=device,
            use_amp=use_amp,
            normalize=normalize,
            overwrite=overwrite_embeddings,
            fail_fast=fail_fast,
            crop_shape=crop_shape,
            axial_axis=axial_axis,
            slices_2p5d=slices_2p5d,
            slice_stride=slice_stride,
            winsor=winsor,
            hist_matching=hist_matching,
            repro=repro,
            delete_extras=delete_extras,
            inflight_factor=inflight_factor,
            profiler=profiler,
        )

    faiss_group_ctx = (
        profiler.stage(
            "near_faiss_group",
            topk=topk,
            retrieval_batch_size=retrieval_batch_size,
            use_ivf=use_ivf,
            nlist=nlist,
            nprobe=nprobe,
            train_size=train_size,
            seed=seed,
            overwrite_retrieval=overwrite_retrieval,
            min_similarity=min_similarity,
        )
        if profiler is not None
        else nullcontext()
    )

    with faiss_group_ctx:
        print("[NEAR] Step 3/6: FAISS retrieval")
        run_faiss_retrieval(
            manifest_csv=embedding_manifest_csv,
            embeddings_npy=embeddings_npy,
            out_csv=raw_neighbors_csv,
            topk=topk,
            batch_size=retrieval_batch_size,
            use_ivf=use_ivf,
            nlist=nlist,
            nprobe=nprobe,
            train_size=train_size,
            seed=seed,
            overwrite=overwrite_retrieval,
        )

        print("[NEAR] Step 4/6: grouping")
        run_grouping(
            raw_neighbors_csv=raw_neighbors_csv,
            out_scan_candidates_csv=scan_candidates_csv,
            out_subject_edges_csv=subject_edges_csv,
            out_subject_groups_csv=subject_groups_csv,
            min_similarity=min_similarity,
        )

    if review_mode != "off":
        review_dir.mkdir(parents=True, exist_ok=True)

        review_candidates_csv = review_dir / "review_candidates.csv"

        print("[NEAR] Step 5/6: build review table")
        build_review_table(
            review_dir=review_dir,
            scan_candidates_csv=scan_candidates_csv,
            max_candidates_per_query=max_scan_candidates_per_query,
        )

        print(f"[NEAR] Step 6/6: build review assets (mode={review_mode})")
        build_review_assets(
            review_dir=review_dir,
            manifest_csv=embedding_manifest_csv,
            review_candidates_csv=review_candidates_csv,
            mode=review_mode,
            workers=review_workers,
            overwrite=overwrite_review_assets,
            winsor=winsor,
            hist_matching=hist_matching,
            repro=repro,
            delete_extras=delete_extras,
        )

        png_dir = review_dir / "assets" / "png"
        diff_dir = review_dir / "assets" / "diff"
        checkerboard_dir = review_dir / "assets" / "checkerboard"
        scan_asset_failures_csv = review_dir / "scan_asset_failures.csv"
        pair_asset_failures_csv = review_dir / "pair_asset_failures.csv"

    print("[NEAR] Done.")
    print(f"[NEAR] outputs saved under -> {near_dir}")

    return NearStageArtifacts(
        near_dir=near_dir,
        no_exact_dup_csv=no_exact_dup_csv,
        exact_removed_csv=exact_removed_csv,
        embeddings_npy=embeddings_npy,
        embedding_manifest_csv=embedding_manifest_csv,
        embedding_failures_csv=embedding_failures_csv,
        raw_neighbors_csv=raw_neighbors_csv,
        scan_candidates_csv=scan_candidates_csv,
        subject_edges_csv=subject_edges_csv,
        subject_groups_csv=subject_groups_csv,
        review_dir=review_dir if review_mode != "off" else None,
        review_candidates_csv=review_candidates_csv,
        png_dir=png_dir,
        diff_dir=diff_dir,
        checkerboard_dir=checkerboard_dir,
        scan_asset_failures_csv=scan_asset_failures_csv,
        pair_asset_failures_csv=pair_asset_failures_csv,
    )
