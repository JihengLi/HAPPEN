from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .discover import (
    glob_bids_candidates,
    parse_triplet,
    classify_candidates,
    write_mapping_to_csv,
)
from .hashing import (
    run_hash_pipeline_chunked,
    group_and_organize_duplicates,
)
from .verify import verify_all_categories
from ..report import (
    report_by_dataset,
    report_by_category,
    report_by_category_across_datasets,
    visualize_file_matrix,
)
from ..utils.runtime_profile import RuntimeProfiler


@dataclass
class ExactStageArtifacts:
    out_dir: Path
    exact_dir: Path
    candidates_csv: Optional[Path]
    valid_csv: Path
    hash_csv: Path
    hash_errors_csv: Optional[Path]
    duplicates_csv: Path
    by_dataset_dir: Path
    by_category_dir: Path
    by_category_stats_dir: Path
    verified_dir: Path
    figures_dir: Path


def _load_valid_csv(valid_csv: Path) -> list[Path]:
    valid_df = pd.read_csv(valid_csv.resolve(), dtype=str)
    required_cols = {
        "dataset",
        "subject_id",
        "session_id",
        "candidate",
        "resolved_path",
    }
    missing = required_cols.difference(valid_df.columns)
    if missing:
        raise ValueError(f"--valid_csv missing required columns: {sorted(missing)}")

    vals = sorted({s for s in valid_df["resolved_path"].dropna().astype(str) if s})
    return [Path(s) for s in vals]


def _build_or_load_candidates(
    root: Optional[str],
    candidates_csv: Optional[str],
    modality: str,
) -> pd.DataFrame:
    if candidates_csv:
        df = pd.read_csv(Path(candidates_csv).resolve(), dtype=str)
        required = {"dataset", "subject_id", "session_id", "candidate"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(
                f"--candidates_csv must contain columns: {sorted(required)}, "
                f"missing: {sorted(missing)}"
            )
        return df[list(required)].fillna("")

    if not root:
        raise ValueError("One of --valid_csv, --candidates_csv, or --root is required.")

    root_path = Path(root).resolve()
    candidates = glob_bids_candidates(root_path, modality)
    print(
        f"[DISCOVERY] Globbed {len(candidates)} candidates under {root_path} ({modality})."
    )

    rows = []
    for p in candidates:
        cstr = str(p)
        ds, subj, ses = parse_triplet(cstr)
        rows.append(
            {
                "dataset": ds,
                "subject_id": subj,
                "session_id": ses,
                "candidate": cstr,
            }
        )

    df = pd.DataFrame(
        rows,
        columns=["dataset", "subject_id", "session_id", "candidate"],
    ).fillna("")
    print(f"[INFO] Built candidates DataFrame with {len(df)} rows from BIDS root.")
    return df


def _classify_and_write(
    candidates_df: pd.DataFrame,
    exact_dir: Path,
    thread_workers: int,
) -> tuple[list[Path], Path]:
    candidates_csv_path = exact_dir / "candidates.csv"
    candidates_df.to_csv(candidates_csv_path, index=False)
    print(f"[INFO] Wrote candidates metadata to {candidates_csv_path}")

    candidates_paths = [
        Path(s) for s in candidates_df["candidate"].astype(str).tolist()
    ]

    ok, missing, permission_denied, not_nifti, unreadable, derivatives = (
        classify_candidates(
            candidates=candidates_paths,
            verify_readable=False,
            max_workers=thread_workers,
            use_threads=True,
        )
    )

    write_mapping_to_csv(
        missing,
        exact_dir / "invalid_missing_paths_discovery.csv",
        candidates_df,
        join_side="left",
    )
    write_mapping_to_csv(
        permission_denied,
        exact_dir / "invalid_permission_denied_paths_discovery.csv",
        candidates_df,
        join_side="left",
    )
    write_mapping_to_csv(
        not_nifti,
        exact_dir / "invalid_not_nifti_discovery.csv",
        candidates_df,
        join_side="left",
    )
    write_mapping_to_csv(
        unreadable,
        exact_dir / "invalid_unreadable_discovery.csv",
        candidates_df,
        join_side="left",
    )
    write_mapping_to_csv(
        derivatives,
        exact_dir / "invalid_derivatives.csv",
        candidates_df,
        join_side="left",
    )

    files = [Path(s) for s in sorted({v for v in ok.keys() if v})]
    valid_csv = exact_dir / "valid.csv"

    if files:
        write_mapping_to_csv(
            ok,
            valid_csv,
            candidates_df,
            left_name="resolved_path",
            right_name="candidate",
        )
        print(f"[INFO] Wrote {len(files)} paths to {valid_csv}")

    return files, valid_csv


def run_exact_stage(
    root: Optional[str] = None,
    candidates_csv: Optional[str] = None,
    valid_csv: Optional[str] = None,
    modality: str = "T1w",
    out: str = "./data",
    thread_workers: int = 4,
    process_workers: int = 4,
    verify: bool = False,
    profiler: Optional[RuntimeProfiler] = None,
) -> ExactStageArtifacts:
    out_dir = Path(out).expanduser().resolve()
    exact_dir = out_dir / "exact"
    by_dataset_dir = exact_dir / "by_dataset"
    by_category_dir = exact_dir / "by_category"
    by_category_stats_dir = exact_dir / "by_category_stats"
    verified_dir = exact_dir / "verified"
    figures_dir = exact_dir / "figures"

    out_dir.mkdir(parents=True, exist_ok=True)
    exact_dir.mkdir(parents=True, exist_ok=True)

    candidates_csv_path: Optional[Path] = None
    hash_errors_csv: Optional[Path] = None

    load_verify_ctx = (
        profiler.stage(
            "exact_load_verify",
            modality=modality,
            thread_workers=thread_workers,
            input_mode=(
                "valid_csv"
                if valid_csv
                else "candidates_csv" if candidates_csv else "root"
            ),
            root=root or "",
            candidates_csv=candidates_csv or "",
            valid_csv=valid_csv or "",
        )
        if profiler is not None
        else nullcontext()
    )

    with load_verify_ctx:
        if valid_csv:
            valid_csv_path = Path(valid_csv).expanduser().resolve()
            files = _load_valid_csv(valid_csv_path)
            if not files:
                print(f"[WARN] No paths in --valid_csv: {valid_csv}")
                raise SystemExit(0)
            print(
                f"[RESUME] Loaded {len(files)} valid paths from {valid_csv}. Skipping classification."
            )
        else:
            candidates_df = _build_or_load_candidates(
                root=root,
                candidates_csv=candidates_csv,
                modality=modality,
            )
            candidates_csv_path = exact_dir / "candidates.csv"
            files, valid_csv_path = _classify_and_write(
                candidates_df=candidates_df,
                exact_dir=exact_dir,
                thread_workers=thread_workers,
            )
            if not files:
                print("No files found after classification; aborting hashing/dedup.")
                raise SystemExit(0)

    hash_group_ctx = (
        profiler.stage(
            "exact_hash_group",
            n_input=len(files),
            modality=modality,
            process_workers=process_workers,
            chunk_size=8000,
            batch_size=1000,
            inflight_factor=2,
        )
        if profiler is not None
        else nullcontext()
    )

    with hash_group_ctx:
        hash_csv, errors = run_hash_pipeline_chunked(
            files=files,
            process_workers=max(1, process_workers),
            out_path=exact_dir / f"{modality}_hashes.csv",
            chunk_size=8000,
            batch_size=1000,
            inflight_factor=2,
        )

        if errors:
            hash_errors_csv = exact_dir / "hash_errors.csv"
            pd.DataFrame(errors).to_csv(hash_errors_csv, index=False)
            print(f"[WARN] {len(errors)} files failed to hash. See {hash_errors_csv}")

        duplicates_csv = exact_dir / "duplicates.csv"

        has_duplicates = group_and_organize_duplicates(
            hash_csv,
            "can_hash",
            modality,
            valid_csv_path,
            duplicates_csv,
        )

        if not has_duplicates:
            by_dataset_dir.mkdir(parents=True, exist_ok=True)
            by_category_dir.mkdir(parents=True, exist_ok=True)
            by_category_stats_dir.mkdir(parents=True, exist_ok=True)
            verified_dir.mkdir(parents=True, exist_ok=True)
            figures_dir.mkdir(parents=True, exist_ok=True)

            print("[EXACT] No exact duplicates found.")
            print(f"[EXACT] Wrote empty duplicates file -> {duplicates_csv}")
            print(
                "[EXACT] Skipping exact reporting/verification and continuing pipeline."
            )

            return ExactStageArtifacts(
                out_dir=out_dir,
                exact_dir=exact_dir,
                candidates_csv=candidates_csv_path,
                valid_csv=Path(valid_csv_path),
                hash_csv=Path(hash_csv),
                hash_errors_csv=hash_errors_csv,
                duplicates_csv=duplicates_csv,
                by_dataset_dir=by_dataset_dir,
                by_category_dir=by_category_dir,
                by_category_stats_dir=by_category_stats_dir,
                verified_dir=verified_dir,
                figures_dir=figures_dir,
            )

    categorize_ctx = (
        profiler.stage(
            "exact_categorize",
            modality=modality,
            process_workers=process_workers,
            thread_workers=thread_workers,
            verify_enabled=verify,
            exclude_verify_categories="SAME_BYTES" if verify else "",
        )
        if profiler is not None
        else nullcontext()
    )

    with categorize_ctx:
        visualize_file_matrix(
            duplicates_csv,
            figures_dir,
            modality,
            process_workers,
        )

        report_by_dataset(
            duplicates_csv,
            by_dataset_dir,
            modality,
        )

        report_by_category(
            by_dataset_dir,
            by_category_dir,
            by_category_stats_dir,
            modality,
        )

        report_by_category_across_datasets(
            duplicates_csv,
            by_category_dir,
            by_category_stats_dir,
            modality,
        )

        if verify:
            verify_all_categories(
                by_category_dir,
                verified_dir,
                thread_workers,
                process_workers,
                exclude_categories={"SAME_BYTES"},
            )
        else:
            verified_dir.mkdir(parents=True, exist_ok=True)
            print("[EXACT] Verification skipped (verify=false).")

    print("[EXACT] Done.")
    print(f"[EXACT] outputs saved under -> {exact_dir}")

    return ExactStageArtifacts(
        out_dir=out_dir,
        exact_dir=exact_dir,
        candidates_csv=candidates_csv_path,
        valid_csv=Path(valid_csv_path),
        hash_csv=Path(hash_csv),
        hash_errors_csv=hash_errors_csv,
        duplicates_csv=duplicates_csv,
        by_dataset_dir=by_dataset_dir,
        by_category_dir=by_category_dir,
        by_category_stats_dir=by_category_stats_dir,
        verified_dir=verified_dir,
        figures_dir=figures_dir,
    )
