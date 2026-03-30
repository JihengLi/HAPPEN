"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

from __future__ import annotations

import shutil
import tomllib
import argparse

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .exact.stage import ExactStageArtifacts, run_exact_stage
from .near.stage import NearStageArtifacts, run_near_stage
from .utils.runtime_profile import RuntimeProfiler


@dataclass
class PipelineArtifacts:
    config_path: Path
    out_dir: Path
    stages: List[str]
    exact: Optional[ExactStageArtifacts]
    near: Optional[NearStageArtifacts]


def _read_toml(config_path: Union[str, Path]) -> Dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")

    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"invalid TOML config: {config_path}")

    return cfg


def _require_table(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    if name not in cfg:
        raise ValueError(f"config missing required table: [{name}]")
    table = cfg[name]
    if not isinstance(table, dict):
        raise ValueError(f"config table [{name}] must be a table/object")
    return table


def _optional_str(d: Dict[str, Any], key: str) -> Optional[str]:
    if key not in d:
        return None
    v = d[key]
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _optional_int(d: Dict[str, Any], key: str) -> Optional[int]:
    if key not in d or d[key] is None:
        return None
    return int(d[key])


def _tuple3_int(v: Any, key: str) -> Tuple[int, int, int]:
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        raise ValueError(f"{key} must be a length-3 list/tuple")
    return (int(v[0]), int(v[1]), int(v[2]))


def _tuple2_float(v: Any, key: str) -> Tuple[float, float]:
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        raise ValueError(f"{key} must be a length-2 list/tuple")
    return (float(v[0]), float(v[1]))


def _resolve_stages(run_cfg: Dict[str, Any]) -> List[str]:
    stages = run_cfg.get("stages", ["exact", "near"])
    if not isinstance(stages, list) or not stages:
        raise ValueError("[run].stages must be a non-empty list")

    out = [str(x).strip().lower() for x in stages]
    allowed = {"exact", "near"}
    bad = [x for x in out if x not in allowed]
    if bad:
        raise ValueError(f"unsupported stage(s) in [run].stages: {bad}")

    seen = set()
    uniq = []
    for s in out:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq


def _copy_config_snapshot(config_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "config_used.toml"
    shutil.copy2(config_path, dst)
    return dst


def _validate_exact_input_mode(
    exact_cfg: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    root = _optional_str(exact_cfg, "root")
    candidates_csv = _optional_str(exact_cfg, "candidates_csv")
    valid_csv = _optional_str(exact_cfg, "valid_csv")

    provided = [
        ("root", root),
        ("candidates_csv", candidates_csv),
        ("valid_csv", valid_csv),
    ]
    provided_nonempty = [name for name, value in provided if value]

    if len(provided_nonempty) == 0:
        raise ValueError(
            "[exact] one of root, candidates_csv, or valid_csv must be provided"
        )

    if len(provided_nonempty) > 1:
        raise ValueError(
            f"[exact] root, candidates_csv, and valid_csv are mutually exclusive; "
            f"got multiple values: {provided_nonempty}"
        )

    return root, candidates_csv, valid_csv


def run_pipeline(
    config_path: Union[str, Path],
) -> PipelineArtifacts:
    config_path = Path(config_path).expanduser().resolve()
    cfg = _read_toml(config_path)

    run_cfg = _require_table(cfg, "run")
    out = run_cfg.get("out", None)
    if out is None or not str(out).strip():
        raise ValueError("[run].out is required")

    out_dir = Path(str(out)).expanduser().resolve()
    stages = _resolve_stages(run_cfg)

    profile_runtime = bool(run_cfg.get("profile_runtime", False))
    perf_dir = out_dir / "_perf"

    out_dir.mkdir(parents=True, exist_ok=True)
    _copy_config_snapshot(config_path, out_dir)

    profiler = RuntimeProfiler(
        out_dir=perf_dir,
        enabled=profile_runtime,
    )
    profiler.set_run_meta(
        config_path=str(config_path),
        out_dir=str(out_dir),
        requested_stages=",".join(stages),
    )

    exact_artifacts: Optional[ExactStageArtifacts] = None
    near_artifacts: Optional[NearStageArtifacts] = None

    try:
        with profiler.stage(
            "total_pipeline",
            requested_stages=",".join(stages),
        ):
            if "exact" in stages:
                exact_cfg = _require_table(cfg, "exact")
                root, candidates_csv, valid_csv = _validate_exact_input_mode(exact_cfg)

                profiler.set_run_meta(
                    exact_input_mode=(
                        "valid_csv"
                        if valid_csv
                        else "candidates_csv" if candidates_csv else "root"
                    ),
                    exact_modality=str(exact_cfg.get("modality", "T1w")),
                    exact_thread_workers=int(exact_cfg.get("thread_workers", 4)),
                    exact_process_workers=int(exact_cfg.get("process_workers", 4)),
                    exact_verify=bool(exact_cfg.get("verify", False)),
                )

                print("[PIPELINE] Running exact stage...")
                exact_artifacts = run_exact_stage(
                    root=root,
                    candidates_csv=candidates_csv,
                    valid_csv=valid_csv,
                    modality=str(exact_cfg.get("modality", "T1w")),
                    out=str(out_dir),
                    thread_workers=int(exact_cfg.get("thread_workers", 4)),
                    process_workers=int(exact_cfg.get("process_workers", 4)),
                    verify=bool(exact_cfg.get("verify", False)),
                    profiler=profiler,
                )

            if "near" in stages:
                near_cfg = _require_table(cfg, "near")

                profiler.set_run_meta(
                    near_preprocess_workers=int(near_cfg.get("preprocess_workers", 1)),
                    near_batch_size=int(near_cfg.get("batch_size", 16)),
                    near_device=str(near_cfg.get("device", "auto")),
                    near_use_amp=bool(near_cfg.get("use_amp", False)),
                    near_topk=int(near_cfg.get("topk", 500)),
                    near_retrieval_batch_size=int(
                        near_cfg.get("retrieval_batch_size", 2048)
                    ),
                    near_use_ivf=bool(near_cfg.get("use_ivf", True)),
                    near_nlist=int(near_cfg.get("nlist", 4096)),
                    near_nprobe=int(near_cfg.get("nprobe", 64)),
                    near_train_size=int(near_cfg.get("train_size", 200000)),
                    near_min_similarity=float(near_cfg.get("min_similarity", 0.8)),
                    near_review_mode=str(near_cfg.get("review_mode", "off")),
                    near_review_workers=int(near_cfg.get("review_workers", 24)),
                )

                print("[PIPELINE] Running near stage...")
                near_artifacts = run_near_stage(
                    out_dir=out_dir,
                    preprocess_workers=int(near_cfg.get("preprocess_workers", 1)),
                    batch_size=int(near_cfg.get("batch_size", 16)),
                    device=str(near_cfg.get("device", "auto")),
                    use_amp=bool(near_cfg.get("use_amp", False)),
                    normalize=bool(near_cfg.get("normalize", True)),
                    overwrite_embeddings=bool(
                        near_cfg.get("overwrite_embeddings", False)
                    ),
                    fail_fast=bool(near_cfg.get("fail_fast", False)),
                    crop_shape=_tuple3_int(
                        near_cfg.get("crop_shape", [160, 192, 160]),
                        "[near].crop_shape",
                    ),
                    axial_axis=int(near_cfg.get("axial_axis", 2)),
                    slices_2p5d=int(near_cfg.get("slices_2p5d", 32)),
                    slice_stride=int(near_cfg.get("slice_stride", 1)),
                    winsor=_tuple2_float(
                        near_cfg.get("winsor", [1.0, 99.0]),
                        "[near].winsor",
                    ),
                    hist_matching=bool(near_cfg.get("hist_matching", False)),
                    repro=bool(near_cfg.get("repro", False)),
                    delete_extras=bool(near_cfg.get("delete_extras", True)),
                    inflight_factor=int(near_cfg.get("inflight_factor", 2)),
                    topk=int(near_cfg.get("topk", 500)),
                    retrieval_batch_size=int(
                        near_cfg.get("retrieval_batch_size", 2048)
                    ),
                    use_ivf=bool(near_cfg.get("use_ivf", True)),
                    nlist=int(near_cfg.get("nlist", 4096)),
                    nprobe=int(near_cfg.get("nprobe", 64)),
                    train_size=int(near_cfg.get("train_size", 200000)),
                    seed=int(near_cfg.get("seed", 0)),
                    overwrite_retrieval=bool(
                        near_cfg.get("overwrite_retrieval", False)
                    ),
                    min_similarity=float(near_cfg.get("min_similarity", 0.92)),
                    review_mode=str(near_cfg.get("review_mode", "off")),
                    review_workers=int(near_cfg.get("review_workers", 24)),
                    overwrite_review_assets=bool(
                        near_cfg.get("overwrite_review_assets", False)
                    ),
                    max_scan_candidates_per_query=_optional_int(
                        near_cfg, "max_scan_candidates_per_query"
                    ),
                    profiler=profiler,
                )

        print("[PIPELINE] Done.")
        print(f"[PIPELINE] outputs saved under -> {out_dir}")

        return PipelineArtifacts(
            config_path=config_path,
            out_dir=out_dir,
            stages=stages,
            exact=exact_artifacts,
            near=near_artifacts,
        )

    finally:
        if profile_runtime:
            try:
                perf_paths = profiler.save()
                print("[PIPELINE] Runtime profiling saved:")
                for k, v in perf_paths.items():
                    if v is not None:
                        print(f"[PIPELINE]   {k} -> {v}")
            except Exception as e:
                print(f"[PIPELINE][WARN] Failed to save runtime profiling outputs: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run pipeline.")
    ap.add_argument(
        "config",
        type=Path,
        help="Path to pipeline config TOML file",
    )
    args = ap.parse_args()

    run_pipeline(args.config)


if __name__ == "__main__":
    main()
