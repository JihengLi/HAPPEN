"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

from __future__ import annotations

import os
import shutil
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import nullcontext
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from .model import choose_device, load_model
from .preprocessing import preprocess_resolved_path_2p5d


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


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(x))
    if n <= eps:
        return x
    return x / n


def atomic_save_npy(out_path: Path, arr: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, out_path)


def atomic_write_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, out_path)


def _safe_unlink(path: Union[str, Path]) -> None:
    try:
        Path(path).unlink()
    except Exception:
        pass


def _safe_rmtree(path: Union[str, Path]) -> None:
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def _read_input_csv(
    in_csv: Union[str, Path],
    required_cols: Tuple[str, ...] = (
        "dataset",
        "subject_id",
        "session_id",
        "candidate",
        "resolved_path",
        "scan_uid",
    ),
    final_sort_keys: Tuple[str, ...] = (
        "scan_uid",
        "dataset",
        "subject_id",
        "session_id",
        "candidate",
        "resolved_path",
    ),
) -> pd.DataFrame:
    in_csv = Path(in_csv)
    df = pd.read_csv(in_csv, dtype=str).fillna("")

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{in_csv} is missing required columns: {missing}")

    for c in required_cols:
        df[c] = df[c].fillna("").astype(str)

    if final_sort_keys:
        df = df.sort_values(list(final_sort_keys), kind="mergesort").reset_index(
            drop=True
        )

    if df["scan_uid"].duplicated().any():
        dups = df.loc[df["scan_uid"].duplicated(), "scan_uid"].tolist()[:10]
        raise ValueError(f"{in_csv} contains duplicated scan_uid values: {dups}")

    return df


def _read_existing_manifest(manifest_csv: Path) -> pd.DataFrame:
    required_cols = [
        "dataset",
        "subject_id",
        "session_id",
        "candidate",
        "resolved_path",
        "scan_uid",
        "emb_row",
    ]

    if not manifest_csv.exists() or manifest_csv.stat().st_size == 0:
        return pd.DataFrame(columns=required_cols)

    df = pd.read_csv(manifest_csv, dtype=str).fillna("")
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{manifest_csv} is missing required columns: {missing}")

    for c in required_cols:
        df[c] = df[c].fillna("").astype(str)

    df["emb_row"] = pd.to_numeric(df["emb_row"], errors="raise").astype(np.int64)

    if df["scan_uid"].duplicated().any():
        dups = df.loc[df["scan_uid"].duplicated(), "scan_uid"].tolist()[:10]
        raise ValueError(f"{manifest_csv} contains duplicated scan_uid values: {dups}")

    return df


def _read_existing_failures(failures_csv: Path) -> pd.DataFrame:
    preferred_cols = [
        "dataset",
        "subject_id",
        "session_id",
        "candidate",
        "resolved_path",
        "scan_uid",
        "stage",
        "error",
    ]

    if not failures_csv.exists() or failures_csv.stat().st_size == 0:
        return pd.DataFrame(columns=preferred_cols)

    df = pd.read_csv(failures_csv, dtype=str).fillna("")
    for c in preferred_cols:
        if c not in df.columns:
            df[c] = ""
    df = df[preferred_cols].copy()

    return df


def _model_emb_dim(model: nn.Module) -> int:
    try:
        return int(model.encoder.fc[2].out_features)
    except Exception:
        return 512


@torch.inference_mode()
def encode_batch(
    model: nn.Module,
    batch_x: torch.Tensor,
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    batch_x = batch_x.to(device, non_blocking=True)

    if use_amp and device.type == "cuda":
        ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        ctx = nullcontext()

    with ctx:
        emb = model.encode(batch_x)

    emb = emb.detach().float().cpu().numpy()
    return np.asarray(emb, dtype=np.float32)


def _validate_temp_tensor(temp_tensor_path: Path) -> None:
    x = np.load(temp_tensor_path)
    if x.ndim != 4:
        raise ValueError(f"expected preprocessed tensor ndim=4, got shape={x.shape}")


def _read_partial_embedding_map(
    partial_emb_dir: Path,
    input_uid_set: set[str],
    ignore_uid_set: Optional[set[str]] = None,
) -> Dict[str, np.ndarray]:
    vec_map: Dict[str, np.ndarray] = {}
    ignore_uid_set = ignore_uid_set or set()

    if not partial_emb_dir.exists():
        return vec_map

    for tmp in partial_emb_dir.glob("*.tmp"):
        _safe_unlink(tmp)

    for emb_path in sorted(partial_emb_dir.glob("*.npy")):
        uid = emb_path.stem

        if uid in ignore_uid_set:
            continue
        if uid not in input_uid_set:
            continue

        try:
            z = np.load(emb_path)
            z = np.asarray(z, dtype=np.float32)
            if z.ndim != 1:
                raise ValueError(f"expected 1D embedding vector, got shape={z.shape}")
        except Exception:
            _safe_unlink(emb_path)
            continue

        if uid in vec_map:
            raise ValueError(f"duplicated partial embedding for scan_uid={uid}")

        vec_map[uid] = z

    return vec_map


def _preprocess_one_worker(
    row: Dict[str, str],
    temp_root: str,
    crop_shape: Tuple[int, int, int],
    axial_axis: int,
    slices_2p5d: int,
    slice_stride: int,
    winsor: Tuple[float, float],
    hist_matching: bool,
    repro: bool,
    delete_extras: bool,
) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    ds = clean_str(row.get("dataset", ""))
    sbj = clean_str(row.get("subject_id", ""))
    ses = clean_str(row.get("session_id", ""))
    cand = clean_str(row.get("candidate", ""))
    rpath = clean_str(row.get("resolved_path", ""))
    scan_uid = clean_str(row.get("scan_uid", ""))

    if not (ds and sbj and cand and rpath and scan_uid):
        return None, "missing required input metadata"

    temp_root_p = Path(temp_root).resolve()
    temp_preproc_dir = temp_root_p / "preprocessed"
    temp_preproc_dir.mkdir(parents=True, exist_ok=True)

    temp_tensor_path = temp_preproc_dir / f"{scan_uid}.npy"
    work_dir = temp_root_p / "work" / scan_uid

    try:
        if temp_tensor_path.exists():
            _validate_temp_tensor(temp_tensor_path)
        else:
            x_np = preprocess_resolved_path_2p5d(
                resolved_path=rpath,
                work_dir=work_dir,
                crop_shape=crop_shape,
                axial_axis=axial_axis,
                slices_2p5d=slices_2p5d,
                slice_stride=slice_stride,
                winsor=winsor,
                hist_matching=hist_matching,
                repro=repro,
                delete_extras=delete_extras,
                cleanup=True,
            )
            atomic_save_npy(temp_tensor_path, x_np)

        meta = {
            "dataset": ds,
            "subject_id": sbj,
            "session_id": ses,
            "candidate": cand,
            "resolved_path": rpath,
            "scan_uid": scan_uid,
            "temp_tensor_path": str(temp_tensor_path),
        }
        return meta, None

    except Exception as e:
        return None, str(e)


def _flush_embedding_buffer(
    model: nn.Module,
    buf_x: List[torch.Tensor],
    buf_meta: List[Dict[str, str]],
    device: torch.device,
    use_amp: bool,
    normalize: bool,
    partial_emb_dir: Path,
    failures: List[Dict[str, str]],
) -> int:
    if not buf_x:
        return 0

    partial_emb_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0

    try:
        X = torch.stack(buf_x, dim=0)
        Z = encode_batch(model, X, device=device, use_amp=use_amp)

        for z, meta in zip(Z, buf_meta):
            try:
                if normalize:
                    z = l2_normalize(z)

                z = np.asarray(z, dtype=np.float32)
                if z.ndim != 1:
                    raise ValueError(
                        f"expected 1D embedding vector, got shape={z.shape}"
                    )

                out_path = partial_emb_dir / f'{meta["scan_uid"]}.npy'
                atomic_save_npy(out_path, z)
                _safe_unlink(meta["temp_tensor_path"])
                n_written += 1

            except Exception as e:
                failures.append(
                    {
                        "dataset": meta["dataset"],
                        "subject_id": meta["subject_id"],
                        "session_id": meta["session_id"],
                        "candidate": meta["candidate"],
                        "resolved_path": meta["resolved_path"],
                        "scan_uid": meta["scan_uid"],
                        "stage": "embedding_persist",
                        "error": str(e),
                    }
                )

    except Exception as batch_e:
        tqdm.write(
            f"[WARN] batch inference failed; fallback to per-scan. error={batch_e}"
        )

        for x, meta in zip(buf_x, buf_meta):
            try:
                X1 = x.unsqueeze(0)
                z = encode_batch(model, X1, device=device, use_amp=use_amp)[0]

                if normalize:
                    z = l2_normalize(z)

                z = np.asarray(z, dtype=np.float32)
                if z.ndim != 1:
                    raise ValueError(
                        f"expected 1D embedding vector, got shape={z.shape}"
                    )

                out_path = partial_emb_dir / f'{meta["scan_uid"]}.npy'
                atomic_save_npy(out_path, z)
                _safe_unlink(meta["temp_tensor_path"])
                n_written += 1

            except Exception as e:
                failures.append(
                    {
                        "dataset": meta["dataset"],
                        "subject_id": meta["subject_id"],
                        "session_id": meta["session_id"],
                        "candidate": meta["candidate"],
                        "resolved_path": meta["resolved_path"],
                        "scan_uid": meta["scan_uid"],
                        "stage": "inference",
                        "error": str(e),
                    }
                )

    buf_x.clear()
    buf_meta.clear()
    return n_written


def run_embedding_inference(
    in_csv: Union[str, Path],
    out_dir: Union[str, Path],
    preprocess_workers: int = 1,
    batch_size: int = 16,
    device: str = "auto",
    use_amp: bool = False,
    normalize: bool = True,
    overwrite: bool = False,
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
) -> Tuple[Path, Path, Path]:
    in_csv = Path(in_csv)
    out_dir = Path(out_dir).expanduser().resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    embeddings_npy = out_dir / "embeddings.npy"
    manifest_csv = out_dir / "embedding_manifest.csv"
    failures_csv = out_dir / "embedding_failures.csv"
    temp_root = out_dir / "_tmp_preproc"
    partial_root = out_dir / "_partial_embeddings"
    partial_emb_dir = partial_root / "by_scan"

    if overwrite:
        _safe_unlink(embeddings_npy)
        _safe_unlink(manifest_csv)
        _safe_unlink(failures_csv)
        _safe_rmtree(temp_root)
        _safe_rmtree(partial_root)

    input_df = _read_input_csv(in_csv)
    input_uid_set = set(input_df["scan_uid"].astype(str).tolist())

    existing_manifest_df = pd.DataFrame()
    existing_failures_df = pd.DataFrame()

    if not overwrite:
        existing_manifest_df = _read_existing_manifest(manifest_csv)
        existing_failures_df = _read_existing_failures(failures_csv)

    existing_vec_map: Dict[str, np.ndarray] = {}
    finalized_uid_set: set[str] = set()

    if not existing_manifest_df.empty:
        if not embeddings_npy.exists():
            raise FileNotFoundError(
                f"{manifest_csv} exists but embeddings file is missing: {embeddings_npy}"
            )

        old_embs = np.load(embeddings_npy)
        if old_embs.ndim != 2:
            raise ValueError(f"{embeddings_npy} must be 2D, got shape={old_embs.shape}")
        if old_embs.shape[0] != len(existing_manifest_df):
            raise ValueError(
                f"row mismatch: {embeddings_npy} has {old_embs.shape[0]} rows but "
                f"{manifest_csv} has {len(existing_manifest_df)} rows"
            )

        existing_manifest_df = existing_manifest_df[
            existing_manifest_df["scan_uid"].isin(set(input_df["scan_uid"]))
        ].copy()

        for _, r in existing_manifest_df.iterrows():
            uid = str(r["scan_uid"])
            row_idx = int(r["emb_row"])
            existing_vec_map[uid] = np.asarray(old_embs[row_idx], dtype=np.float32)
            finalized_uid_set.add(uid)

    partial_vec_map = _read_partial_embedding_map(
        partial_emb_dir=partial_emb_dir,
        input_uid_set=input_uid_set,
        ignore_uid_set=finalized_uid_set,
    )
    partial_uid_set = set(partial_vec_map.keys())

    existing_ok_uids: set[str] = finalized_uid_set | partial_uid_set

    pending_df = input_df[~input_df["scan_uid"].isin(existing_ok_uids)].copy()
    pending_df = pending_df.sort_values(["scan_uid"], kind="mergesort").reset_index(
        drop=True
    )

    dev = choose_device(device)
    use_amp = bool(use_amp) and dev.type == "cuda"
    model = load_model(device=dev)
    emb_dim = _model_emb_dim(model)

    failures: List[Dict[str, str]] = []
    new_success_count = 0

    preprocess_workers = max(1, int(preprocess_workers))
    max_inflight = max(1, preprocess_workers * max(1, int(inflight_factor)))

    buf_x: List[torch.Tensor] = []
    buf_meta: List[Dict[str, str]] = []

    records = pending_df.to_dict(orient="records")

    try:
        with ProcessPoolExecutor(
            max_workers=preprocess_workers,
            mp_context=get_context("spawn"),
        ) as ex, tqdm(total=len(records), desc="embed_near", unit="scan") as pbar:
            rec_iter = iter(records)
            fut2row: Dict[Any, Dict[str, str]] = {}

            def _submit_one() -> bool:
                try:
                    row = next(rec_iter)
                except StopIteration:
                    return False

                fut = ex.submit(
                    _preprocess_one_worker,
                    row,
                    str(temp_root),
                    crop_shape,
                    int(axial_axis),
                    int(slices_2p5d),
                    int(slice_stride),
                    winsor,
                    bool(hist_matching),
                    bool(repro),
                    bool(delete_extras),
                )
                fut2row[fut] = row
                return True

            while len(fut2row) < max_inflight and _submit_one():
                pass

            stop_early = False

            while fut2row:
                done, _ = wait(fut2row.keys(), return_when=FIRST_COMPLETED)

                for fut in done:
                    row = fut2row.pop(fut)

                    meta, err = fut.result()

                    if meta is not None:
                        try:
                            x_np = np.load(meta["temp_tensor_path"])
                            if x_np.ndim != 4:
                                raise ValueError(
                                    f"expected preprocessed tensor ndim=4, got shape={x_np.shape}"
                                )
                            x_t = torch.from_numpy(
                                np.ascontiguousarray(
                                    x_np.astype(np.float32, copy=False)
                                )
                            )
                            buf_x.append(x_t)
                            buf_meta.append(meta)

                            if len(buf_x) >= int(batch_size):
                                new_success_count += _flush_embedding_buffer(
                                    model=model,
                                    buf_x=buf_x,
                                    buf_meta=buf_meta,
                                    device=dev,
                                    use_amp=use_amp,
                                    normalize=bool(normalize),
                                    partial_emb_dir=partial_emb_dir,
                                    failures=failures,
                                )

                        except Exception as e:
                            failures.append(
                                {
                                    "dataset": clean_str(row.get("dataset", "")),
                                    "subject_id": clean_str(row.get("subject_id", "")),
                                    "session_id": clean_str(row.get("session_id", "")),
                                    "candidate": clean_str(row.get("candidate", "")),
                                    "resolved_path": clean_str(
                                        row.get("resolved_path", "")
                                    ),
                                    "scan_uid": clean_str(row.get("scan_uid", "")),
                                    "stage": "preprocessed_tensor_load",
                                    "error": str(e),
                                }
                            )
                            if fail_fast:
                                stop_early = True

                    else:
                        failures.append(
                            {
                                "dataset": clean_str(row.get("dataset", "")),
                                "subject_id": clean_str(row.get("subject_id", "")),
                                "session_id": clean_str(row.get("session_id", "")),
                                "candidate": clean_str(row.get("candidate", "")),
                                "resolved_path": clean_str(
                                    row.get("resolved_path", "")
                                ),
                                "scan_uid": clean_str(row.get("scan_uid", "")),
                                "stage": "preprocessing",
                                "error": clean_str(err),
                            }
                        )
                        if fail_fast:
                            stop_early = True

                    pbar.update(1)

                    if stop_early:
                        break

                    while len(fut2row) < max_inflight and _submit_one():
                        pass

                if stop_early:
                    break

            new_success_count += _flush_embedding_buffer(
                model=model,
                buf_x=buf_x,
                buf_meta=buf_meta,
                device=dev,
                use_amp=use_amp,
                normalize=bool(normalize),
                partial_emb_dir=partial_emb_dir,
                failures=failures,
            )

    except KeyboardInterrupt:
        tqdm.write(
            "[WARN] KeyboardInterrupt received. Flushing in-memory buffer before exit..."
        )
        try:
            new_success_count += _flush_embedding_buffer(
                model=model,
                buf_x=buf_x,
                buf_meta=buf_meta,
                device=dev,
                use_amp=use_amp,
                normalize=bool(normalize),
                partial_emb_dir=partial_emb_dir,
                failures=failures,
            )
        except Exception as e:
            tqdm.write(f"[ERROR] final interrupt flush failed: {e}")
        raise

    new_vec_map = _read_partial_embedding_map(
        partial_emb_dir=partial_emb_dir,
        input_uid_set=input_uid_set,
        ignore_uid_set=finalized_uid_set,
    )

    all_success_rows: List[Dict[str, str]] = []
    all_success_embs: List[np.ndarray] = []

    for _, r in input_df.sort_values(["scan_uid"], kind="mergesort").iterrows():
        uid = str(r["scan_uid"])
        if uid in new_vec_map:
            vec = new_vec_map[uid]
        elif uid in existing_vec_map:
            vec = existing_vec_map[uid]
        else:
            continue

        out_row = {
            "dataset": str(r["dataset"]),
            "subject_id": str(r["subject_id"]),
            "session_id": str(r["session_id"]),
            "candidate": str(r["candidate"]),
            "resolved_path": str(r["resolved_path"]),
            "scan_uid": str(r["scan_uid"]),
            "emb_row": str(len(all_success_rows)),
        }
        all_success_rows.append(out_row)
        all_success_embs.append(np.asarray(vec, dtype=np.float32))

    if all_success_embs:
        emb_matrix = np.stack(all_success_embs, axis=0).astype(np.float32, copy=False)
    else:
        emb_matrix = np.zeros((0, emb_dim), dtype=np.float32)

    manifest_df = pd.DataFrame(
        all_success_rows,
        columns=[
            "dataset",
            "subject_id",
            "session_id",
            "candidate",
            "resolved_path",
            "scan_uid",
            "emb_row",
        ],
    )

    old_fail_map: Dict[str, Dict[str, str]] = {}
    if not existing_failures_df.empty:
        for _, r in existing_failures_df.iterrows():
            uid = str(r.get("scan_uid", "")).strip()
            if uid:
                old_fail_map[uid] = {k: str(v) for k, v in r.to_dict().items()}

    new_fail_map: Dict[str, Dict[str, str]] = {}
    for r in failures:
        uid = str(r.get("scan_uid", "")).strip()
        if uid:
            new_fail_map[uid] = {k: str(v) for k, v in r.items()}

    success_uid_set = set(manifest_df["scan_uid"].astype(str).tolist())
    current_uid_set = set(input_df["scan_uid"].astype(str).tolist())

    merged_fail_map: Dict[str, Dict[str, str]] = {}
    for uid, row in old_fail_map.items():
        if uid in current_uid_set and uid not in success_uid_set:
            merged_fail_map[uid] = row
    for uid, row in new_fail_map.items():
        if uid in current_uid_set and uid not in success_uid_set:
            merged_fail_map[uid] = row

    failure_cols = [
        "dataset",
        "subject_id",
        "session_id",
        "candidate",
        "resolved_path",
        "scan_uid",
        "stage",
        "error",
    ]
    failures_df = pd.DataFrame(
        list(merged_fail_map.values()),
        columns=failure_cols,
    )

    if not failures_df.empty:
        failures_df = failures_df.sort_values(
            ["scan_uid", "dataset", "subject_id", "session_id", "candidate"],
            kind="mergesort",
        ).reset_index(drop=True)

    manifest_df["emb_row"] = pd.to_numeric(
        manifest_df["emb_row"], errors="raise"
    ).astype(np.int64)

    manifest_df = manifest_df.sort_values(["scan_uid"], kind="mergesort").reset_index(
        drop=True
    )
    manifest_df["emb_row"] = np.arange(len(manifest_df), dtype=np.int64)

    if len(manifest_df) != emb_matrix.shape[0]:
        raise RuntimeError(
            f"manifest rows ({len(manifest_df)}) != embedding rows ({emb_matrix.shape[0]})"
        )

    atomic_save_npy(embeddings_npy, emb_matrix)
    atomic_write_csv(manifest_df, manifest_csv)
    atomic_write_csv(failures_df, failures_csv)

    _safe_rmtree(partial_root)

    print(f"[NEAR EMBED] input rows: {len(input_df):,}")
    print(f"[NEAR EMBED] already completed rows reused: {len(existing_ok_uids):,}")
    print(f"[NEAR EMBED] new successful rows: {new_success_count:,}")
    print(f"[NEAR EMBED] total successful rows: {len(manifest_df):,}")
    print(f"[NEAR EMBED] total failed rows: {len(failures_df):,}")
    print(f"[NEAR EMBED] embeddings -> {embeddings_npy}")
    print(f"[NEAR EMBED] manifest   -> {manifest_csv}")
    print(f"[NEAR EMBED] failures   -> {failures_csv}")

    return embeddings_npy, manifest_csv, failures_csv
