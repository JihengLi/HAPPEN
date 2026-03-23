"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm


def atomic_write_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp.replace(out_path)


def _read_manifest(
    manifest_csv: Union[str, Path],
    required_cols: Tuple[str, ...] = (
        "dataset",
        "subject_id",
        "session_id",
        "candidate",
        "resolved_path",
        "scan_uid",
        "emb_row",
    ),
) -> pd.DataFrame:
    manifest_csv = Path(manifest_csv)
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

    df = df.sort_values(["emb_row"], kind="mergesort").reset_index(drop=True)

    expected = np.arange(len(df), dtype=np.int64)
    if not np.array_equal(df["emb_row"].to_numpy(dtype=np.int64), expected):
        raise ValueError(
            f"{manifest_csv} emb_row must be consecutive 0..N-1 after sorting."
        )

    return df


def _read_embeddings(
    embeddings_npy: Union[str, Path],
    expected_rows: int,
) -> np.ndarray:
    embeddings_npy = Path(embeddings_npy)
    if not embeddings_npy.exists():
        raise FileNotFoundError(f"embeddings file not found: {embeddings_npy}")

    X = np.load(embeddings_npy)
    if X.ndim != 2:
        raise ValueError(f"{embeddings_npy} must be 2D, got shape={X.shape}")

    if X.shape[0] != expected_rows:
        raise ValueError(
            f"{embeddings_npy} row count {X.shape[0]} != manifest row count {expected_rows}"
        )

    return np.ascontiguousarray(X.astype(np.float32, copy=False))


def _l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return np.ascontiguousarray(X / norms, dtype=np.float32)


def _build_faiss_index(
    X: np.ndarray,
    use_ivf: bool = True,
    nlist: int = 4096,
    nprobe: int = 64,
    train_size: int = 200000,
    seed: int = 0,
):
    n, d = X.shape

    if (not use_ivf) or (n < max(10000, nlist * 10)):
        index = faiss.IndexFlatIP(d)
        index.add(X)
        return index, "flat"

    rng = np.random.default_rng(seed)
    train_n = min(int(train_size), n)
    train_idx = rng.choice(n, size=train_n, replace=False)
    train_x = np.ascontiguousarray(X[train_idx], dtype=np.float32)

    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, int(nlist), faiss.METRIC_INNER_PRODUCT)
    index.train(train_x)
    index.add(X)
    index.nprobe = int(nprobe)

    return index, "ivf"


def run_faiss_retrieval(
    manifest_csv: Union[str, Path],
    embeddings_npy: Union[str, Path],
    out_csv: Union[str, Path],
    topk: int = 500,
    batch_size: int = 2048,
    use_ivf: bool = True,
    nlist: int = 4096,
    nprobe: int = 64,
    train_size: int = 200000,
    seed: int = 0,
    overwrite: bool = False,
) -> Path:
    manifest_csv = Path(manifest_csv)
    embeddings_npy = Path(embeddings_npy)
    out_csv = Path(out_csv)

    if out_csv.exists() and (not overwrite):
        return out_csv

    manifest_df = _read_manifest(manifest_csv)
    n = len(manifest_df)

    if n == 0:
        empty_df = pd.DataFrame(
            columns=[
                "query_dataset",
                "query_subject_id",
                "query_session_id",
                "query_candidate",
                "query_resolved_path",
                "query_scan_uid",
                "cand_dataset",
                "cand_subject_id",
                "cand_session_id",
                "cand_candidate",
                "cand_resolved_path",
                "cand_scan_uid",
                "rank",
                "similarity",
            ]
        )
        atomic_write_csv(empty_df, out_csv)
        return out_csv

    X = _read_embeddings(embeddings_npy, expected_rows=n)
    X = _l2_normalize_rows(X)

    index, index_kind = _build_faiss_index(
        X,
        use_ivf=use_ivf,
        nlist=nlist,
        nprobe=nprobe,
        train_size=train_size,
        seed=seed,
    )

    search_k = min(int(topk) + 1, n)

    rows = []

    for start in tqdm(range(0, n, batch_size), desc="faiss_retrieval", unit="batch"):
        end = min(start + batch_size, n)
        q = X[start:end]

        sims, inds = index.search(q, search_k)

        for local_i in range(end - start):
            query_row = start + local_i
            qmeta = manifest_df.iloc[query_row]

            q_dataset = str(qmeta["dataset"])
            q_subject_id = str(qmeta["subject_id"])
            q_subject_key = (q_dataset, q_subject_id)

            rank = 0
            for sim, nn_idx in zip(sims[local_i], inds[local_i]):
                if nn_idx < 0:
                    continue
                if int(nn_idx) == int(query_row):
                    continue

                cmeta = manifest_df.iloc[int(nn_idx)]
                c_subject_key = (str(cmeta["dataset"]), str(cmeta["subject_id"]))

                if c_subject_key == q_subject_key:
                    continue

                rank += 1
                if rank > int(topk):
                    break

                rows.append(
                    {
                        "query_dataset": qmeta["dataset"],
                        "query_subject_id": qmeta["subject_id"],
                        "query_session_id": qmeta["session_id"],
                        "query_candidate": qmeta["candidate"],
                        "query_resolved_path": qmeta["resolved_path"],
                        "query_scan_uid": qmeta["scan_uid"],
                        "cand_dataset": cmeta["dataset"],
                        "cand_subject_id": cmeta["subject_id"],
                        "cand_session_id": cmeta["session_id"],
                        "cand_candidate": cmeta["candidate"],
                        "cand_resolved_path": cmeta["resolved_path"],
                        "cand_scan_uid": cmeta["scan_uid"],
                        "rank": int(rank),
                        "similarity": float(sim),
                    }
                )

    out_df = pd.DataFrame(
        rows,
        columns=[
            "query_dataset",
            "query_subject_id",
            "query_session_id",
            "query_candidate",
            "query_resolved_path",
            "query_scan_uid",
            "cand_dataset",
            "cand_subject_id",
            "cand_session_id",
            "cand_candidate",
            "cand_resolved_path",
            "cand_scan_uid",
            "rank",
            "similarity",
        ],
    )

    if not out_df.empty:
        out_df = out_df.sort_values(
            ["query_scan_uid", "rank", "cand_scan_uid"],
            kind="mergesort",
        ).reset_index(drop=True)

    atomic_write_csv(out_df, out_csv)

    print(f"[NEAR RETRIEVAL] scans in manifest: {n:,}")
    print(f"[NEAR RETRIEVAL] FAISS index type: {index_kind}")
    print(f"[NEAR RETRIEVAL] topk per query: {int(topk)}")
    print(f"[NEAR RETRIEVAL] output rows: {len(out_df):,}")
    print(f"[NEAR RETRIEVAL] saved -> {out_csv}")

    return out_csv
