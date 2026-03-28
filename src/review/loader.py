from __future__ import annotations

import re

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .store import DecisionStore


_CANDIDATE_SLOT_RE = re.compile(r"^candidate_(\d+)_")


def _norm_str(x: Any) -> str:
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


def _get_first_present(row: Dict[str, Any], keys: Sequence[str]) -> str:
    for k in keys:
        if k in row:
            v = _norm_str(row.get(k))
            if v:
                return v
    return ""


@dataclass(frozen=True)
class ScanRef:
    dataset: str
    subject_id: str
    session_id: str
    scan_uid: str
    src_path: str

    def is_renderable(self) -> bool:
        return bool(
            self.dataset and self.subject_id and self.scan_uid and self.src_path
        )

    def scan_folder_parts(self) -> Tuple[str, ...]:
        if self.session_id:
            return (self.dataset, self.subject_id, self.session_id, self.scan_uid)
        return (self.dataset, self.subject_id, self.scan_uid)


@dataclass(frozen=True)
class CandidateRef:
    scanref: ScanRef
    similarity: Optional[float]


@dataclass(frozen=True)
class QueryRecord:
    query: ScanRef
    candidates: Tuple[CandidateRef, ...]
    row_id: int

    @property
    def dataset(self) -> str:
        return self.query.dataset

    @property
    def subject_id(self) -> str:
        return self.query.subject_id


def load_in_csv(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str, na_filter=True)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def detect_candidate_slots(df: pd.DataFrame) -> List[int]:
    slots = set()
    for col in df.columns:
        m = _CANDIDATE_SLOT_RE.match(str(col).strip())
        if m:
            slots.add(int(m.group(1)))
    return sorted(slots)


def _parse_scanref_from_row(row: Dict[str, Any], prefix: str) -> ScanRef:
    dataset = _get_first_present(row, [f"{prefix}dataset"])
    subject_id = _get_first_present(row, [f"{prefix}subject_id"])
    session_id = _get_first_present(row, [f"{prefix}session_id"])
    scan_uid = _get_first_present(row, [f"{prefix}scan_uid", f"{prefix}scan"])
    src_path = _get_first_present(
        row,
        [
            f"{prefix}src_path",
            f"{prefix}path",
            f"{prefix}resolved_path",
        ],
    )

    return ScanRef(
        dataset=dataset,
        subject_id=subject_id,
        session_id=session_id,
        scan_uid=scan_uid,
        src_path=src_path,
    )


def _parse_similarity(row: Dict[str, Any], slot: int) -> Optional[float]:
    prefix = f"candidate_{slot}_"
    s = _get_first_present(
        row,
        [
            f"{prefix}similarity",
            f"{prefix}sim",
            f"{prefix}score",
        ],
    )
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def parse_queries(df: pd.DataFrame, slots: Sequence[int]) -> List[QueryRecord]:
    records: List[Dict[str, Any]] = df.to_dict(orient="records")
    out: List[QueryRecord] = []

    for i, row in enumerate(records):
        query = _parse_scanref_from_row(row, "query_")
        if not query.is_renderable():
            continue

        candidates: List[CandidateRef] = []
        for slot in slots:
            candidate = _parse_scanref_from_row(row, f"candidate_{slot}_")
            if not candidate.is_renderable():
                continue

            similarity = _parse_similarity(row, slot)
            candidates.append(
                CandidateRef(
                    scanref=candidate,
                    similarity=similarity,
                )
            )

        if not candidates:
            continue

        out.append(
            QueryRecord(
                query=query,
                candidates=tuple(candidates),
                row_id=i,
            )
        )

    return out


def build_index(
    queries: Sequence[QueryRecord],
) -> Dict[str, Dict[str, List[QueryRecord]]]:
    index: Dict[str, Dict[str, List[QueryRecord]]] = {}
    for qr in queries:
        dataset = qr.dataset
        subject_id = qr.subject_id
        if not dataset or not subject_id:
            continue
        index.setdefault(dataset, {}).setdefault(subject_id, []).append(qr)
    return index


def _scan_item_key(scan: ScanRef) -> str:
    return DecisionStore.make_item_key(
        dataset=scan.dataset,
        subject_id=scan.subject_id,
        session_id=scan.session_id,
        scan_uid=scan.scan_uid,
        src_path=scan.src_path,
    )


def dedup_undirected_pairs(queries: Sequence[QueryRecord]) -> List[QueryRecord]:
    occ: Dict[Tuple[str, str], List[Tuple[int, int, float, bool]]] = {}

    for qi, qr in enumerate(queries):
        a_key = _scan_item_key(qr.query)

        for ci, cand in enumerate(qr.candidates):
            b_key = _scan_item_key(cand.scanref)

            pair_key = (a_key, b_key) if a_key <= b_key else (b_key, a_key)
            similarity = cand.similarity
            sim_val = float(similarity) if similarity is not None else float("-inf")
            is_canon = a_key <= b_key

            occ.setdefault(pair_key, []).append((qi, ci, sim_val, is_canon))

    keep_pairs = set()
    for items in occ.values():
        if len(items) == 1:
            qi, ci, _, _ = items[0]
            keep_pairs.add((qi, ci))
            continue

        items_sorted = sorted(
            items,
            key=lambda t: (-t[2], -int(t[3]), t[0], t[1]),
        )
        qi, ci, _, _ = items_sorted[0]
        keep_pairs.add((qi, ci))

    out: List[QueryRecord] = []
    for qi, qr in enumerate(queries):
        new_candidates = [
            cand for ci, cand in enumerate(qr.candidates) if (qi, ci) in keep_pairs
        ]
        if new_candidates:
            out.append(
                QueryRecord(
                    query=qr.query,
                    candidates=tuple(new_candidates),
                    row_id=qr.row_id,
                )
            )

    return out


def collect_valid_item_keys(queries: Sequence[QueryRecord]) -> set[str]:
    valid = set()

    for qr in queries:
        valid.add(_scan_item_key(qr.query))
        for cand in qr.candidates:
            valid.add(_scan_item_key(cand.scanref))

    return valid
