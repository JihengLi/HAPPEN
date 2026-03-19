from __future__ import annotations

import re
import pandas as pd

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from store import DecisionStore


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


def _norm_scan_uid(x: Any) -> str:
    return _norm_str(x)


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
        return bool(self.dataset and self.subject_id and self.scan_uid)

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


def load_in_csv(csv_path: str | pd.PathLike) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str, na_filter=True)
    df.columns = [c.strip() for c in df.columns]
    return df


def detect_candidate_slots(df: pd.DataFrame) -> List[int]:
    slots = set()
    for col in df.columns:
        m = re.compile(r"^candidate_(\d+)_").match(col)
        if m:
            slots.add(int(m.group(1)))
    return sorted(slots)


def _parse_scanref_from_row(row: Dict[str, Any], prefix: str) -> ScanRef:
    dataset = _norm_str(row.get(f"{prefix}dataset"))
    subject_id = _norm_str(row.get(f"{prefix}subject_id"))
    session_id = _norm_str(row.get(f"{prefix}session_id"))
    scan_uid = _norm_scan_uid(row.get(f"{prefix}scan_uid"))
    src_path = _norm_str(row.get(f"{prefix}path"))
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
        row, [f"{prefix}similarity", f"{prefix}sim", f"{prefix}score"]
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
        q = _parse_scanref_from_row(row, "query_")
        if not q.is_renderable():
            continue

        cands: List[CandidateRef] = []
        for k in slots:
            cand_prefix = f"candidate_{k}_"
            c = _parse_scanref_from_row(row, cand_prefix)

            if not c.is_renderable():
                continue

            sim = _parse_similarity(row, k)
            cands.append(CandidateRef(scanref=c, similarity=sim))

        if not cands:
            continue

        out.append(QueryRecord(query=q, candidates=tuple(cands), row_id=i))

    return out


def build_index(
    queries: Sequence[QueryRecord],
) -> Dict[str, Dict[str, List[QueryRecord]]]:
    index: Dict[str, Dict[str, List[QueryRecord]]] = {}
    for qr in queries:
        ds = qr.dataset
        sbj = qr.subject_id
        if not ds or not sbj:
            continue
        index.setdefault(ds, {}).setdefault(sbj, []).append(qr)
    return index


def dedup_undirected_pairs(queries: Sequence[QueryRecord]) -> List[QueryRecord]:
    occ: Dict[Tuple[str, str], List[Tuple[int, int, float, bool]]] = {}
    for qi, qr in enumerate(queries):
        a_uid = qr.query.scan_uid
        for ci, cand in enumerate(qr.candidates):
            b_uid = cand.scanref.scan_uid
            k = (a_uid, b_uid) if a_uid <= b_uid else (b_uid, a_uid)
            sim = cand.similarity
            simv = float(sim) if (sim is not None) else float("-inf")
            is_canon = a_uid <= b_uid
            occ.setdefault(k, []).append((qi, ci, simv, is_canon))

    keep_pairs = set()
    for k, items in occ.items():
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
        new_cands = [c for ci, c in enumerate(qr.candidates) if (qi, ci) in keep_pairs]
        if new_cands:
            out.append(
                QueryRecord(
                    query=qr.query, candidates=tuple(new_cands), row_id=qr.row_id
                )
            )
    return out


def collect_valid_item_keys(queries: Sequence[QueryRecord]) -> set[str]:
    valid = set()

    for q in queries:
        qref = q.query
        qk = DecisionStore.make_item_key(
            dataset=qref.dataset,
            subject_id=qref.subject_id,
            session_id=qref.session_id,
            scan_uid=qref.scan_uid,
            src_path=qref.src_path,
        )
        valid.add(qk)

        for c in q.candidates:
            cref = c.scanref
            ck = DecisionStore.make_item_key(
                dataset=cref.dataset,
                subject_id=cref.subject_id,
                session_id=cref.session_id,
                scan_uid=cref.scan_uid,
                src_path=cref.src_path,
            )
            valid.add(ck)

    return valid
