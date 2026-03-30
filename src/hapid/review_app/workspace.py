"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TypeAlias

from . import loader
from .store import DecisionStore


IndexType: TypeAlias = Dict[str, Dict[str, List[loader.QueryRecord]]]
SubjectRow: TypeAlias = Dict[str, Any]
SubjectPayload: TypeAlias = Dict[str, Any]
DatasetPayload: TypeAlias = Dict[str, Any]


def _clean(s: Optional[str]) -> str:
    return (s or "").strip()


def build_subject_data(
    query_records: List[loader.QueryRecord],
) -> List[Dict[str, Any]]:
    subject_data: List[Dict[str, Any]] = []

    for qr in query_records:
        query = qr.query
        candidates: List[Dict[str, Any]] = []

        for cand in qr.candidates:
            scan = cand.scanref
            candidates.append(
                {
                    "dataset": scan.dataset,
                    "subject_id": scan.subject_id,
                    "session_id": scan.session_id,
                    "scan_uid": scan.scan_uid,
                    "src_path": scan.src_path,
                    "similarity": cand.similarity,
                }
            )

        subject_data.append(
            {
                "query": {
                    "dataset": query.dataset,
                    "subject_id": query.subject_id,
                    "session_id": query.session_id,
                    "scan_uid": query.scan_uid,
                    "src_path": query.src_path,
                },
                "candidates": candidates,
            }
        )

    return subject_data


def build_subject_pair_progress(
    index: IndexType,
    dataset: str,
    subject_id: str,
    decision_store: DecisionStore,
) -> Tuple[int, int]:
    total = 0
    done = 0

    query_records = index.get(dataset, {}).get(subject_id, [])
    for qr in query_records:
        for cand in qr.candidates:
            key = decision_store.make_pair_key(
                a_dataset=qr.query.dataset,
                a_subject_id=qr.query.subject_id,
                a_session_id=qr.query.session_id,
                a_scan_uid=qr.query.scan_uid,
                a_src_path=qr.query.src_path,
                b_dataset=cand.scanref.dataset,
                b_subject_id=cand.scanref.subject_id,
                b_session_id=cand.scanref.session_id,
                b_scan_uid=cand.scanref.scan_uid,
                b_src_path=cand.scanref.src_path,
            )
            total += 1
            if decision_store.is_decided(key):
                done += 1

    return done, total


def build_dataset_subject_progress(
    index: IndexType,
    dataset: str,
    decision_store: DecisionStore,
) -> Tuple[int, int]:
    subject_map = index.get(dataset, {})
    total_subjects = len(subject_map)
    done_subjects = 0

    for subject_id in subject_map.keys():
        done_pairs, total_pairs = build_subject_pair_progress(
            index=index,
            dataset=dataset,
            subject_id=subject_id,
            decision_store=decision_store,
        )
        if total_pairs > 0 and done_pairs == total_pairs:
            done_subjects += 1

    return done_subjects, total_subjects


def build_workspace_subject_payload(
    index: IndexType,
    dataset: str,
    subject_id: str,
    decision_store: DecisionStore,
) -> SubjectPayload:
    dataset = _clean(dataset)
    subject_id = _clean(subject_id)

    if not dataset or dataset not in index:
        raise KeyError(f"unknown dataset: {dataset}")
    if not subject_id or subject_id not in index[dataset]:
        raise KeyError(f"unknown subject in dataset {dataset}: {subject_id}")

    subjects = sorted(index[dataset].keys())
    pos = subjects.index(subject_id)

    prev_subject_id = subjects[pos - 1] if pos > 0 else None
    next_subject_id = subjects[pos + 1] if pos < (len(subjects) - 1) else None

    query_records = index[dataset][subject_id]
    subject_data = build_subject_data(query_records)

    done_pairs, total_pairs = build_subject_pair_progress(
        index=index,
        dataset=dataset,
        subject_id=subject_id,
        decision_store=decision_store,
    )

    return {
        "dataset": dataset,
        "subject_id": subject_id,
        "done_pairs": done_pairs,
        "total_pairs": total_pairs,
        "is_done": (total_pairs > 0 and done_pairs == total_pairs),
        "prev_subject_id": prev_subject_id,
        "next_subject_id": next_subject_id,
        "subject_data": subject_data,
    }


def build_workspace_dataset_payload(
    index: IndexType,
    dataset: str,
    decision_store: DecisionStore,
    include_subject_data: bool = True,
) -> DatasetPayload:
    dataset = _clean(dataset)

    if not dataset or dataset not in index:
        raise KeyError(f"unknown dataset: {dataset}")

    subjects = sorted(index[dataset].keys())
    subject_rows: List[SubjectRow] = []
    review_by_subject: Dict[str, SubjectPayload] = {}

    for subject_id in subjects:
        done_pairs, total_pairs = build_subject_pair_progress(
            index=index,
            dataset=dataset,
            subject_id=subject_id,
            decision_store=decision_store,
        )

        row: SubjectRow = {
            "subject_id": subject_id,
            "done_pairs": done_pairs,
            "total_pairs": total_pairs,
            "is_done": (total_pairs > 0 and done_pairs == total_pairs),
        }
        subject_rows.append(row)

        if include_subject_data:
            review_by_subject[subject_id] = build_workspace_subject_payload(
                index=index,
                dataset=dataset,
                subject_id=subject_id,
                decision_store=decision_store,
            )

    done_subjects, total_subjects = build_dataset_subject_progress(
        index=index,
        dataset=dataset,
        decision_store=decision_store,
    )

    payload: DatasetPayload = {
        "dataset": dataset,
        "done_subjects": done_subjects,
        "total_subjects": total_subjects,
        "subjects": subject_rows,
    }

    if include_subject_data:
        payload["review_by_subject"] = review_by_subject

    return payload
