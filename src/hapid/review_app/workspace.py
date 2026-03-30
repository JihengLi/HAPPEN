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
DecisionJSON: TypeAlias = Dict[str, Any]
DecisionMap: TypeAlias = Dict[str, DecisionJSON]


def _clean(s: Optional[str]) -> str:
    return (s or "").strip()


def _scan_to_payload(scan: Any) -> Dict[str, Any]:
    return {
        "dataset": scan.dataset,
        "subject_id": scan.subject_id,
        "session_id": scan.session_id,
        "scan_uid": scan.scan_uid,
        "src_path": scan.src_path,
    }


def _pair_key_for_query_candidate(
    query: Any,
    cand_scan: Any,
    decision_store: DecisionStore,
) -> Tuple[str, str]:
    return decision_store.make_pair_key(
        a_dataset=query.dataset,
        a_subject_id=query.subject_id,
        a_session_id=query.session_id,
        a_scan_uid=query.scan_uid,
        a_src_path=query.src_path,
        b_dataset=cand_scan.dataset,
        b_subject_id=cand_scan.subject_id,
        b_session_id=cand_scan.session_id,
        b_scan_uid=cand_scan.scan_uid,
        b_src_path=cand_scan.src_path,
    )


def _pair_key_to_str(key: Tuple[str, str]) -> str:
    return f"{key[0]}__{key[1]}"


def _decision_to_json(
    decision_store: DecisionStore,
    key: Tuple[str, str],
) -> DecisionJSON:
    decision = decision_store.get_decision(key)
    is_decided = decision_store.is_decided(key)

    if decision is None:
        return {
            "qa_status": "no",
            "reason": "",
            "date": "",
            "is_decided": False,
        }

    return {
        "qa_status": decision.qa_status,
        "reason": decision.reason,
        "date": decision.updated_at,
        "is_decided": bool(is_decided),
    }


def _build_subject_bundle(
    query_records: List[loader.QueryRecord],
    decision_store: DecisionStore,
) -> Tuple[List[Dict[str, Any]], DecisionMap, int, int]:
    subject_data: List[Dict[str, Any]] = []
    decision_map: DecisionMap = {}

    total_pairs = 0
    done_pairs = 0

    for qr in query_records:
        query = qr.query
        query_payload = _scan_to_payload(query)
        candidates: List[Dict[str, Any]] = []

        for cand in qr.candidates:
            scan = cand.scanref
            pair_key = _pair_key_for_query_candidate(
                query=query,
                cand_scan=scan,
                decision_store=decision_store,
            )
            pair_key_str = _pair_key_to_str(pair_key)

            decision_json = _decision_to_json(
                decision_store=decision_store,
                key=pair_key,
            )

            total_pairs += 1
            if decision_json["is_decided"]:
                done_pairs += 1

            decision_map[pair_key_str] = decision_json

            candidates.append(
                {
                    **_scan_to_payload(scan),
                    "similarity": cand.similarity,
                    "pair_key": pair_key_str,
                }
            )

        subject_data.append(
            {
                "query": query_payload,
                "candidates": candidates,
            }
        )

    return subject_data, decision_map, done_pairs, total_pairs


def _build_workspace_subject_payload_from_records(
    dataset: str,
    subject_id: str,
    subjects_sorted: List[str],
    subject_pos: int,
    query_records: List[loader.QueryRecord],
    decision_store: DecisionStore,
) -> SubjectPayload:
    subject_data, decision_map, done_pairs, total_pairs = _build_subject_bundle(
        query_records=query_records,
        decision_store=decision_store,
    )

    prev_subject_id = subjects_sorted[subject_pos - 1] if subject_pos > 0 else None
    next_subject_id = (
        subjects_sorted[subject_pos + 1]
        if subject_pos < (len(subjects_sorted) - 1)
        else None
    )

    return {
        "dataset": dataset,
        "subject_id": subject_id,
        "done_pairs": done_pairs,
        "total_pairs": total_pairs,
        "query_count": len(subject_data),
        "is_done": (total_pairs > 0 and done_pairs == total_pairs),
        "prev_subject_id": prev_subject_id,
        "next_subject_id": next_subject_id,
        "subject_data": subject_data,
        "decision_map": decision_map,
    }


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
    dataset = _clean(dataset)
    subject_id = _clean(subject_id)

    if not dataset or dataset not in index:
        raise KeyError(f"unknown dataset: {dataset}")
    if not subject_id or subject_id not in index[dataset]:
        raise KeyError(f"unknown subject in dataset {dataset}: {subject_id}")

    query_records = index[dataset][subject_id]
    _, _, done_pairs, total_pairs = _build_subject_bundle(
        query_records=query_records,
        decision_store=decision_store,
    )
    return done_pairs, total_pairs


def build_dataset_subject_progress(
    index: IndexType,
    dataset: str,
    decision_store: DecisionStore,
) -> Tuple[int, int]:
    dataset = _clean(dataset)

    if not dataset or dataset not in index:
        raise KeyError(f"unknown dataset: {dataset}")

    subject_map = index[dataset]
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

    subjects_sorted = sorted(index[dataset].keys())
    subject_pos = subjects_sorted.index(subject_id)
    query_records = index[dataset][subject_id]

    return _build_workspace_subject_payload_from_records(
        dataset=dataset,
        subject_id=subject_id,
        subjects_sorted=subjects_sorted,
        subject_pos=subject_pos,
        query_records=query_records,
        decision_store=decision_store,
    )


def build_workspace_dataset_payload(
    index: IndexType,
    dataset: str,
    decision_store: DecisionStore,
    include_subject_data: bool = True,
) -> DatasetPayload:
    dataset = _clean(dataset)

    if not dataset or dataset not in index:
        raise KeyError(f"unknown dataset: {dataset}")

    subjects_sorted = sorted(index[dataset].keys())
    subject_rows: List[SubjectRow] = []
    review_by_subject: Dict[str, SubjectPayload] = {}
    dataset_decision_map: DecisionMap = {}

    done_subjects = 0
    total_subjects = len(subjects_sorted)

    for pos, subject_id in enumerate(subjects_sorted):
        query_records = index[dataset][subject_id]

        subject_payload = _build_workspace_subject_payload_from_records(
            dataset=dataset,
            subject_id=subject_id,
            subjects_sorted=subjects_sorted,
            subject_pos=pos,
            query_records=query_records,
            decision_store=decision_store,
        )

        row: SubjectRow = {
            "subject_id": subject_id,
            "done_pairs": subject_payload["done_pairs"],
            "total_pairs": subject_payload["total_pairs"],
            "query_count": subject_payload["query_count"],
            "is_done": subject_payload["is_done"],
            "prev_subject_id": subject_payload["prev_subject_id"],
            "next_subject_id": subject_payload["next_subject_id"],
        }
        subject_rows.append(row)

        if row["is_done"]:
            done_subjects += 1

        if include_subject_data:
            review_by_subject[subject_id] = subject_payload
            dataset_decision_map.update(subject_payload["decision_map"])

    payload: DatasetPayload = {
        "dataset": dataset,
        "done_subjects": done_subjects,
        "total_subjects": total_subjects,
        "subjects": subject_rows,
    }

    if include_subject_data:
        payload["review_by_subject"] = review_by_subject
        payload["decision_map"] = dataset_decision_map

    return payload
