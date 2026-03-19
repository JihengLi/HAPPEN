from __future__ import annotations

import csv
import threading

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from zoneinfo import ZoneInfo

TZ = ZoneInfo("America/Chicago")


def _now_iso() -> str:
    dt = datetime.now(TZ)
    return dt.isoformat(timespec="seconds")


def _norm(s: Optional[str]) -> str:
    if s is None:
        return ""
    s2 = str(s).strip()
    if s2.lower() in {"nan", "none", "null", "na", "n/a"}:
        return ""
    return s2


def _identity_fallback(
    dataset: str, subject_id: str, session_id: str, scan_uid: str
) -> str:
    ds = _norm(dataset)
    sbj = _norm(subject_id)
    ses = _norm(session_id)
    uid = _norm(scan_uid)
    ses_part = ses if ses else "(no-session)"
    return f"ID::{ds}::{sbj}::{ses_part}::{uid}"


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    a2 = _norm(a).replace("\\", "/")
    b2 = _norm(b).replace("\\", "/")
    return (a2, b2) if a2 <= b2 else (b2, a2)


@dataclass(frozen=True)
class Decision:
    qa_status: str
    reason: str
    updated_at: str


class DecisionStore:
    HEADER = [
        "dataset_1",
        "subject_id_1",
        "session_id_1",
        "path_1",
        "dataset_2",
        "subject_id_2",
        "session_id_2",
        "path_2",
        "QA_status",
        "reason",
        "date",
    ]

    def __init__(self, out_csv: Path, autosave_every: int = 1) -> None:
        self.out_csv = Path(out_csv)
        self.autosave_every = max(1, int(autosave_every))

        self._lock = threading.Lock()
        self._cache: Dict[Tuple[str, str], Decision] = {}
        self._meta: Dict[
            Tuple[str, str], Tuple[str, str, str, str, str, str, str, str]
        ] = {}
        self._writes_since_export = 0

    def init(self) -> None:
        with self._lock:
            self._cache.clear()
            self._meta.clear()
            if self.out_csv.exists():
                self._load_from_csv_locked(self.out_csv)

    @staticmethod
    def make_item_key(
        dataset: str,
        subject_id: str,
        session_id: str,
        scan_uid: str,
        src_path: str,
    ) -> str:
        p = _norm(src_path)
        if p:
            return p.replace("\\", "/")
        return _identity_fallback(dataset, subject_id, session_id, scan_uid)

    @classmethod
    def make_pair_key(
        cls,
        a_dataset: str,
        a_subject_id: str,
        a_session_id: str,
        a_scan_uid: str,
        a_src_path: str,
        b_dataset: str,
        b_subject_id: str,
        b_session_id: str,
        b_scan_uid: str,
        b_src_path: str,
    ) -> Tuple[str, str]:
        ka = cls.make_item_key(
            dataset=a_dataset,
            subject_id=a_subject_id,
            session_id=a_session_id,
            scan_uid=a_scan_uid,
            src_path=a_src_path,
        )
        kb = cls.make_item_key(
            dataset=b_dataset,
            subject_id=b_subject_id,
            session_id=b_session_id,
            scan_uid=b_scan_uid,
            src_path=b_src_path,
        )
        return _pair_key(ka, kb)

    @staticmethod
    def _canonicalize_meta(
        a_dataset: str,
        a_subject_id: str,
        a_session_id: str,
        a_src_path: str,
        a_item_key: str,
        b_dataset: str,
        b_subject_id: str,
        b_session_id: str,
        b_src_path: str,
        b_item_key: str,
    ) -> Tuple[str, str, str, str, str, str, str, str]:
        a_key = a_item_key
        b_key = b_item_key
        if a_key <= b_key:
            return (
                _norm(a_dataset),
                _norm(a_subject_id),
                _norm(a_session_id),
                _norm(a_src_path),
                _norm(b_dataset),
                _norm(b_subject_id),
                _norm(b_session_id),
                _norm(b_src_path),
            )
        else:
            return (
                _norm(b_dataset),
                _norm(b_subject_id),
                _norm(b_session_id),
                _norm(b_src_path),
                _norm(a_dataset),
                _norm(a_subject_id),
                _norm(a_session_id),
                _norm(a_src_path),
            )

    def is_decided(self, key: Tuple[str, str]) -> bool:
        return key in self._cache

    def get_decision(self, key: Tuple[str, str]) -> Optional[Decision]:
        return self._cache.get(key)

    def get_reason_or_empty(self, key: Tuple[str, str]) -> str:
        d = self._cache.get(key)
        return d.reason if d else ""

    def ensure_default_no(
        self,
        a_dataset: str,
        a_subject_id: str,
        a_session_id: str,
        a_scan_uid: str,
        a_src_path: str,
        b_dataset: str,
        b_subject_id: str,
        b_session_id: str,
        b_scan_uid: str,
        b_src_path: str,
        reason: str = "",
    ) -> Tuple[str, str]:
        key = self.make_pair_key(
            a_dataset=a_dataset,
            a_subject_id=a_subject_id,
            a_session_id=a_session_id,
            a_scan_uid=a_scan_uid,
            a_src_path=a_src_path,
            b_dataset=b_dataset,
            b_subject_id=b_subject_id,
            b_session_id=b_session_id,
            b_scan_uid=b_scan_uid,
            b_src_path=b_src_path,
        )
        with self._lock:
            if key in self._cache:
                return key
        return self.upsert_decision(
            a_dataset=a_dataset,
            a_subject_id=a_subject_id,
            a_session_id=a_session_id,
            a_scan_uid=a_scan_uid,
            a_src_path=a_src_path,
            b_dataset=b_dataset,
            b_subject_id=b_subject_id,
            b_session_id=b_session_id,
            b_scan_uid=b_scan_uid,
            b_src_path=b_src_path,
            qa_status="no",
            reason=reason,
            keep_reason_if_blank=True,
        )

    def upsert_decision(
        self,
        a_dataset: str,
        a_subject_id: str,
        a_session_id: str,
        a_scan_uid: str,
        a_src_path: str,
        b_dataset: str,
        b_subject_id: str,
        b_session_id: str,
        b_scan_uid: str,
        b_src_path: str,
        qa_status: str,
        reason: str = "",
        keep_reason_if_blank: bool = True,
        timestamp_iso: Optional[str] = None,
    ) -> Tuple[str, str]:
        qa = _norm(qa_status).lower()
        if qa not in {"yes", "no", "maybe"}:
            raise ValueError(f"Invalid qa_status={qa_status!r}, expected yes/no/maybe.")

        ts = timestamp_iso or _now_iso()

        a_item_key = self.make_item_key(
            dataset=a_dataset,
            subject_id=a_subject_id,
            session_id=a_session_id,
            scan_uid=a_scan_uid,
            src_path=a_src_path,
        )
        b_item_key = self.make_item_key(
            dataset=b_dataset,
            subject_id=b_subject_id,
            session_id=b_session_id,
            scan_uid=b_scan_uid,
            src_path=b_src_path,
        )
        key = _pair_key(a_item_key, b_item_key)

        meta = self._canonicalize_meta(
            a_dataset=a_dataset,
            a_subject_id=a_subject_id,
            a_session_id=a_session_id,
            a_src_path=a_src_path,
            a_item_key=a_item_key,
            b_dataset=b_dataset,
            b_subject_id=b_subject_id,
            b_session_id=b_session_id,
            b_src_path=b_src_path,
            b_item_key=b_item_key,
        )

        incoming_reason = _norm(reason)

        with self._lock:
            if keep_reason_if_blank and not incoming_reason:
                old = self._cache.get(key)
                if old is not None and old.reason:
                    incoming_reason = old.reason

            self._cache[key] = Decision(
                qa_status=qa, reason=incoming_reason, updated_at=ts
            )
            self._meta[key] = meta

            self._writes_since_export += 1
            if self._writes_since_export >= self.autosave_every:
                self._export_locked()
                self._writes_since_export = 0

        return key

    def export(self) -> None:
        with self._lock:
            self._export_locked()

    def _export_locked(self) -> None:
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.out_csv.with_suffix(self.out_csv.suffix + ".tmp")

        items = list(self._cache.items())
        items.sort(key=lambda kv: (kv[0][0], kv[0][1]))

        with tmp.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            w.writerow(self.HEADER)
            for key, dec in items:
                meta = self._meta.get(key)
                if meta is None:
                    meta = ("", "", "", key[0], "", "", "", key[1])
                ds1, sbj1, ses1, p1, ds2, sbj2, ses2, p2 = meta
                w.writerow(
                    [
                        ds1,
                        sbj1,
                        ses1,
                        p1,
                        ds2,
                        sbj2,
                        ses2,
                        p2,
                        dec.qa_status,
                        dec.reason,
                        dec.updated_at,
                    ]
                )

        tmp.replace(self.out_csv)

    def _load_from_csv_locked(self, path: Path) -> None:
        with path.open("r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            if not r.fieldnames:
                return
            for row in r:
                p1 = _norm(row.get("path_1"))
                p2 = _norm(row.get("path_2"))

                if not p1 and not p2:
                    continue

                k = _pair_key(p1, p2)

                qa = _norm(row.get("QA_status")).lower()
                if qa not in {"yes", "no", "maybe"}:
                    continue
                reason = _norm(row.get("reason"))
                dt = _norm(row.get("date")) or _now_iso()

                ds1 = _norm(row.get("dataset_1"))
                sbj1 = _norm(row.get("subject_id_1"))
                ses1 = _norm(row.get("session_id_1"))
                ds2 = _norm(row.get("dataset_2"))
                sbj2 = _norm(row.get("subject_id_2"))
                ses2 = _norm(row.get("session_id_2"))

                self._cache[k] = Decision(qa_status=qa, reason=reason, updated_at=dt)
                self._meta[k] = (ds1, sbj1, ses1, p1, ds2, sbj2, ses2, p2)

    def num_decisions(self) -> int:
        return len(self._cache)

    def iter_keys(self) -> Iterable[Tuple[str, str]]:
        return self._cache.keys()

    def prune_to_existing_items(self, valid_item_keys: Iterable[str]) -> int:
        valid = {_norm(x).replace("\\", "/") for x in valid_item_keys if _norm(x)}
        with self._lock:
            to_delete = [
                key
                for key in self._cache.keys()
                if key[0] not in valid or key[1] not in valid
            ]
            for key in to_delete:
                self._cache.pop(key, None)
                self._meta.pop(key, None)
            return len(to_delete)
