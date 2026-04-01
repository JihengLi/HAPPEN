"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

from __future__ import annotations

import csv
import threading

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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


def _require_nonempty(name: str, value: Optional[str]) -> str:
    v = _norm(value)
    if not v:
        raise ValueError(f"Missing required non-empty field: {name}")
    return v


def _normalize_path_or_raise(name: str, value: Optional[str]) -> str:
    v = _require_nonempty(name, value)
    return v.replace("\\", "/")


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    a2 = _normalize_path_or_raise("pair_key.a", a)
    b2 = _normalize_path_or_raise("pair_key.b", b)
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
            self._writes_since_export = 0
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
        if not p:
            raise ValueError(
                "src_path is required and cannot be empty "
                f"(dataset={_norm(dataset)!r}, "
                f"subject_id={_norm(subject_id)!r}, "
                f"session_id={_norm(session_id)!r}, "
                f"scan_uid={_norm(scan_uid)!r})"
            )
        return p.replace("\\", "/")

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
        a_item_key = cls.make_item_key(
            dataset=a_dataset,
            subject_id=a_subject_id,
            session_id=a_session_id,
            scan_uid=a_scan_uid,
            src_path=a_src_path,
        )
        b_item_key = cls.make_item_key(
            dataset=b_dataset,
            subject_id=b_subject_id,
            session_id=b_session_id,
            scan_uid=b_scan_uid,
            src_path=b_src_path,
        )
        return _pair_key(a_item_key, b_item_key)

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
        a_store_path = _normalize_path_or_raise("a_src_path", a_src_path)
        b_store_path = _normalize_path_or_raise("b_src_path", b_src_path)

        if a_store_path != a_item_key:
            raise ValueError(
                f"a_item_key mismatch: normalized a_src_path={a_store_path!r}, "
                f"a_item_key={a_item_key!r}"
            )
        if b_store_path != b_item_key:
            raise ValueError(
                f"b_item_key mismatch: normalized b_src_path={b_store_path!r}, "
                f"b_item_key={b_item_key!r}"
            )

        if a_item_key <= b_item_key:
            return (
                _norm(a_dataset),
                _norm(a_subject_id),
                _norm(a_session_id),
                a_store_path,
                _norm(b_dataset),
                _norm(b_subject_id),
                _norm(b_session_id),
                b_store_path,
            )

        return (
            _norm(b_dataset),
            _norm(b_subject_id),
            _norm(b_session_id),
            b_store_path,
            _norm(a_dataset),
            _norm(a_subject_id),
            _norm(a_session_id),
            a_store_path,
        )

    @classmethod
    def _build_key_and_meta(
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
    ) -> Tuple[Tuple[str, str], Tuple[str, str, str, str, str, str, str, str]]:
        a_item_key = cls.make_item_key(
            dataset=a_dataset,
            subject_id=a_subject_id,
            session_id=a_session_id,
            scan_uid=a_scan_uid,
            src_path=a_src_path,
        )
        b_item_key = cls.make_item_key(
            dataset=b_dataset,
            subject_id=b_subject_id,
            session_id=b_session_id,
            scan_uid=b_scan_uid,
            src_path=b_src_path,
        )
        key = _pair_key(a_item_key, b_item_key)

        meta = cls._canonicalize_meta(
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
        return key, meta

    def _maybe_export_locked(self) -> None:
        if self._writes_since_export >= self.autosave_every:
            self._export_locked()
            self._writes_since_export = 0

    def _set_decision_locked(
        self,
        key: Tuple[str, str],
        meta: Tuple[str, str, str, str, str, str, str, str],
        qa_status: str,
        reason: str,
        timestamp_iso: str,
    ) -> None:
        self._cache[key] = Decision(
            qa_status=qa_status,
            reason=_norm(reason),
            updated_at=timestamp_iso,
        )
        self._meta[key] = meta
        self._writes_since_export += 1

    def has_key(self, key: Tuple[str, str]) -> bool:
        with self._lock:
            return key in self._cache

    def get_decision(self, key: Tuple[str, str]) -> Optional[Decision]:
        with self._lock:
            return self._cache.get(key)

    def get_decisions(
        self,
        keys: Iterable[Tuple[str, str]],
    ) -> Dict[Tuple[str, str], Optional[Decision]]:
        with self._lock:
            return {key: self._cache.get(key) for key in keys}

    def get_reason_or_empty(self, key: Tuple[str, str]) -> str:
        with self._lock:
            d = self._cache.get(key)
            return d.reason if d else ""

    def get_all_decisions(self) -> Dict[Tuple[str, str], Decision]:
        with self._lock:
            return dict(self._cache)

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
        timestamp_iso: Optional[str] = None,
    ) -> Tuple[str, str]:
        ts = timestamp_iso or _now_iso()

        key, meta = self._build_key_and_meta(
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
            if key not in self._cache:
                self._set_decision_locked(
                    key=key,
                    meta=meta,
                    qa_status="no",
                    reason="",
                    timestamp_iso=ts,
                )
                self._maybe_export_locked()
            elif key not in self._meta:
                raise KeyError(f"Missing metadata for existing decision key: {key!r}")

        return key

    def ensure_default_no_batch(
        self,
        rows: Iterable[Dict[str, Any]],
        timestamp_iso: Optional[str] = None,
    ) -> List[Tuple[str, str]]:
        ts = timestamp_iso or _now_iso()
        keys: List[Tuple[str, str]] = []

        with self._lock:
            for row in rows:
                key, meta = self._build_key_and_meta(
                    a_dataset=row["a_dataset"],
                    a_subject_id=row["a_subject_id"],
                    a_session_id=row["a_session_id"],
                    a_scan_uid=row["a_scan_uid"],
                    a_src_path=row["a_src_path"],
                    b_dataset=row["b_dataset"],
                    b_subject_id=row["b_subject_id"],
                    b_session_id=row["b_session_id"],
                    b_scan_uid=row["b_scan_uid"],
                    b_src_path=row["b_src_path"],
                )
                keys.append(key)

                if key not in self._cache:
                    self._set_decision_locked(
                        key=key,
                        meta=meta,
                        qa_status="no",
                        reason="",
                        timestamp_iso=ts,
                    )
                elif key not in self._meta:
                    raise KeyError(
                        f"Missing metadata for existing decision key: {key!r}"
                    )

            self._maybe_export_locked()

        return keys

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
        timestamp_iso: Optional[str] = None,
    ) -> Tuple[str, str]:
        qa = _norm(qa_status).lower()
        if qa not in {"yes", "no", "maybe"}:
            raise ValueError(f"Invalid qa_status={qa_status!r}, expected yes/no/maybe.")

        ts = timestamp_iso or _now_iso()

        key, meta = self._build_key_and_meta(
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
            self._set_decision_locked(
                key=key,
                meta=meta,
                qa_status=qa,
                reason=reason,
                timestamp_iso=ts,
            )
            self._maybe_export_locked()

        return key

    def upsert_decisions_batch(
        self,
        rows: Iterable[Dict[str, Any]],
        timestamp_iso: Optional[str] = None,
    ) -> List[Tuple[str, str]]:
        ts = timestamp_iso or _now_iso()
        keys: List[Tuple[str, str]] = []

        with self._lock:
            for row in rows:
                qa = _norm(row.get("qa_status")).lower()
                if qa not in {"yes", "no", "maybe"}:
                    raise ValueError(
                        f"Invalid qa_status={row.get('qa_status')!r}, expected yes/no/maybe."
                    )

                key, meta = self._build_key_and_meta(
                    a_dataset=row["a_dataset"],
                    a_subject_id=row["a_subject_id"],
                    a_session_id=row["a_session_id"],
                    a_scan_uid=row["a_scan_uid"],
                    a_src_path=row["a_src_path"],
                    b_dataset=row["b_dataset"],
                    b_subject_id=row["b_subject_id"],
                    b_session_id=row["b_session_id"],
                    b_scan_uid=row["b_scan_uid"],
                    b_src_path=row["b_src_path"],
                )
                keys.append(key)

                self._set_decision_locked(
                    key=key,
                    meta=meta,
                    qa_status=qa,
                    reason=_norm(row.get("reason", "")),
                    timestamp_iso=_norm(row.get("timestamp_iso")) or ts,
                )

            self._maybe_export_locked()

        return keys

    def export(self) -> None:
        with self._lock:
            self._export_locked()
            self._writes_since_export = 0

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
                    raise KeyError(
                        f"Missing metadata for decision key during export: {key!r}"
                    )

                ds1, sbj1, ses1, p1, ds2, sbj2, ses2, p2 = meta

                if not p1 or not p2:
                    raise ValueError(
                        f"Cannot export decision with empty path(s): key={key!r}, meta={meta!r}"
                    )

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

            for row_idx, row in enumerate(r, start=2):
                p1 = _normalize_path_or_raise(
                    f"path_1 at CSV row {row_idx}", row.get("path_1")
                )
                p2 = _normalize_path_or_raise(
                    f"path_2 at CSV row {row_idx}", row.get("path_2")
                )

                k = _pair_key(p1, p2)

                qa = _norm(row.get("QA_status")).lower()
                if qa not in {"yes", "no", "maybe"}:
                    raise ValueError(
                        f"Invalid QA_status at CSV row {row_idx}: {row.get('QA_status')!r}"
                    )

                reason = _norm(row.get("reason"))
                dt = _norm(row.get("date")) or _now_iso()

                ds1 = _norm(row.get("dataset_1"))
                sbj1 = _norm(row.get("subject_id_1"))
                ses1 = _norm(row.get("session_id_1"))
                ds2 = _norm(row.get("dataset_2"))
                sbj2 = _norm(row.get("subject_id_2"))
                ses2 = _norm(row.get("session_id_2"))

                self._cache[k] = Decision(
                    qa_status=qa,
                    reason=reason,
                    updated_at=dt,
                )
                self._meta[k] = (ds1, sbj1, ses1, p1, ds2, sbj2, ses2, p2)

    def num_decisions(self) -> int:
        with self._lock:
            return len(self._cache)

    def iter_keys(self) -> Iterable[Tuple[str, str]]:
        with self._lock:
            return list(self._cache.keys())

    def iter_items(self) -> Iterable[Tuple[Tuple[str, str], Decision]]:
        with self._lock:
            return list(self._cache.items())

    def prune_to_existing_items(self, valid_item_keys: Iterable[str]) -> int:
        valid = {_normalize_path_or_raise("valid_item_key", x) for x in valid_item_keys}

        with self._lock:
            to_delete = [
                key
                for key in self._cache.keys()
                if key[0] not in valid or key[1] not in valid
            ]
            for key in to_delete:
                self._cache.pop(key, None)
                self._meta.pop(key, None)

            if to_delete:
                self._writes_since_export += 1
                self._maybe_export_locked()

            return len(to_delete)
