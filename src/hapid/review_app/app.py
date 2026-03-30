"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypeAlias

from flask import (
    Flask,
    abort,
    jsonify,
    render_template,
    request,
    send_file,
    url_for,
)

from . import loader
from .store import DecisionStore


ScanPayload: TypeAlias = Dict[str, str]
PairPayload: TypeAlias = Dict[str, Any]
IndexType: TypeAlias = Dict[str, Dict[str, List[loader.QueryRecord]]]


def _clean(s: Optional[str]) -> str:
    return (s or "").strip()


def _pair_key(scan_uid_a: str, scan_uid_b: str) -> str:
    a = _clean(scan_uid_a)
    b = _clean(scan_uid_b)
    if not (a and b):
        abort(400, "Missing scan_uid_a/scan_uid_b")
    x, y = sorted([a, b])
    return f"{x}__{y}"


def _resolve_png_path(
    png_root: Path,
    dataset: str,
    subject_id: str,
    session_id: str,
    scan_uid: str,
) -> Path:
    ds = _clean(dataset)
    sbj = _clean(subject_id)
    ses = _clean(session_id)
    uid = _clean(scan_uid)

    if not (ds and sbj and uid):
        abort(400, "Missing dataset/subject_id/scan_uid")

    rel = Path(ds) / sbj / ses / f"{uid}.png" if ses else Path(ds) / sbj / f"{uid}.png"

    path = (png_root / rel).resolve()
    root = png_root.resolve()

    try:
        path.relative_to(root)
    except Exception:
        abort(403, "Path traversal blocked")

    return path


def _resolve_pair_asset_path(
    assets_root: Path,
    kind: str,
    scan_uid_a: str,
    scan_uid_b: str,
) -> Path:
    kind_clean = _clean(kind).lower()
    if kind_clean not in {"diff", "checkerboard"}:
        abort(400, "kind must be 'diff' or 'checkerboard'")

    filename = f"{_pair_key(scan_uid_a, scan_uid_b)}.png"
    path = (assets_root / kind_clean / filename).resolve()
    root = assets_root.resolve()

    try:
        path.relative_to(root)
    except Exception:
        abort(403, "Path traversal blocked")

    return path


def _pair_payload_ok(payload: PairPayload) -> bool:
    return (
        isinstance(payload, dict)
        and "a" in payload
        and "b" in payload
        and isinstance(payload["a"], dict)
        and isinstance(payload["b"], dict)
    )


def _extract_pair_fields(payload: PairPayload) -> Tuple[ScanPayload, ScanPayload]:
    a0 = payload.get("a", {})
    b0 = payload.get("b", {})

    a: ScanPayload = {
        "dataset": _clean(a0.get("dataset")),
        "subject_id": _clean(a0.get("subject_id")),
        "session_id": _clean(a0.get("session_id")),
        "scan_uid": _clean(a0.get("scan_uid")),
        "src_path": _clean(a0.get("src_path")),
    }
    b: ScanPayload = {
        "dataset": _clean(b0.get("dataset")),
        "subject_id": _clean(b0.get("subject_id")),
        "session_id": _clean(b0.get("session_id")),
        "scan_uid": _clean(b0.get("scan_uid")),
        "src_path": _clean(b0.get("src_path")),
    }

    if not (
        a["dataset"]
        and a["subject_id"]
        and a["scan_uid"]
        and a["src_path"]
        and b["dataset"]
        and b["subject_id"]
        and b["scan_uid"]
        and b["src_path"]
    ):
        abort(
            400,
            "Missing required fields in pair (dataset/subject_id/scan_uid/src_path)",
        )

    return a, b


def _decision_to_json(decision: Any) -> Optional[Dict[str, str]]:
    if decision is None:
        return None
    return {
        "qa_status": decision.qa_status,
        "reason": decision.reason,
        "date": decision.updated_at,
    }


def _subject_pair_progress(
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


def _dataset_subject_progress(
    index: IndexType,
    dataset: str,
    decision_store: DecisionStore,
) -> Tuple[int, int]:
    subject_map = index.get(dataset, {})
    total_subjects = len(subject_map)
    done_subjects = 0

    for subject_id in subject_map.keys():
        done_pairs, total_pairs = _subject_pair_progress(
            index=index,
            dataset=dataset,
            subject_id=subject_id,
            decision_store=decision_store,
        )
        if total_pairs > 0 and done_pairs == total_pairs:
            done_subjects += 1

    return done_subjects, total_subjects


def _build_subject_data(
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


def create_app(
    index: IndexType,
    png_root: Path,
    decision_store: DecisionStore,
    review_mode: str,
) -> Flask:
    review_mode = _clean(review_mode).lower()
    if review_mode not in {"lazy", "precompute"}:
        raise ValueError(f"invalid review_mode: {review_mode}")

    app = Flask(__name__)

    png_root = Path(png_root).resolve()
    assets_root = png_root.parent.resolve()

    app.config["PNG_ROOT"] = str(png_root)
    app.config["ASSETS_ROOT"] = str(assets_root)
    app.config["REVIEW_MODE"] = review_mode

    @app.get("/")
    def datasets_page():
        datasets = sorted(index.keys())
        rows = []

        for dataset in datasets:
            done_subjects, total_subjects = _dataset_subject_progress(
                index=index,
                dataset=dataset,
                decision_store=decision_store,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "done_subjects": done_subjects,
                    "total_subjects": total_subjects,
                    "is_done": (total_subjects > 0 and done_subjects == total_subjects),
                    "href": url_for("subjects_page", dataset=dataset),
                }
            )

        total_datasets = len(rows)
        done_datasets = sum(1 for row in rows if row["is_done"])

        return render_template(
            "datasets.html",
            rows=rows,
            done_datasets=done_datasets,
            total_datasets=total_datasets,
        )

    @app.get("/dataset/<dataset>")
    def subjects_page(dataset: str):
        dataset = _clean(dataset)
        if dataset not in index:
            abort(404)

        subjects = sorted(index[dataset].keys())
        rows = []

        for subject_id in subjects:
            done_pairs, total_pairs = _subject_pair_progress(
                index=index,
                dataset=dataset,
                subject_id=subject_id,
                decision_store=decision_store,
            )
            rows.append(
                {
                    "subject_id": subject_id,
                    "done_pairs": done_pairs,
                    "total_pairs": total_pairs,
                    "is_done": (total_pairs > 0 and done_pairs == total_pairs),
                    "href": url_for(
                        "review_page",
                        dataset=dataset,
                        subject=subject_id,
                    ),
                }
            )

        done_subjects, total_subjects = _dataset_subject_progress(
            index=index,
            dataset=dataset,
            decision_store=decision_store,
        )

        return render_template(
            "subjects.html",
            dataset=dataset,
            rows=rows,
            done_subjects=done_subjects,
            total_subjects=total_subjects,
        )

    @app.get("/review/<dataset>/<subject>")
    def review_page(dataset: str, subject: str):
        dataset = _clean(dataset)
        subject_id = _clean(subject)

        if dataset not in index or subject_id not in index[dataset]:
            abort(404)

        subjects = sorted(index[dataset].keys())
        try:
            pos = subjects.index(subject_id)
        except ValueError:
            abort(404)

        prev_subject = subjects[pos - 1] if pos > 0 else None
        next_subject = subjects[pos + 1] if pos < (len(subjects) - 1) else None

        prev_review_href = (
            url_for("review_page", dataset=dataset, subject=prev_subject)
            if prev_subject
            else None
        )
        next_review_href = (
            url_for("review_page", dataset=dataset, subject=next_subject)
            if next_subject
            else None
        )

        query_records = index[dataset][subject_id]
        subject_data = _build_subject_data(query_records)

        return render_template(
            "review.html",
            dataset=dataset,
            subject_id=subject_id,
            subject_data=subject_data,
            prev_review_href=prev_review_href,
            next_review_href=next_review_href,
            review_mode=app.config["REVIEW_MODE"],
        )

    @app.get("/png")
    def png():
        png_root = Path(app.config["PNG_ROOT"])

        dataset = request.args.get("dataset", "")
        subject_id = request.args.get("subject_id", "")
        session_id = request.args.get("session_id", "")
        scan_uid = request.args.get("scan_uid", "")

        path = _resolve_png_path(
            png_root=png_root,
            dataset=dataset,
            subject_id=subject_id,
            session_id=session_id,
            scan_uid=scan_uid,
        )
        if not path.exists():
            abort(404)

        return send_file(path, mimetype="image/png", conditional=True)

    @app.get("/pair_asset")
    def pair_asset():
        if app.config["REVIEW_MODE"] != "precompute":
            abort(404)
        assets_root = Path(app.config["ASSETS_ROOT"])

        kind = request.args.get("kind", "")
        scan_uid_a = request.args.get("scan_uid_a", "")
        scan_uid_b = request.args.get("scan_uid_b", "")

        path = _resolve_pair_asset_path(
            assets_root=assets_root,
            kind=kind,
            scan_uid_a=scan_uid_a,
            scan_uid_b=scan_uid_b,
        )
        if not path.exists():
            abort(404)

        return send_file(path, mimetype="image/png", conditional=True)

    @app.post("/api/view")
    def api_view():
        payload = request.get_json(force=True, silent=False)
        if not _pair_payload_ok(payload):
            abort(400, "Bad payload")

        a, b = _extract_pair_fields(payload)

        key = decision_store.ensure_default_no(
            a_dataset=a["dataset"],
            a_subject_id=a["subject_id"],
            a_session_id=a["session_id"],
            a_scan_uid=a["scan_uid"],
            a_src_path=a["src_path"],
            b_dataset=b["dataset"],
            b_subject_id=b["subject_id"],
            b_session_id=b["session_id"],
            b_scan_uid=b["scan_uid"],
            b_src_path=b["src_path"],
        )
        decision = decision_store.get_decision(key)

        return jsonify(
            {
                "ok": True,
                "decision": _decision_to_json(decision),
            }
        )

    @app.post("/api/decision")
    def api_decision():
        payload = request.get_json(force=True, silent=False)
        if not _pair_payload_ok(payload):
            abort(400, "Bad payload")

        a, b = _extract_pair_fields(payload)

        qa_status = _clean(payload.get("qa_status", "")).lower()
        reason = _clean(payload.get("reason", ""))

        key = decision_store.upsert_decision(
            a_dataset=a["dataset"],
            a_subject_id=a["subject_id"],
            a_session_id=a["session_id"],
            a_scan_uid=a["scan_uid"],
            a_src_path=a["src_path"],
            b_dataset=b["dataset"],
            b_subject_id=b["subject_id"],
            b_session_id=b["session_id"],
            b_scan_uid=b["scan_uid"],
            b_src_path=b["src_path"],
            qa_status=qa_status,
            reason=reason,
        )
        decision = decision_store.get_decision(key)

        return jsonify(
            {
                "ok": True,
                "decision": _decision_to_json(decision),
            }
        )

    @app.post("/api/get_decision")
    def api_get_decision():
        payload = request.get_json(force=True, silent=False)
        if not _pair_payload_ok(payload):
            abort(400, "Bad payload")

        a, b = _extract_pair_fields(payload)

        key = decision_store.make_pair_key(
            a_dataset=a["dataset"],
            a_subject_id=a["subject_id"],
            a_session_id=a["session_id"],
            a_scan_uid=a["scan_uid"],
            a_src_path=a["src_path"],
            b_dataset=b["dataset"],
            b_subject_id=b["subject_id"],
            b_session_id=b["session_id"],
            b_scan_uid=b["scan_uid"],
            b_src_path=b["src_path"],
        )
        decision = decision_store.get_decision(key)

        return jsonify(
            {
                "ok": True,
                "decision": _decision_to_json(decision),
            }
        )

    return app
