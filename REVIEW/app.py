from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import (
    Flask,
    abort,
    jsonify,
    render_template,
    request,
    send_file,
    url_for,
)

import loader
from store import DecisionStore


def _clean(s: Optional[str]) -> str:
    return (s or "").strip()


def _png_path(
    root_dir: Path,
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

    p = (root_dir / rel).resolve()
    root = root_dir.resolve()

    try:
        p.relative_to(root)
    except Exception:
        abort(403, "Path traversal blocked")

    return p


def _pair_payload_ok(d: Dict[str, Any]) -> bool:
    return (
        isinstance(d, dict)
        and "a" in d
        and "b" in d
        and isinstance(d["a"], dict)
        and isinstance(d["b"], dict)
    )


def _extract_pair_fields(
    payload: Dict[str, Any],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    a0 = payload.get("a", {})
    b0 = payload.get("b", {})
    a = {
        "dataset": _clean(a0.get("dataset")),
        "subject_id": _clean(a0.get("subject_id")),
        "session_id": _clean(a0.get("session_id")),
        "scan_uid": _clean(a0.get("scan_uid")),
        "src_path": _clean(a0.get("src_path")),
    }
    b = {
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
        and b["dataset"]
        and b["subject_id"]
        and b["scan_uid"]
    ):
        abort(400, "Missing required fields in pair (dataset/subject_id/scan_uid)")
    return a, b


def _subject_pairs(
    index: Dict[str, Dict[str, List[loader.QueryRecord]]],
    ds: str,
    sbj: str,
    store: DecisionStore,
) -> Tuple[int, int]:
    total = 0
    done = 0
    qrs = index.get(ds, {}).get(sbj, [])
    for qr in qrs:
        for cand in qr.candidates:
            key = store.make_pair_key(
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
            if store.is_decided(key):
                done += 1
    return done, total


def _dataset_subject_done_counts(
    index: Dict[str, Dict[str, List[loader.QueryRecord]]],
    ds: str,
    store: DecisionStore,
) -> Tuple[int, int]:
    subj_map = index.get(ds, {})
    total_subjects = len(subj_map)
    done_subjects = 0
    for sbj in subj_map.keys():
        done_pairs, total_pairs = _subject_pairs(index, ds, sbj, store)
        if total_pairs > 0 and done_pairs == total_pairs:
            done_subjects += 1
    return done_subjects, total_subjects


def create_app(
    index: Dict[str, Dict[str, List[loader.QueryRecord]]],
    root_dir: Path,
    decision_store: DecisionStore,
) -> Flask:
    app = Flask(__name__)
    app.config["ROOT_DIR"] = str(Path(root_dir).resolve())

    @app.get("/")
    def datasets_page():
        datasets = sorted(index.keys())
        rows = []
        for ds in datasets:
            done_s, total_s = _dataset_subject_done_counts(index, ds, decision_store)
            rows.append(
                {
                    "dataset": ds,
                    "done_subjects": done_s,
                    "total_subjects": total_s,
                    "is_done": (total_s > 0 and done_s == total_s),
                    "href": url_for("subjects_page", dataset=ds),
                }
            )

        total_datasets = len(rows)
        done_datasets = sum(1 for r in rows if r["is_done"])

        return render_template(
            "datasets.html",
            rows=rows,
            done_datasets=done_datasets,
            total_datasets=total_datasets,
        )

    @app.get("/dataset/<dataset>")
    def subjects_page(dataset: str):
        ds = _clean(dataset)
        if ds not in index:
            abort(404)

        subjects = sorted(index[ds].keys())
        rows = []
        for sbj in subjects:
            done_pairs, total_pairs = _subject_pairs(index, ds, sbj, decision_store)
            rows.append(
                {
                    "subject_id": sbj,
                    "done_pairs": done_pairs,
                    "total_pairs": total_pairs,
                    "is_done": (total_pairs > 0 and done_pairs == total_pairs),
                    "href": url_for("review_page", dataset=ds, subject=sbj),
                }
            )

        done_s, total_s = _dataset_subject_done_counts(index, ds, decision_store)
        return render_template(
            "subjects.html",
            dataset=ds,
            rows=rows,
            done_subjects=done_s,
            total_subjects=total_s,
        )

    @app.get("/review/<dataset>/<subject>")
    def review_page(dataset: str, subject: str):
        ds = _clean(dataset)
        sbj = _clean(subject)
        if ds not in index or sbj not in index[ds]:
            abort(404)

        subjects = sorted(index[ds].keys())
        try:
            pos = subjects.index(sbj)
        except ValueError:
            abort(404)

        prev_subject = subjects[pos - 1] if pos > 0 else None
        next_subject = subjects[pos + 1] if pos < (len(subjects) - 1) else None

        prev_review_href = (
            url_for("review_page", dataset=ds, subject=prev_subject)
            if prev_subject
            else None
        )
        next_review_href = (
            url_for("review_page", dataset=ds, subject=next_subject)
            if next_subject
            else None
        )

        qrs = index[ds][sbj]

        data = []
        for qr in qrs:
            q = qr.query
            cands = []
            for cand in qr.candidates:
                c = cand.scanref
                cands.append(
                    {
                        "dataset": c.dataset,
                        "subject_id": c.subject_id,
                        "session_id": c.session_id,
                        "scan_uid": c.scan_uid,
                        "src_path": c.src_path,
                        "similarity": cand.similarity,
                    }
                )
            data.append(
                {
                    "query": {
                        "dataset": q.dataset,
                        "subject_id": q.subject_id,
                        "session_id": q.session_id,
                        "scan_uid": q.scan_uid,
                        "src_path": q.src_path,
                    },
                    "candidates": cands,
                }
            )

        return render_template(
            "review.html",
            dataset=ds,
            subject_id=sbj,
            subject_data=data,
            prev_review_href=prev_review_href,
            next_review_href=next_review_href,
        )

    @app.get("/png")
    def png():
        root = Path(app.config["ROOT_DIR"])

        ds = request.args.get("dataset", "")
        sbj = request.args.get("subject_id", "")
        ses = request.args.get("session_id", "")
        uid = request.args.get("scan_uid", "")

        p = _png_path(root, dataset=ds, subject_id=sbj, session_id=ses, scan_uid=uid)
        if not p.exists():
            abort(404)

        return send_file(p, mimetype="image/png", conditional=True)

    @app.post("/api/view")
    def api_view():
        payload = request.get_json(force=True, silent=False)
        if not _pair_payload_ok(payload):
            abort(400, "Bad payload")
        a, b = _extract_pair_fields(payload)
        reason = _clean(payload.get("reason", ""))

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
            reason=reason,
        )
        dec = decision_store.get_decision(key)
        return jsonify(
            {
                "ok": True,
                "decision": (
                    None
                    if dec is None
                    else {
                        "qa_status": dec.qa_status,
                        "reason": dec.reason,
                        "date": dec.updated_at,
                    }
                ),
            }
        )

    @app.post("/api/decision")
    def api_decision():
        payload = request.get_json(force=True, silent=False)
        if not _pair_payload_ok(payload):
            abort(400, "Bad payload")
        a, b = _extract_pair_fields(payload)

        status = _clean(payload.get("qa_status", "")).lower()
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
            qa_status=status,
            reason=reason,
            keep_reason_if_blank=True,
        )
        dec = decision_store.get_decision(key)
        return jsonify(
            {
                "ok": True,
                "decision": (
                    None
                    if dec is None
                    else {
                        "qa_status": dec.qa_status,
                        "reason": dec.reason,
                        "date": dec.updated_at,
                    }
                ),
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
        dec = decision_store.get_decision(key)
        if dec is None:
            return jsonify({"ok": True, "decision": None})
        return jsonify(
            {
                "ok": True,
                "decision": {
                    "qa_status": dec.qa_status,
                    "reason": dec.reason,
                    "date": dec.updated_at,
                },
            }
        )

    return app
