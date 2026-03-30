"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TypeAlias

from flask import (
    Flask,
    abort,
    jsonify,
    render_template,
    request,
    send_file,
    url_for,
)

from .store import DecisionStore
from .workspace import (
    IndexType,
    build_dataset_subject_progress,
    build_workspace_dataset_payload,
)


ScanPayload: TypeAlias = Dict[str, str]
PairPayload: TypeAlias = Dict[str, Any]


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


def _build_workspace_bootstrap(
    dataset: str,
    initial_view: str,
    subject_id: Optional[str],
    review_mode: str,
) -> Dict[str, Any]:
    dataset = _clean(dataset)
    initial_view = _clean(initial_view).lower()
    subject_id = _clean(subject_id)
    review_mode = _clean(review_mode).lower()

    if initial_view not in {"subjects", "review"}:
        raise ValueError(f"invalid initial_view: {initial_view}")

    if review_mode not in {"lazy", "precompute"}:
        raise ValueError(f"invalid review_mode: {review_mode}")

    return {
        "dataset": dataset,
        "initial_view": initial_view,
        "initial_subject_id": subject_id,
        "review_mode": review_mode,
    }


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
            done_subjects, total_subjects = build_dataset_subject_progress(
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

        bootstrap = _build_workspace_bootstrap(
            dataset=dataset,
            initial_view="subjects",
            subject_id=None,
            review_mode=app.config["REVIEW_MODE"],
        )

        return render_template(
            "workspace.html",
            bootstrap=bootstrap,
            datasets_index_url=url_for("datasets_page"),
        )

    @app.get("/review/<dataset>/<subject>")
    def review_page(dataset: str, subject: str):
        dataset = _clean(dataset)
        subject_id = _clean(subject)

        if dataset not in index or subject_id not in index[dataset]:
            abort(404)

        bootstrap = _build_workspace_bootstrap(
            dataset=dataset,
            initial_view="review",
            subject_id=subject_id,
            review_mode=app.config["REVIEW_MODE"],
        )

        return render_template(
            "workspace.html",
            bootstrap=bootstrap,
            datasets_index_url=url_for("datasets_page"),
        )

    @app.get("/api/workspace_dataset")
    def api_workspace_dataset():
        dataset = _clean(request.args.get("dataset", ""))
        if not dataset:
            abort(400, "Missing dataset")
        if dataset not in index:
            abort(404)

        payload = build_workspace_dataset_payload(
            index=index,
            dataset=dataset,
            decision_store=decision_store,
            include_subject_data=True,
        )
        payload["ok"] = True
        payload["review_mode"] = app.config["REVIEW_MODE"]

        return jsonify(payload)

    @app.get("/png")
    def png():
        png_root_local = Path(app.config["PNG_ROOT"])

        dataset = request.args.get("dataset", "")
        subject_id = request.args.get("subject_id", "")
        session_id = request.args.get("session_id", "")
        scan_uid = request.args.get("scan_uid", "")

        path = _resolve_png_path(
            png_root=png_root_local,
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

        assets_root_local = Path(app.config["ASSETS_ROOT"])

        kind = request.args.get("kind", "")
        scan_uid_a = request.args.get("scan_uid_a", "")
        scan_uid_b = request.args.get("scan_uid_b", "")

        path = _resolve_pair_asset_path(
            assets_root=assets_root_local,
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
