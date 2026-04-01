"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from io import BytesIO
from PIL import Image
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


def _clean(s: Optional[str]) -> str:
    return (s or "").strip()


def _parse_boolish(s: Optional[str], default: bool = True) -> bool:
    if s is None:
        return default
    t = _clean(s).lower()
    if t in {"1", "true", "yes", "y", "on"}:
        return True
    if t in {"0", "false", "no", "n", "off"}:
        return False
    return default


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


def _extract_pair_rows_from_batch_payload(
    payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        abort(400, "Bad payload")

    updates = payload.get("updates")
    if not isinstance(updates, list):
        abort(400, "Payload must contain list field 'updates'")

    rows: List[Dict[str, Any]] = []

    for idx, item in enumerate(updates):
        if not isinstance(item, dict):
            abort(400, f"Bad update at updates[{idx}]")

        a0 = item.get("a")
        b0 = item.get("b")
        if not isinstance(a0, dict) or not isinstance(b0, dict):
            abort(400, f"Bad pair payload at updates[{idx}]")

        a_dataset = _clean(a0.get("dataset"))
        a_subject_id = _clean(a0.get("subject_id"))
        a_session_id = _clean(a0.get("session_id"))
        a_scan_uid = _clean(a0.get("scan_uid"))
        a_src_path = _clean(a0.get("src_path"))

        b_dataset = _clean(b0.get("dataset"))
        b_subject_id = _clean(b0.get("subject_id"))
        b_session_id = _clean(b0.get("session_id"))
        b_scan_uid = _clean(b0.get("scan_uid"))
        b_src_path = _clean(b0.get("src_path"))

        if not (
            a_dataset
            and a_subject_id
            and a_scan_uid
            and a_src_path
            and b_dataset
            and b_subject_id
            and b_scan_uid
            and b_src_path
        ):
            abort(
                400,
                f"Missing required fields in updates[{idx}] "
                "(dataset/subject_id/scan_uid/src_path)",
            )

        qa_status = _clean(item.get("qa_status", "")).lower()
        reason = _clean(item.get("reason", ""))

        rows.append(
            {
                "a_dataset": a_dataset,
                "a_subject_id": a_subject_id,
                "a_session_id": a_session_id,
                "a_scan_uid": a_scan_uid,
                "a_src_path": a_src_path,
                "b_dataset": b_dataset,
                "b_subject_id": b_subject_id,
                "b_session_id": b_session_id,
                "b_scan_uid": b_scan_uid,
                "b_src_path": b_src_path,
                "qa_status": qa_status,
                "reason": reason,
            }
        )

    return rows


def _decision_map_from_keys(
    decision_store: DecisionStore,
    keys: List[Tuple[str, str]],
) -> Dict[str, Dict[str, Any]]:
    decisions = decision_store.get_decisions(keys)
    out: Dict[str, Dict[str, Any]] = {}

    for key in keys:
        pair_key_str = f"{key[0]}__{key[1]}"
        decision = decisions.get(key)
        is_decided = decision_store.is_decided(key)

        if decision is None:
            out[pair_key_str] = {
                "qa_status": "no",
                "reason": "",
                "date": "",
                "is_decided": False,
            }
        else:
            out[pair_key_str] = {
                "qa_status": decision.qa_status,
                "reason": decision.reason,
                "date": decision.updated_at,
                "is_decided": bool(is_decided),
            }

    return out


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


def _render_workspace(
    app: Flask,
    dataset: str,
    initial_view: str,
    subject_id: Optional[str],
):
    bootstrap = _build_workspace_bootstrap(
        dataset=dataset,
        initial_view=initial_view,
        subject_id=subject_id,
        review_mode=app.config["REVIEW_MODE"],
    )
    return render_template(
        "workspace.html",
        bootstrap=bootstrap,
        datasets_index_url=url_for("datasets_page"),
    )

def _send_resize_png(
    path: Path,
    max_width: int,
):
    with Image.open(path) as img:
        img = img.convert("RGBA")

        w, h = img.size
        if w > max_width:
            new_h = int(h * (max_width / w))
            img = img.resize((max_width, new_h), Image.Resampling.LANCZOS)

        buf = BytesIO()
        img.save(buf, format="PNG", optimize=True)
        buf.seek(0)

    return send_file(buf, mimetype="image/png", conditional=False)

def create_app(
    index: IndexType,
    png_root: Path,
    decision_store: DecisionStore,
    review_mode: str,
    preview_max_width: int,
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
    app.config["PREVIEW_MAX_WIDTH"] = int(preview_max_width)

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

        return _render_workspace(
            app=app,
            dataset=dataset,
            initial_view="subjects",
            subject_id=None,
        )

    @app.get("/review/<dataset>/<subject>")
    def review_page(dataset: str, subject: str):
        dataset = _clean(dataset)
        subject_id = _clean(subject)

        if dataset not in index or subject_id not in index[dataset]:
            abort(404)

        return _render_workspace(
            app=app,
            dataset=dataset,
            initial_view="review",
            subject_id=subject_id,
        )

    @app.get("/api/workspace_dataset")
    def api_workspace_dataset():
        dataset = _clean(request.args.get("dataset", ""))
        if not dataset:
            abort(400, "Missing dataset")
        if dataset not in index:
            abort(404)

        include_subject_data = _parse_boolish(
            request.args.get("include_subject_data"),
            default=True,
        )

        payload = build_workspace_dataset_payload(
            index=index,
            dataset=dataset,
            decision_store=decision_store,
            include_subject_data=include_subject_data,
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

        return _send_resize_png(path, max_width=app.config["PREVIEW_MAX_WIDTH"])

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

        return _send_resize_png(path, max_width=app.config["PREVIEW_MAX_WIDTH"])

    @app.post("/api/flush_decisions")
    def api_flush_decisions():
        payload = request.get_json(force=True, silent=False)
        rows = _extract_pair_rows_from_batch_payload(payload)

        keys = decision_store.upsert_decisions_batch(rows)
        decision_map = _decision_map_from_keys(
            decision_store=decision_store,
            keys=keys,
        )

        return jsonify(
            {
                "ok": True,
                "decision_map": decision_map,
                "num_updates": len(keys),
            }
        )

    return app
