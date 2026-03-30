"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

from __future__ import annotations

import logging
import tomllib
import argparse

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

from .review_app_new import loader
from .review_app_new.app import create_app
from .review_app_new.store import DecisionStore


@dataclass
class ReviewConfig:
    config_path: Path
    out_dir: Path
    review_mode: str
    png_root: Path
    candidates_csv: Path
    decisions_csv: Path
    autosave_every: int
    host: str
    port: int
    debug: bool


def _read_toml(config_path: Union[str, Path]) -> Dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")

    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"invalid TOML config: {config_path}")

    return cfg


def _require_table(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    if name not in cfg:
        raise ValueError(f"config missing required table: [{name}]")
    table = cfg[name]
    if not isinstance(table, dict):
        raise ValueError(f"config table [{name}] must be a table/object")
    return table


def _build_review_config(config_path: Union[str, Path]) -> ReviewConfig:
    config_path = Path(config_path).expanduser().resolve()
    cfg = _read_toml(config_path)

    run_cfg = _require_table(cfg, "run")
    near_cfg = _require_table(cfg, "near")
    review_cfg = _require_table(cfg, "review")

    out = run_cfg.get("out", None)
    if out is None or not str(out).strip():
        raise ValueError("[run].out is required")

    out_dir = Path(str(out)).expanduser().resolve()

    review_mode = str(near_cfg.get("review_mode", "off")).strip().lower()
    if review_mode not in {"off", "lazy", "precompute"}:
        raise ValueError("[near].review_mode must be one of: off, lazy, precompute")

    review_root = out_dir / "near" / "review"
    png_root = review_root / "assets" / "png"
    candidates_csv = review_root / "review_candidates.csv"
    decisions_csv = review_root / "review_decisions.csv"

    autosave_every = max(1, int(review_cfg.get("autosave_every", 1)))
    host = str(review_cfg.get("host", "127.0.0.1")).strip() or "127.0.0.1"
    port = int(review_cfg.get("port", 5291))
    debug = bool(review_cfg.get("debug", False))

    return ReviewConfig(
        config_path=config_path,
        out_dir=out_dir,
        review_mode=review_mode,
        png_root=png_root,
        candidates_csv=candidates_csv,
        decisions_csv=decisions_csv,
        autosave_every=autosave_every,
        host=host,
        port=port,
        debug=debug,
    )


def run_review_app(cfg: ReviewConfig) -> int:
    if cfg.review_mode == "off":
        logging.error(
            "[REVIEW] review_mode=off. Review assets were not enabled in the pipeline config."
        )
        return 2

    if not cfg.png_root.exists():
        logging.error("[REVIEW] png_root does not exist: %s", cfg.png_root)
        return 2

    if not cfg.candidates_csv.exists():
        logging.error(
            "[REVIEW] candidates_csv does not exist: %s",
            cfg.candidates_csv,
        )
        return 2

    cfg.decisions_csv.parent.mkdir(parents=True, exist_ok=True)

    logging.info("[REVIEW] config -> %s", cfg.config_path)
    logging.info("[REVIEW] review_mode -> %s", cfg.review_mode)
    logging.info("[REVIEW] png_root -> %s", cfg.png_root)
    logging.info("[REVIEW] candidates_csv -> %s", cfg.candidates_csv)
    logging.info("[REVIEW] decisions_csv -> %s", cfg.decisions_csv)

    logging.info("[REVIEW] Loading candidates CSV: %s", cfg.candidates_csv)
    df = loader.load_in_csv(cfg.candidates_csv)
    slots = loader.detect_candidate_slots(df)
    queries = loader.parse_queries(df, slots=slots)

    if not queries:
        logging.error("[REVIEW] No queries parsed (check CSV headers/content).")
        return 3

    queries = loader.dedup_undirected_pairs(queries)
    index = loader.build_index(queries)
    logging.info("[REVIEW] Parsed datasets=%d", len(index))

    store = DecisionStore(
        out_csv=cfg.decisions_csv,
        autosave_every=cfg.autosave_every,
    )
    store.init()

    loaded = store.num_decisions() if hasattr(store, "num_decisions") else 0
    logging.info("[REVIEW] Loaded existing decisions=%d", loaded)

    valid_item_keys = loader.collect_valid_item_keys(queries)
    pruned = store.prune_to_existing_items(valid_item_keys)

    logging.info(
        "[REVIEW] Loaded existing decisions=%d, pruned=%d, remaining=%d",
        loaded,
        pruned,
        store.num_decisions(),
    )
    store.export()

    app = create_app(
        index=index,
        png_root=cfg.png_root,
        decision_store=store,
        review_mode=cfg.review_mode,
    )

    logging.info("[REVIEW] Starting server: http://%s:%d", cfg.host, cfg.port)
    try:
        app.run(
            host=cfg.host,
            port=cfg.port,
            debug=False,
            use_reloader=False,
            threaded=True,
        )
    except KeyboardInterrupt:
        logging.info("[REVIEW] Interrupted. Exporting decisions before exit.")
        try:
            store.export()
        except Exception as e:
            logging.warning("[REVIEW] Failed to export decisions on exit: %s", e)

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Run review app.")
    ap.add_argument(
        "config",
        type=Path,
        help="Path to review config TOML file",
    )
    args = ap.parse_args()

    try:
        cfg = _build_review_config(args.config)
    except Exception as e:
        print(f"[REVIEW][ERROR] {e}")
        return 2

    logging.basicConfig(
        level=logging.DEBUG if cfg.debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        return run_review_app(cfg)
    except Exception:
        logging.exception("[REVIEW] Unhandled exception.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
