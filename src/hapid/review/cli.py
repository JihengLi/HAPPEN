from __future__ import annotations

import logging
import tomllib

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

from . import loader
from .app import create_app
from .store import DecisionStore


@dataclass
class ReviewConfig:
    config_path: Path
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


def _require_path_str(d: Dict[str, Any], key: str, table_name: str) -> Path:
    if key not in d:
        raise ValueError(f"[{table_name}].{key} is required")
    v = str(d[key]).strip()
    if not v:
        raise ValueError(f"[{table_name}].{key} is required")
    return Path(v).expanduser().resolve()


def _build_review_config(config_path: Union[str, Path]) -> ReviewConfig:
    config_path = Path(config_path).expanduser().resolve()
    cfg = _read_toml(config_path)
    review_cfg = _require_table(cfg, "review")

    png_root = _require_path_str(review_cfg, "png_root", "review")
    candidates_csv = _require_path_str(review_cfg, "candidates_csv", "review")
    decisions_csv = _require_path_str(review_cfg, "decisions_csv", "review")

    autosave_every = max(1, int(review_cfg.get("autosave_every", 1)))
    host = str(review_cfg.get("host", "127.0.0.1")).strip() or "127.0.0.1"
    port = int(review_cfg.get("port", 5291))
    debug = bool(review_cfg.get("debug", False))

    return ReviewConfig(
        config_path=config_path,
        png_root=png_root,
        candidates_csv=candidates_csv,
        decisions_csv=decisions_csv,
        autosave_every=autosave_every,
        host=host,
        port=port,
        debug=debug,
    )


def run_review_app(cfg: ReviewConfig) -> int:
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


def run_review_main(config_path: Union[str, Path]) -> int:
    try:
        cfg = _build_review_config(config_path)
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
