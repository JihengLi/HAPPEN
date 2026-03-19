from __future__ import annotations

import argparse
import logging

from pathlib import Path

import loader
from store import DecisionStore
from app import create_app


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("Human-in-the-loop duplicate review server")
    ap.add_argument(
        "--root-dir",
        type=Path,
        required=True,
        help="Format: <root_dir>/<dataset>/<subject>/<session?>/<scan_uid>.png",
    )
    ap.add_argument(
        "--in-csv",
        type=Path,
        required=True,
        help="CSV storing candidate groups.",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        required=True,
        help="Output QA CSV (one pair per row).",
    )
    ap.add_argument(
        "--autosave-every",
        type=int,
        default=1,
        help="Write out_csv every N updates.",
    )
    ap.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host",
    )
    ap.add_argument("--port", type=int, default=5291, help="Port")
    ap.add_argument("--debug", action="store_true", help="Debug mode")
    return ap


def main() -> int:
    args = build_argparser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    root_dir = args.root_dir.expanduser().resolve()
    in_csv = args.in_csv.expanduser().resolve()
    out_csv = args.out_csv.expanduser().resolve()

    if not root_dir.exists():
        logging.error("root_dir does not exist: %s", root_dir)
        return 2
    if not in_csv.exists():
        logging.error("in_csv does not exist: %s", in_csv)
        return 2

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Loading duplicates CSV: %s", in_csv)
    df = loader.load_in_csv(in_csv)
    slots = loader.detect_candidate_slots(df)
    queries = loader.parse_queries(df, slots=slots)
    if not queries:
        logging.error("No queries parsed (check CSV headers/content).")
        return 3

    queries = loader.dedup_undirected_pairs(queries)
    index = loader.build_index(queries)
    logging.info("Parsed datasets=%d", len(index))

    store = DecisionStore(
        out_csv=out_csv, autosave_every=max(1, int(args.autosave_every))
    )
    store.init()
    logging.info(
        "Loaded existing decisions=%d",
        store.num_decisions() if hasattr(store, "num_decisions") else 0,
    )
    valid_item_keys = loader.collect_valid_item_keys(queries)
    pruned = store.prune_to_existing_items(valid_item_keys)
    logging.info(
        "Loaded existing decisions=%d, pruned=%d, remaining=%d",
        store.num_decisions() + pruned,
        pruned,
        store.num_decisions(),
    )
    store.export()

    app = create_app(
        index=index,
        root_dir=root_dir,
        decision_store=store,
    )

    logging.info("Starting server: http://%s:%d", args.host, args.port)
    try:
        app.run(
            host=args.host,
            port=int(args.port),
            debug=False,
            use_reloader=False,
            threaded=True,
        )
    except KeyboardInterrupt:
        try:
            store.export()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
