from __future__ import annotations

import csv
import json
import os
import platform
import socket
import sys
import time
import uuid
import torch
import resource

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


def _now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_default(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    return str(x)


def _safe_json_dumps(d: Dict[str, Any]) -> str:
    return json.dumps(d, ensure_ascii=False, sort_keys=True, default=_json_default)


def _coerce_scalar(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def _cpu_snapshot() -> Dict[str, float]:
    r_self = resource.getrusage(resource.RUSAGE_SELF)
    r_child = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "user_s": float(r_self.ru_utime + r_child.ru_utime),
        "sys_s": float(r_self.ru_stime + r_child.ru_stime),
    }


@dataclass
class StageRecord:
    run_id: str
    stage_order: int
    stage_name: str
    parent_stage: str
    depth: int
    status: str
    error: str
    n_input: int
    n_output: int
    wall_time_s: float
    cpu_user_s: float
    cpu_sys_s: float
    cpu_total_s: float
    gpu_time_s: float
    notes_json: str


class RuntimeProfiler:
    def __init__(
        self,
        out_dir: str | Path,
        enabled: bool = True,
        run_id: Optional[str] = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.enabled = bool(enabled)

        self.run_id = run_id or (
            time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        )
        self.started_at_utc = _now_iso_utc()

        self.run_meta: Dict[str, Any] = {
            "run_id": self.run_id,
            "started_at_utc": self.started_at_utc,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "pid": os.getpid(),
            "cpu_count": os.cpu_count(),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": (
                int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
            ),
            "gpu_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
            ),
        }

        self._stage_records: List[StageRecord] = []
        self._active_stack: List[Dict[str, Any]] = []
        self._gpu_seconds_by_stage: Dict[str, float] = {}
        self._near_embedding_detail_rows: List[Dict[str, Any]] = []

        self._next_stage_order = 1
        self.finished_at_utc: Optional[str] = None

    @property
    def stage_records(self) -> List[StageRecord]:
        return self._stage_records

    def set_run_meta(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            self.run_meta[k] = _coerce_scalar(v)

    def _maybe_sync_cuda(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def add_gpu_time(self, stage_name: str, seconds: float) -> None:
        if not self.enabled:
            return
        self._gpu_seconds_by_stage[stage_name] = self._gpu_seconds_by_stage.get(
            stage_name, 0.0
        ) + float(seconds)

    @contextmanager
    def cuda_timer(self, stage_name: str) -> Iterator[None]:
        if (not self.enabled) or (not torch.cuda.is_available()):
            yield
            return

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        try:
            yield
        finally:
            end.record()
            end.synchronize()
            elapsed_ms = float(start.elapsed_time(end))
            self.add_gpu_time(stage_name, elapsed_ms / 1000.0)

    @contextmanager
    def stage(
        self,
        stage_name: str,
        n_input: int = 0,
        n_output: int = 0,
        **notes: Any,
    ) -> Iterator["RuntimeProfiler"]:
        if not self.enabled:
            yield self
            return

        self._maybe_sync_cuda()

        parent_stage = (
            self._active_stack[-1]["stage_name"] if self._active_stack else ""
        )
        depth = len(self._active_stack)
        stage_order = self._next_stage_order
        self._next_stage_order += 1

        active = {
            "stage_name": stage_name,
            "parent_stage": parent_stage,
            "depth": depth,
            "stage_order": stage_order,
            "n_input": int(n_input),
            "n_output": int(n_output),
            "notes": {k: _coerce_scalar(v) for k, v in notes.items()},
            "t0": time.perf_counter(),
            "cpu0": _cpu_snapshot(),
            "gpu0": float(self._gpu_seconds_by_stage.get(stage_name, 0.0)),
        }
        self._active_stack.append(active)

        status = "ok"
        error = ""

        try:
            yield self
        except BaseException as e:
            status = "error"
            error = f"{type(e).__name__}: {e}"
            raise
        finally:
            popped = self._active_stack.pop()

            self._maybe_sync_cuda()

            t1 = time.perf_counter()
            cpu1 = _cpu_snapshot()
            gpu1 = float(self._gpu_seconds_by_stage.get(stage_name, 0.0))

            wall_time_s = float(t1 - popped["t0"])
            cpu_user_s = float(cpu1["user_s"] - popped["cpu0"]["user_s"])
            cpu_sys_s = float(cpu1["sys_s"] - popped["cpu0"]["sys_s"])
            cpu_total_s = cpu_user_s + cpu_sys_s
            gpu_time_s = float(gpu1 - popped["gpu0"])

            self._stage_records.append(
                StageRecord(
                    run_id=self.run_id,
                    stage_order=int(popped["stage_order"]),
                    stage_name=str(popped["stage_name"]),
                    parent_stage=str(popped["parent_stage"]),
                    depth=int(popped["depth"]),
                    status=status,
                    error=error,
                    n_input=int(popped["n_input"]),
                    n_output=int(popped["n_output"]),
                    wall_time_s=wall_time_s,
                    cpu_user_s=cpu_user_s,
                    cpu_sys_s=cpu_sys_s,
                    cpu_total_s=cpu_total_s,
                    gpu_time_s=gpu_time_s,
                    notes_json=_safe_json_dumps(popped["notes"]),
                )
            )

    def record_near_embedding_detail(self, **kwargs: Any) -> None:
        if not self.enabled:
            return

        row: Dict[str, Any] = {"run_id": self.run_id}
        for k, v in kwargs.items():
            row[k] = _coerce_scalar(v)
        self._near_embedding_detail_rows.append(row)

    def _choose_total_stage(self) -> Optional[StageRecord]:
        totals = [r for r in self._stage_records if r.stage_name == "total_pipeline"]
        if totals:
            return totals[-1]
        return None

    def _write_stage_breakdown_csv(self) -> Path:
        out_path = self.out_dir / "perf_stage_breakdown.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "run_id",
            "stage_order",
            "stage_name",
            "parent_stage",
            "depth",
            "status",
            "error",
            "n_input",
            "n_output",
            "wall_time_s",
            "cpu_user_s",
            "cpu_sys_s",
            "cpu_total_s",
            "gpu_time_s",
            "notes_json",
        ]

        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in sorted(self._stage_records, key=lambda x: x.stage_order):
                writer.writerow(asdict(r))

        return out_path

    def _write_run_summary_csv(self) -> Path:
        out_path = self.out_dir / "perf_run_summary.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        total_stage = self._choose_total_stage()

        if total_stage is not None:
            total_wall_time_s = float(total_stage.wall_time_s)
            total_cpu_user_s = float(total_stage.cpu_user_s)
            total_cpu_sys_s = float(total_stage.cpu_sys_s)
            total_cpu_total_s = float(total_stage.cpu_total_s)
        else:
            # Fallback: sum only top-level stages (depth == 0)
            top_level = [r for r in self._stage_records if r.depth == 0]
            total_wall_time_s = float(sum(r.wall_time_s for r in top_level))
            total_cpu_user_s = float(sum(r.cpu_user_s for r in top_level))
            total_cpu_sys_s = float(sum(r.cpu_sys_s for r in top_level))
            total_cpu_total_s = float(sum(r.cpu_total_s for r in top_level))

        total_gpu_time_s = float(
            sum(
                r.gpu_time_s
                for r in self._stage_records
                if r.stage_name != "total_pipeline"
            )
        )

        self.finished_at_utc = _now_iso_utc()

        summary_row: Dict[str, Any] = {
            **{k: _coerce_scalar(v) for k, v in self.run_meta.items()},
            "finished_at_utc": self.finished_at_utc,
            "num_stage_records": len(self._stage_records),
            "total_wall_time_s": total_wall_time_s,
            "total_cpu_user_s": total_cpu_user_s,
            "total_cpu_sys_s": total_cpu_sys_s,
            "total_cpu_total_s": total_cpu_total_s,
            "total_gpu_time_s": total_gpu_time_s,
        }

        fieldnames = list(summary_row.keys())
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(summary_row)

        return out_path

    def _write_near_embedding_detail_csv(self) -> Optional[Path]:
        if not self._near_embedding_detail_rows:
            return None

        out_path = self.out_dir / "perf_near_embedding_detail.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames: List[str] = []
        seen = set()
        for row in self._near_embedding_detail_rows:
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)

        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self._near_embedding_detail_rows:
                writer.writerow(row)

        return out_path

    def save(self) -> Dict[str, Optional[Path]]:
        if not self.enabled:
            return {
                "stage_breakdown_csv": None,
                "run_summary_csv": None,
                "near_embedding_detail_csv": None,
            }

        self.out_dir.mkdir(parents=True, exist_ok=True)

        stage_csv = self._write_stage_breakdown_csv()
        summary_csv = self._write_run_summary_csv()
        near_detail_csv = self._write_near_embedding_detail_csv()

        return {
            "stage_breakdown_csv": stage_csv,
            "run_summary_csv": summary_csv,
            "near_embedding_detail_csv": near_detail_csv,
        }
