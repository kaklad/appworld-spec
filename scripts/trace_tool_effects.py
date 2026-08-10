#!/usr/bin/env python3
"""Trace per-tool SQLite effects while running AppWorld ground-truth solutions."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import types
from collections import Counter
from pathlib import Path
from typing import Any

from sqlalchemy import event

from appworld import AppWorld, load_task_ids


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "speculative" / "traces"
WRITE_OPERATIONS = {"ALTER", "CREATE", "DELETE", "DROP", "INSERT", "REPLACE", "UPDATE"}
READ_OPERATIONS = {"EXPLAIN", "PRAGMA", "SELECT", "WITH"}


def operation_of(statement: str) -> str:
    operation = statement.lstrip().split(None, 1)[0].upper() if statement.strip() else "UNKNOWN"
    if operation == "WITH":
        upper = statement.upper()
        for candidate in (" INSERT ", " UPDATE ", " DELETE ", " REPLACE "):
            if candidate in upper:
                return candidate.strip()
    return operation


def rng_fingerprint() -> str:
    return hashlib.sha256(repr(random.getstate()).encode()).hexdigest()


class ToolDBTracer:
    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.active_span_ids: list[int] = []
        self.next_span_id = 0
        self.spans: dict[int, dict[str, Any]] = {}
        self.completed: list[dict[str, Any]] = []
        self.listeners: list[tuple[Any, Any]] = []

    def attach(self, world: AppWorld) -> None:
        assert world.models is not None
        for app_name, models in world.models.items():
            engine = models.SQLModel.db.engine

            def listener(
                conn: Any,
                cursor: Any,
                statement: str,
                parameters: Any,
                context: Any,
                executemany: bool,
                app_name: str = app_name,
            ) -> None:
                if not self.active_span_ids:
                    return
                operation = operation_of(statement)
                for span_id in self.active_span_ids:
                    span = self.spans[span_id]
                    span["sql_counts"][(app_name, operation)] += 1

            event.listen(engine, "before_cursor_execute", listener)
            self.listeners.append((engine, listener))

    def detach(self) -> None:
        for engine, listener in self.listeners:
            event.remove(engine, "before_cursor_execute", listener)
        self.listeners.clear()

    def start(self, tool: str) -> int:
        span_id = self.next_span_id
        self.next_span_id += 1
        self.spans[span_id] = {
            "span_id": span_id,
            "task_id": self.task_id,
            "tool": tool,
            "sql_counts": Counter(),
            "rng_before": rng_fingerprint(),
        }
        self.active_span_ids.append(span_id)
        return span_id

    def stop(self, span_id: int, status_code: int | None, error: str | None) -> None:
        popped = self.active_span_ids.pop()
        if popped != span_id:
            raise RuntimeError(f"Unbalanced trace stack: expected {span_id}, found {popped}")
        span = self.spans.pop(span_id)
        sql_counts: Counter[tuple[str, str]] = span.pop("sql_counts")
        read_dbs: set[str] = set()
        write_dbs: set[str] = set()
        unknown_operations: set[str] = set()
        for (app_name, operation), _ in sql_counts.items():
            if operation in READ_OPERATIONS:
                read_dbs.add(app_name)
            elif operation in WRITE_OPERATIONS:
                read_dbs.add(app_name)
                write_dbs.add(app_name)
            else:
                unknown_operations.add(operation)
        span.update(
            {
                "status_code": status_code,
                "success": error is None and status_code is not None and status_code < 400,
                "error": error,
                "read_dbs": sorted(read_dbs),
                "write_dbs": sorted(write_dbs),
                "other_effects": ["rng"] if span.pop("rng_before") != rng_fingerprint() else [],
                "sql_counts": [
                    {"db": db, "operation": operation, "count": count}
                    for (db, operation), count in sorted(sql_counts.items())
                ],
                "unknown_operations": sorted(unknown_operations),
            }
        )
        self.completed.append(span)


def trace_task(task_id: str, output_path: Path) -> dict[str, Any]:
    tracer = ToolDBTracer(task_id)
    task_error: str | None = None
    execution_output = ""
    with AppWorld(
        task_id=task_id,
        experiment_name=f"tool_effect_trace/{task_id}",
        load_ground_truth=True,
        ground_truth_mode="full",
        raise_on_failure=False,
    ) as world:
        tracer.attach(world)
        original_request = world.requester._request

        def traced_request(
            requester_self: Any,
            _app_name: str,
            _api_name: str,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            span_id = tracer.start(f"{_app_name}.{_api_name}")
            response = None
            error = None
            try:
                response = original_request(
                    _app_name=_app_name,
                    _api_name=_api_name,
                    *args,
                    **kwargs,
                )
                return response
            except Exception as exception:
                error = f"{type(exception).__name__}: {exception}"
                raise
            finally:
                tracer.stop(span_id, getattr(response, "status_code", None), error)

        world.requester._request = types.MethodType(traced_request, world.requester)  # type: ignore[method-assign]
        try:
            code = world.task.ground_truth.compiled_solution_code
            code += "\nsolution(apis, requester)"
            execution_output = world.execute(code)
            if execution_output.startswith("Execution failed"):
                task_error = execution_output[-2000:]
        except Exception as exception:
            task_error = f"{type(exception).__name__}: {exception}"
        finally:
            world.requester._request = original_request  # type: ignore[method-assign]
            tracer.detach()

    result = {
        "task_id": task_id,
        "task_error": task_error,
        "call_count": len(tracer.completed),
        "calls": tracer.completed,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task-id")
    group.add_argument("--dataset", choices=("train", "dev"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-tasks", type=int)
    args = parser.parse_args()

    task_ids = [args.task_id] if args.task_id else load_task_ids(args.dataset)
    if args.max_tasks is not None:
        task_ids = task_ids[: args.max_tasks]
    failures = 0
    calls = 0
    for index, task_id in enumerate(task_ids, 1):
        assert task_id is not None
        print(f"[{index}/{len(task_ids)}] tracing {task_id}", flush=True)
        result = trace_task(task_id, args.output_dir / f"{task_id}.json")
        calls += result["call_count"]
        if result["task_error"]:
            failures += 1
            print(f"  failed: {result['task_error'][-300:]}", file=sys.stderr, flush=True)
    print(json.dumps({"tasks": len(task_ids), "failures": failures, "calls": calls}, indent=2))


if __name__ == "__main__":
    main()
