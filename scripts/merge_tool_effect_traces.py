#!/usr/bin/env python3
"""Merge raw dynamic tool traces into the static effect manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "speculative" / "tool_effects.json")
    parser.add_argument(
        "--traces",
        type=Path,
        nargs="+",
        default=[ROOT / "speculative" / "traces"],
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.manifest

    manifest = json.loads(args.manifest.read_text())
    aggregates: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "task_ids": set(),
            "read_dbs": set(),
            "write_dbs": set(),
            "other_effects": set(),
            "sql_operations": Counter(),
            "unknown_operations": set(),
        }
    )
    task_failures = 0
    trace_files = sorted(path for directory in args.traces for path in directory.rglob("*.json"))
    for path in trace_files:
        trace = json.loads(path.read_text())
        task_failures += bool(trace.get("task_error"))
        for call in trace["calls"]:
            aggregate = aggregates[call["tool"]]
            aggregate["runs"] += 1
            aggregate["successful_runs"] += bool(call["success"])
            aggregate["failed_runs"] += not call["success"]
            aggregate["task_ids"].add(call["task_id"])
            aggregate["read_dbs"].update(call["read_dbs"])
            aggregate["write_dbs"].update(call["write_dbs"])
            aggregate["other_effects"].update(call["other_effects"])
            aggregate["unknown_operations"].update(call["unknown_operations"])
            for sql_count in call["sql_counts"]:
                aggregate["sql_operations"][sql_count["operation"]] += sql_count["count"]

    widened_tools: list[str] = []
    unknown_tools: list[str] = []
    for tool_name, aggregate in aggregates.items():
        if tool_name not in manifest["tools"]:
            unknown_tools.append(tool_name)
            continue
        tool = manifest["tools"][tool_name]
        static = tool["static"]
        observed_reads = set(aggregate["read_dbs"])
        observed_writes = set(aggregate["write_dbs"])
        if not observed_reads <= set(static["read_dbs"]) or not observed_writes <= set(
            static["write_dbs"]
        ):
            widened_tools.append(tool_name)
        tool["dynamic"] = {
            "tested": True,
            "runs": aggregate["runs"],
            "successful_runs": aggregate["successful_runs"],
            "failed_runs": aggregate["failed_runs"],
            "task_count": len(aggregate["task_ids"]),
            "read_dbs": sorted(observed_reads),
            "write_dbs": sorted(observed_writes),
            "other_effects": sorted(aggregate["other_effects"]),
            "sql_operations": dict(sorted(aggregate["sql_operations"].items())),
            "unknown_operations": sorted(aggregate["unknown_operations"]),
        }
        tool["effective"] = {
            "read_dbs": sorted(set(static["read_dbs"]) | observed_reads),
            "write_dbs": sorted(set(static["write_dbs"]) | observed_writes),
            "other_effects": sorted(set(static["other_effects"]) | set(aggregate["other_effects"])),
        }

    # Give consumers one uniform field even for tools that were not exercised.
    for tool in manifest["tools"].values():
        static = tool["static"]
        dynamic = tool["dynamic"]
        tool.setdefault(
            "effective",
            {
                "read_dbs": sorted(set(static["read_dbs"]) | set(dynamic["read_dbs"])),
                "write_dbs": sorted(set(static["write_dbs"]) | set(dynamic["write_dbs"])),
                "other_effects": sorted(
                    set(static["other_effects"]) | set(dynamic.get("other_effects", []))
                ),
            },
        )

    manifest["dynamic_summary"] = {
        "trace_files": len(trace_files),
        "task_failures": task_failures,
        "traced_tools": len(set(aggregates) & set(manifest["tools"])),
        "untraced_tools": len(set(manifest["tools"]) - set(aggregates)),
        "widened_tools": sorted(widened_tools),
        "unknown_tools": sorted(unknown_tools),
    }
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["dynamic_summary"], indent=2))


if __name__ == "__main__":
    main()
