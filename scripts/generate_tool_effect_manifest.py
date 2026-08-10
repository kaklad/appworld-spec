#!/usr/bin/env python3
"""Generate a conservative static DB-effect manifest for AppWorld tools.

The output is intentionally an over-approximation.  Dynamic SQL tracing can
later add evidence, but must never silently narrow the static effect sets.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
APPS_DIR = ROOT / "src" / "appworld" / "apps"
DOCS_DIR = ROOT / "data" / "api_docs" / "standard"
DEFAULT_OUTPUT = ROOT / "speculative" / "tool_effects.json"

WRITE_METHODS = {
    "save",
    "save_all",
    "create_save",
    "delete",
    "delete_all",
    "create_db",
    "drop_db",
    "clean_db",
}
READ_METHODS = {
    "all",
    "by_id",
    "count",
    "exists",
    "find_all",
    "find_one",
    "find_one_or_raise",
    "first",
    "last",
    "size",
}
AUTH_WRITES = {
    "_signup",
    "_send_verification_code",
    "_verify_account",
    "_send_password_reset_code",
    "_reset_password",
    "_delete_account",
    "_update_account_name",
}
AUTH_NOTIFICATION_WRITES = {
    "_signup",
    "_send_verification_code",
    "_send_password_reset_code",
    "_reset_password",
}
AUTH_PROCESS_EFFECTS = {
    "_login": {"rng", "process_global_auth"},
    "_logout": {"process_global_auth"},
}
CALLER_EFFECTS: dict[str, tuple[set[str], set[str], set[str]]] = {
    "request_payment_card_validation": ({"admin"}, set(), set()),
    "request_payment_card_balance": ({"admin"}, set(), set()),
    "request_payment_card_debit": ({"admin"}, {"admin"}, set()),
    "request_payment_card_credit": ({"admin"}, {"admin"}, set()),
    "request_payment_card_creation": ({"admin"}, {"admin"}, set()),
    "does_file_exist": ({"file_system"}, set(), set()),
    "get_file_data": ({"file_system"}, set(), set()),
    "enlist_files": ({"file_system"}, set(), set()),
    "create_file": ({"file_system"}, {"file_system"}, set()),
    "update_file": ({"file_system"}, {"file_system"}, set()),
    "delete_file": ({"file_system"}, {"file_system"}, set()),
}


def call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def literal_string(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


class FunctionEffects(ast.NodeVisitor):
    def __init__(self, app_name: str, local_functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef]):
        self.app_name = app_name
        self.local_functions = local_functions
        self.read_dbs: set[str] = {app_name}
        self.write_dbs: set[str] = set()
        self.other_effects: set[str] = set()
        self.reasons: set[str] = {f"tool is implemented by the {app_name} app"}
        self.unknown_calls: set[str] = set()
        self._visited_functions: set[str] = set()

    def analyze(self, function: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if function.name in self._visited_functions:
            return
        self._visited_functions.add(function.name)
        for statement in function.body:
            self.visit(statement)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        name = call_name(node)
        if name in READ_METHODS:
            self.reasons.add(f"calls model read method {name}")
        if name in WRITE_METHODS:
            self.write_dbs.add(self.app_name)
            self.reasons.add(f"calls model write method {name}")
        if name in AUTH_WRITES:
            self.write_dbs.add(self.app_name)
            self.reasons.add(f"calls mutating authentication helper {name}")
            if name in AUTH_NOTIFICATION_WRITES:
                notification_app = "phone" if self.app_name == "phone" else "gmail"
                self.read_dbs.add(notification_app)
                self.write_dbs.add(notification_app)
                self.reasons.add(
                    f"authentication helper may send a notification through {notification_app}"
                )
        if name in AUTH_PROCESS_EFFECTS:
            self.other_effects.update(AUTH_PROCESS_EFFECTS[name])
            self.reasons.add(f"calls stateful authentication helper {name}")
        if name in CALLER_EFFECTS:
            reads, writes, other = CALLER_EFFECTS[name]
            self.read_dbs.update(reads)
            self.write_dbs.update(writes)
            self.other_effects.update(other)
            self.reasons.add(f"calls cross-app helper {name}")
        if name == "notify_on_email":
            self.read_dbs.add("gmail")
            self.write_dbs.add("gmail")
            self.reasons.add("sends a notification through Gmail")
        if name == "notify_on_phone":
            self.read_dbs.add("phone")
            self.write_dbs.add("phone")
            self.reasons.add("sends a notification through Phone")
        if name in {"get_unique_id", "randrange", "random", "choice", "sample", "shuffle"}:
            self.other_effects.add("rng")
        if name == "call_api" and len(node.args) >= 2:
            target_app = literal_string(node.args[0])
            target_api = literal_string(node.args[1])
            if target_app:
                self.read_dbs.add(target_app)
                # Private cross-app calls are conservatively considered writes unless
                # their name clearly denotes a read.
                if not target_api or not target_api.startswith(("show_", "search_", "list_", "get_", "check_")):
                    self.write_dbs.add(target_app)
                self.reasons.add(f"calls {target_app}.{target_api or '*'}")
        if name and name in self.local_functions and name not in self._visited_functions:
            self.analyze(self.local_functions[name])
        self.generic_visit(node)


def load_docs() -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for path in sorted(DOCS_DIR.glob("*.json")):
        app_docs = json.loads(path.read_text())
        docs.extend(app_docs.values())
    return sorted(docs, key=lambda item: (item["app_name"], item["api_name"]))


def parse_module(app_name: str) -> tuple[dict[str, ast.FunctionDef | ast.AsyncFunctionDef], Path]:
    path = APPS_DIR / app_name / "apis.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    return functions, path


def generate() -> dict[str, Any]:
    tools: dict[str, Any] = {}
    module_cache: dict[str, tuple[dict[str, ast.FunctionDef | ast.AsyncFunctionDef], Path]] = {}
    for doc in load_docs():
        app_name = doc["app_name"]
        api_name = doc["api_name"]
        functions, source_path = module_cache.setdefault(app_name, parse_module(app_name))
        function = functions.get(api_name)
        if function is None:
            tools[f"{app_name}.{api_name}"] = {
                "app_name": app_name,
                "api_name": api_name,
                "method": doc["method"],
                "path": doc["path"],
                "source": str(source_path.relative_to(ROOT)),
                "static": {
                    "read_dbs": [app_name],
                    "write_dbs": [app_name],
                    "other_effects": [],
                    "confidence": "low",
                    "needs_dynamic_trace": True,
                    "reasons": ["endpoint function was not found; using conservative fallback"],
                },
                "dynamic": {"tested": False, "runs": 0, "read_dbs": [], "write_dbs": []},
            }
            continue

        analyzer = FunctionEffects(app_name, functions)
        analyzer.analyze(function)
        non_get_without_detected_write = doc["method"].upper() != "GET" and not analyzer.write_dbs
        confidence = "high" if analyzer.write_dbs or doc["method"].upper() == "GET" else "medium"
        if non_get_without_detected_write or analyzer.other_effects:
            confidence = "medium"
        tools[f"{app_name}.{api_name}"] = {
            "app_name": app_name,
            "api_name": api_name,
            "method": doc["method"].upper(),
            "path": doc["path"],
            "source": str(source_path.relative_to(ROOT)),
            "source_line": function.lineno,
            "static": {
                "read_dbs": sorted(analyzer.read_dbs),
                "write_dbs": sorted(analyzer.write_dbs),
                "other_effects": sorted(analyzer.other_effects),
                "confidence": confidence,
                "needs_dynamic_trace": confidence != "high" or len(analyzer.read_dbs) > 1,
                "reasons": sorted(analyzer.reasons),
            },
            "dynamic": {"tested": False, "runs": 0, "read_dbs": [], "write_dbs": []},
        }

    methods = Counter(tool["method"] for tool in tools.values())
    confidences = Counter(tool["static"]["confidence"] for tool in tools.values())
    return {
        "schema_version": 1,
        "generator": "scripts/generate_tool_effect_manifest.py",
        "policy": {
            "description": "Conservative static approximation; dynamic evidence may widen but not narrow effects.",
            "effective_read_dbs": "static.read_dbs union dynamic.read_dbs",
            "effective_write_dbs": "static.write_dbs union dynamic.write_dbs",
        },
        "summary": {
            "tool_count": len(tools),
            "methods": dict(sorted(methods.items())),
            "confidence": dict(sorted(confidences.items())),
        },
        "tools": tools,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = generate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    print(f"Wrote {manifest['summary']['tool_count']} tools to {args.output}")
    print(json.dumps(manifest["summary"], indent=2))


if __name__ == "__main__":
    main()
