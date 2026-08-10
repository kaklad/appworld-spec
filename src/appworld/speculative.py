"""Manifest-guided snapshots and speculative AppWorld tool execution.

This module provides a correctness-first, single-process implementation.  It
computes branches sequentially, rolls the live world back after every branch,
and retains branch snapshots that can later be promoted.  The same snapshot
format can be used by isolated worker processes in a parallel implementation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from appworld.apps.lib.models.db import ModelHashHandler
from appworld.common.imports import import_apis_module
from appworld.common.path_store import path_store
from appworld.speculative_trace import TraceRecorder, elapsed_ms, monotonic_ns


if TYPE_CHECKING:
    from appworld.environment import AppWorld


SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
DEFAULT_API_DOCS_DIRECTORY = Path("data") / "api_docs" / "standard"


@dataclass(frozen=True)
class CredentialRef:
    """A stable logical credential handle used instead of a volatile token."""

    app_name: str
    principal: str = "current_user"
    field: str = "access_token"

    def to_json(self) -> dict[str, str]:
        return {
            "$credential": self.app_name,
            "$principal": self.principal,
            "$field": self.field,
        }

    @property
    def key(self) -> str:
        return f"{self.app_name}:{self.principal}:{self.field}"


def _restore_credential_refs(value: Any) -> Any:
    if isinstance(value, list):
        return [_restore_credential_refs(item) for item in value]
    if isinstance(value, dict):
        if "$credential" in value:
            return CredentialRef(
                app_name=str(value["$credential"]),
                principal=str(value.get("$principal", "current_user")),
                field=str(value.get("$field", "access_token")),
            )
        return {key: _restore_credential_refs(item) for key, item in value.items()}
    return value


def _normalize_json_value(value: Any, declared_type: str | None = None) -> Any:
    if isinstance(value, CredentialRef):
        return value.to_json()
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, datetime | date):
        return value.isoformat()
    if declared_type == "integer" and not isinstance(value, bool):
        if isinstance(value, int | float) and int(value) == value:
            return int(value)
        if isinstance(value, str) and re.fullmatch(r"[-+]?\d+", value.strip()):
            return int(value)
    if declared_type == "number" and not isinstance(value, bool):
        if isinstance(value, int | float | str):
            try:
                return float(value)
            except ValueError:
                pass
    if declared_type == "boolean":
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return value.lower() == "true"
        if isinstance(value, bool):
            return value
    if declared_type and declared_type.startswith("list[") and isinstance(value, tuple | list):
        item_type = declared_type.removeprefix("list[").removesuffix("]")
        return [_normalize_json_value(item, item_type) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple | list):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, float) and (value != value or value in {float("inf"), float("-inf")}):
        raise ValueError("NaN and infinity are not valid canonical tool arguments.")
    return value


class ToolSchemaRegistry:
    """Load the public AppWorld API parameter schemas used for canonicalization."""

    def __init__(self, directory: str | os.PathLike[str] | None = None) -> None:
        if directory is None:
            directory = os.path.join(path_store.root, DEFAULT_API_DOCS_DIRECTORY)
        self.directory = Path(directory)
        self.tools: dict[str, dict[str, Any]] = {}
        for file_path in sorted(self.directory.glob("*.json")):
            for doc in json.loads(file_path.read_text()).values():
                self.tools[f"{doc['app_name']}.{doc['api_name']}"] = doc

    def parameters(self, app_name: str, api_name: str) -> list[dict[str, Any]]:
        tool_name = f"{app_name}.{api_name}"
        if tool_name not in self.tools:
            raise KeyError(f"Tool {tool_name!r} is missing from {self.directory}.")
        return self.tools[tool_name]["parameters"]


@dataclass(frozen=True)
class CanonicalToolCall:
    """Schema-normalized identity of one tool call, used for exact hit matching."""

    app_name: str
    api_name: str
    arguments_json: str

    @classmethod
    def from_action(
        cls,
        action: "ToolAction",
        schemas: ToolSchemaRegistry | None = None,
    ) -> "CanonicalToolCall":
        schemas = schemas or ToolSchemaRegistry()
        parameters = schemas.parameters(action.app_name, action.api_name)
        declared = {parameter["name"]: parameter for parameter in parameters}
        unknown = set(action.arguments) - set(declared)
        if unknown:
            raise ValueError(f"Unknown arguments for {action.tool_name}: {sorted(unknown)}")
        missing = {
            name
            for name, parameter in declared.items()
            if parameter["required"] and name not in action.arguments
        }
        if missing:
            raise ValueError(f"Missing required arguments for {action.tool_name}: {sorted(missing)}")
        normalized: dict[str, Any] = {}
        for name, parameter in declared.items():
            if name in action.arguments:
                value = action.arguments[name]
            elif not parameter["required"]:
                value = parameter["default"]
            else:  # covered by missing, kept explicit for type checkers.
                continue
            normalized[name] = _normalize_json_value(value, parameter["type"])
        arguments_json = json.dumps(
            normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return cls(action.app_name, action.api_name, arguments_json)

    @property
    def arguments(self) -> dict[str, Any]:
        return dict(json.loads(self.arguments_json))

    @property
    def key(self) -> str:
        return f"{self.app_name}.{self.api_name}:{self.arguments_json}"

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.key.encode()).hexdigest()

    def to_action(self, branch_id: str = "") -> "ToolAction":
        return ToolAction(
            self.app_name,
            self.api_name,
            _restore_credential_refs(self.arguments),
            branch_id,
        )


@dataclass
class WorkingMemory:
    """Compact observation-derived state supplied to the actor instead of raw history."""

    credentials: dict[str, CredentialRef | str] = field(default_factory=dict)
    credential_values: dict[str, str] = field(default_factory=dict, repr=False)
    entities: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    pagination: dict[str, Any] = field(default_factory=dict)
    completed_steps: list[str] = field(default_factory=list)
    pending_steps: list[str] = field(default_factory=list)
    completed_action_keys: list[str] = field(default_factory=list)
    recent_errors: list[dict[str, Any]] = field(default_factory=list)

    def clone(self) -> "WorkingMemory":
        return copy.deepcopy(self)

    def record_tool_result(
        self,
        action: CanonicalToolCall,
        observation: Any,
        succeeded: bool,
        max_errors: int = 5,
    ) -> None:
        if succeeded:
            self.completed_action_keys.append(action.key)
            self._record_credentials(action, observation)
        else:
            self.recent_errors.append(
                {"action": action.key, "observation": _normalize_json_value(observation)}
            )
            self.recent_errors = self.recent_errors[-max_errors:]

    def seed_usernames(self, app_names: list[str] | tuple[str, ...], username: str) -> None:
        for app_name in app_names:
            reference = CredentialRef(app_name, field="username")
            self.credentials[f"{app_name}.username"] = reference
            self.credential_values[reference.key] = username

    def _record_credentials(self, action: CanonicalToolCall, observation: Any) -> None:
        if action.app_name == "supervisor" and action.api_name == "show_account_passwords":
            if isinstance(observation, list):
                for item in observation:
                    if not isinstance(item, dict):
                        continue
                    app_name = item.get("account_name")
                    password = item.get("password")
                    if isinstance(app_name, str) and isinstance(password, str):
                        reference = CredentialRef(app_name, field="password")
                        self.credentials[f"{app_name}.password"] = reference
                        self.credential_values[reference.key] = password
        if action.api_name == "login" and isinstance(observation, dict):
            access_token = observation.get("access_token")
            if isinstance(access_token, str):
                reference = CredentialRef(action.app_name, field="access_token")
                self.credentials[f"{action.app_name}.access_token"] = reference
                self.credential_values[reference.key] = access_token

    def resolve_credentials(self, value: Any) -> Any:
        if isinstance(value, CredentialRef):
            if value.key not in self.credential_values:
                raise KeyError(f"Credential {value.key!r} is not available in working memory.")
            return self.credential_values[value.key]
        if isinstance(value, list):
            return [self.resolve_credentials(item) for item in value]
        if isinstance(value, dict):
            return {key: self.resolve_credentials(item) for key, item in value.items()}
        return value

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "credentials": {
                key: value.to_json() if isinstance(value, CredentialRef) else value
                for key, value in sorted(self.credentials.items())
            },
            "entities": _normalize_json_value(self.entities),
            "constraints": _normalize_json_value(self.constraints),
            "pagination": _normalize_json_value(self.pagination),
            "completed_steps": list(self.completed_steps),
            "pending_steps": list(self.pending_steps),
            "completed_action_keys": list(self.completed_action_keys),
            "recent_errors": copy.deepcopy(self.recent_errors),
        }


@dataclass(frozen=True)
class ActorEvent:
    action: CanonicalToolCall
    observation: Any
    succeeded: bool

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "action": {
                "app_name": self.action.app_name,
                "api_name": self.action.api_name,
                "arguments": self.action.arguments,
            },
            "observation": _normalize_json_value(self.observation),
            "succeeded": self.succeeded,
        }


@dataclass
class ActorContext:
    """Compact, branchable input context for a deterministic large actor."""

    task_instruction: str
    working_memory: WorkingMemory = field(default_factory=WorkingMemory)
    available_tools: tuple[str, ...] = ()
    environment_state_id: str = ""
    context_id: str = "root"
    recent_events: list[ActorEvent] = field(default_factory=list)
    recent_event_limit: int = 4

    def after_tool(
        self,
        action: CanonicalToolCall,
        observation: Any,
        succeeded: bool,
        environment_state_id: str,
    ) -> "ActorContext":
        memory = self.working_memory.clone()
        memory.record_tool_result(action, observation, succeeded)
        event = ActorEvent(action, copy.deepcopy(observation), succeeded)
        recent_events = [*copy.deepcopy(self.recent_events), event][-self.recent_event_limit :]
        identity_payload = json.dumps(
            {
                "parent": self.context_id,
                "action": action.key,
                "observation": _normalize_json_value(observation),
                "succeeded": succeeded,
                "environment_state_id": environment_state_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return ActorContext(
            task_instruction=self.task_instruction,
            working_memory=memory,
            available_tools=self.available_tools,
            environment_state_id=environment_state_id,
            context_id=hashlib.sha256(identity_payload.encode()).hexdigest(),
            recent_events=recent_events,
            recent_event_limit=self.recent_event_limit,
        )

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "task": self.task_instruction,
            "working_memory": self.working_memory.to_prompt_dict(),
            "recent_events": [event.to_prompt_dict() for event in self.recent_events],
            "available_tools": list(self.available_tools),
            "environment_state_id": self.environment_state_id,
            "context_id": self.context_id,
        }


@dataclass(frozen=True)
class ToolEffect:
    read_dbs: tuple[str, ...]
    write_dbs: tuple[str, ...]
    other_effects: tuple[str, ...]
    dynamically_tested: bool


class ToolEffectManifest:
    """Read effect information from ``speculative/tool_effects.json``."""

    def __init__(self, file_path: str | os.PathLike[str] | None = None) -> None:
        if file_path is None:
            file_path = os.path.join(path_store.root, "speculative", "tool_effects.json")
        self.file_path = Path(file_path)
        data = json.loads(self.file_path.read_text())
        self.schema_version = data["schema_version"]
        self.tools: dict[str, dict[str, Any]] = data["tools"]

    def effect(self, app_name: str, api_name: str) -> ToolEffect:
        tool_name = f"{app_name}.{api_name}"
        if tool_name not in self.tools:
            raise KeyError(f"Tool {tool_name!r} is missing from {self.file_path}.")
        entry = self.tools[tool_name]
        effective = entry.get("effective", entry["static"])
        dynamic = entry.get("dynamic", {})
        return ToolEffect(
            read_dbs=tuple(effective["read_dbs"]),
            write_dbs=tuple(effective["write_dbs"]),
            other_effects=tuple(effective.get("other_effects", [])),
            dynamically_tested=bool(dynamic.get("tested", False)),
        )


@dataclass(frozen=True)
class ToolAction:
    app_name: str
    api_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    branch_id: str = ""

    @property
    def tool_name(self) -> str:
        return f"{self.app_name}.{self.api_name}"


@dataclass
class RuntimeState:
    random_state: object
    request_log: list[dict[str, Any]]
    request_count_reset_on: int
    auth_blacklists: dict[str, set[str]]
    model_hashes: Any

    @classmethod
    def capture(cls, world: "AppWorld") -> "RuntimeState":
        auth_blacklists: dict[str, set[str]] = {}
        for app_name in world.requester.apps:
            module = import_apis_module(app_name)
            manager = getattr(module, "logging_manager", None)
            if manager is not None:
                auth_blacklists[app_name] = set(manager.blacklisted_tokens)
        tracker = world.requester.request_tracker
        return cls(
            random_state=random.getstate(),
            request_log=copy.deepcopy(tracker.requests),
            request_count_reset_on=tracker.num_requests_reset_on,
            auth_blacklists=auth_blacklists,
            model_hashes=copy.deepcopy(ModelHashHandler.data),
        )

    def restore(self, world: "AppWorld") -> None:
        random.setstate(self.random_state)
        tracker = world.requester.request_tracker
        tracker.requests = copy.deepcopy(self.request_log)
        tracker.num_requests_reset_on = self.request_count_reset_on
        for app_name, tokens in self.auth_blacklists.items():
            module = import_apis_module(app_name)
            manager = getattr(module, "logging_manager", None)
            if manager is not None:
                manager.blacklisted_tokens = set(tokens)
        ModelHashHandler.data = copy.deepcopy(self.model_hashes)


@dataclass
class WorldSnapshot:
    snapshot_id: str
    directory: Path
    app_names: tuple[str, ...]
    tracked_changes: dict[str, list[Any]]
    runtime: RuntimeState

    def restore(self, world: "AppWorld") -> None:
        if world.models is None:
            raise RuntimeError("Speculative snapshots require a local AppWorld.")
        for app_name in self.app_names:
            source_path = self.directory / f"{app_name}.db"
            models = world.models[app_name]
            target_connection = models.SQLModel.db.connection
            source_connection = sqlite3.connect(source_path)
            try:
                source_connection.backup(target_connection)
                target_connection.commit()
            finally:
                source_connection.close()
            models.SQLModel.db.tracker.changes = copy.deepcopy(self.tracked_changes[app_name])
        self.runtime.restore(world)


class SnapshotStore:
    """Capture only selected app databases from a live local world."""

    def __init__(self, root_directory: str | os.PathLike[str]) -> None:
        self.root_directory = Path(root_directory)
        self.root_directory.mkdir(parents=True, exist_ok=True)

    def capture(
        self, world: "AppWorld", snapshot_id: str, app_names: list[str] | tuple[str, ...]
    ) -> WorldSnapshot:
        if not SAFE_ID_PATTERN.fullmatch(snapshot_id):
            raise ValueError("snapshot_id may contain only letters, numbers, '.', '_' and '-'.")
        if world.models is None:
            raise RuntimeError("Speculative snapshots require a local AppWorld.")
        app_names = tuple(sorted(set(app_names)))
        unknown_apps = set(app_names) - set(world.models.keys())
        if unknown_apps:
            raise ValueError(f"Snapshot requested unloaded apps: {sorted(unknown_apps)}")
        directory = self.root_directory / snapshot_id
        directory.mkdir(parents=True, exist_ok=True)
        tracked_changes: dict[str, list[Any]] = {}
        for app_name in app_names:
            models = world.models[app_name]
            target_path = directory / f"{app_name}.db"
            target_connection = sqlite3.connect(target_path)
            try:
                models.SQLModel.db.connection.backup(target_connection)
                target_connection.commit()
            finally:
                target_connection.close()
            tracked_changes[app_name] = copy.deepcopy(models.SQLModel.db.tracker.changes)
        metadata = {"snapshot_id": snapshot_id, "app_names": list(app_names)}
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return WorldSnapshot(
            snapshot_id=snapshot_id,
            directory=directory,
            app_names=app_names,
            tracked_changes=tracked_changes,
            runtime=RuntimeState.capture(world),
        )


@dataclass
class BranchResult:
    action: ToolAction
    canonical_action: CanonicalToolCall
    effect: ToolEffect
    observation: dict[str, Any] | list[Any] | None
    status_code: int | None
    error: str | None
    parent_snapshot: WorldSnapshot
    snapshot: WorldSnapshot

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.status_code is not None and self.status_code < 400


@dataclass
class BranchContinuation:
    """Actor-side continuation precomputed from one speculative observation."""

    branch_id: str
    parent_context_id: str
    parent_environment_state_id: str
    candidate_action: CanonicalToolCall
    observation: Any
    next_environment_state_id: str
    next_actor_context: ActorContext
    precomputed_next_action: CanonicalToolCall | None = None

    @classmethod
    def from_result(
        cls,
        result: BranchResult,
        parent_context: ActorContext,
        branch_id: str | None = None,
    ) -> "BranchContinuation":
        branch_id = branch_id or result.action.branch_id
        if not branch_id:
            raise ValueError("A BranchContinuation requires a non-empty branch_id.")
        next_state_id = result.snapshot.snapshot_id
        next_context = parent_context.after_tool(
            action=result.canonical_action,
            observation=result.observation,
            succeeded=result.succeeded,
            environment_state_id=next_state_id,
        )
        return cls(
            branch_id=branch_id,
            parent_context_id=parent_context.context_id,
            parent_environment_state_id=parent_context.environment_state_id,
            candidate_action=result.canonical_action,
            observation=copy.deepcopy(result.observation),
            next_environment_state_id=next_state_id,
            next_actor_context=next_context,
        )

    def set_next_action(self, action: CanonicalToolCall) -> None:
        """Attach the temperature-zero large actor result for this branch."""
        self.precomputed_next_action = action

    def is_hit(self, actor_action: CanonicalToolCall) -> bool:
        return self.candidate_action == actor_action

    def is_fresh_for(self, context: ActorContext) -> bool:
        return (
            self.parent_context_id == context.context_id
            and self.parent_environment_state_id == context.environment_state_id
        )


class Speculator:
    """Evaluate tool branches and leave the supplied world unchanged."""

    def __init__(
        self,
        world: "AppWorld",
        manifest: ToolEffectManifest | None = None,
        snapshot_directory: str | os.PathLike[str] | None = None,
        trace: TraceRecorder | None = None,
    ) -> None:
        if world.remote_environment_url or world.remote_apis_url or world.remote_mcp_url:
            raise ValueError("The single-process speculator supports only a fully local AppWorld.")
        if world.models is None:
            raise RuntimeError("The world does not expose local models.")
        self.world = world
        self.manifest = manifest or ToolEffectManifest()
        self.schemas = ToolSchemaRegistry()
        if snapshot_directory is None:
            snapshot_directory = os.path.join(world.output_checkpoints_directory, "speculator")
        self.store = SnapshotStore(snapshot_directory)
        self.branches: dict[str, BranchResult] = {}
        self._round = 0
        self.trace = trace

    def speculate(
        self,
        actions: list[ToolAction],
        working_memory: WorkingMemory | None = None,
    ) -> list[BranchResult]:
        if not actions:
            return []
        self._round += 1
        round_id = f"round-{self._round}"
        round_started_ns = monotonic_ns()
        branch_ids = [action.branch_id or f"branch-{index}" for index, action in enumerate(actions)]
        if len(set(branch_ids)) != len(branch_ids):
            raise ValueError("branch_id values must be unique within a speculative round.")
        effects = [self.manifest.effect(action.app_name, action.api_name) for action in actions]
        canonical_actions = [
            CanonicalToolCall.from_action(action, self.schemas) for action in actions
        ]
        if len(set(canonical_actions)) != len(canonical_actions):
            raise ValueError("Speculative actions must be distinct after canonicalization.")
        parent_apps = sorted({app for effect in effects for app in effect.write_dbs})
        prefix = round_id
        parent_started_ns = monotonic_ns()
        parent = self.store.capture(self.world, f"{prefix}-parent", parent_apps)
        if self.trace:
            self.trace.record(
                "parent_snapshot_created",
                round_id=round_id,
                snapshot_id=parent.snapshot_id,
                app_names=list(parent.app_names),
                duration_ms=elapsed_ms(parent_started_ns),
            )
        results: list[BranchResult] = []
        try:
            for action, canonical_action, branch_id, effect in zip(
                actions, canonical_actions, branch_ids, effects, strict=True
            ):
                branch_started_ns = monotonic_ns()
                parent.restore(self.world)
                if self.trace:
                    self.trace.record(
                        "branch_started",
                        round_id=round_id,
                        branch_id=branch_id,
                        action=canonical_action.key,
                        read_dbs=list(effect.read_dbs),
                        write_dbs=list(effect.write_dbs),
                    )
                observation: dict[str, Any] | list[Any] | None = None
                status_code: int | None = None
                error: str | None = None
                try:
                    arguments = copy.deepcopy(action.arguments)
                    if working_memory is not None:
                        arguments = working_memory.resolve_credentials(arguments)
                    response = self.world.requester._request(
                        _app_name=action.app_name,
                        _api_name=action.api_name,
                        raise_on_failure=False,
                        track=False,
                        **arguments,
                    )
                    status_code = response.status_code
                    observation = self.world.requester.response_to_json(response)
                except Exception as exception:
                    error = f"{type(exception).__name__}: {exception}"
                snapshot = self.store.capture(
                    self.world, f"{prefix}-{branch_id}", list(effect.write_dbs)
                )
                result = BranchResult(
                    action,
                    canonical_action,
                    effect,
                    observation,
                    status_code,
                    error,
                    parent,
                    snapshot,
                )
                self.branches[branch_id] = result
                results.append(result)
                if self.trace:
                    self.trace.record(
                        "branch_completed",
                        round_id=round_id,
                        branch_id=branch_id,
                        action=canonical_action.key,
                        succeeded=result.succeeded,
                        status_code=status_code,
                        error=error,
                        observation=observation,
                        snapshot_id=snapshot.snapshot_id,
                        snapshot_apps=list(snapshot.app_names),
                        duration_ms=elapsed_ms(branch_started_ns),
                    )
        finally:
            parent.restore(self.world)
            if self.trace:
                self.trace.record(
                    "speculation_completed",
                    round_id=round_id,
                    branch_count=len(results),
                    duration_ms=elapsed_ms(round_started_ns),
                )
        return results

    def promote(self, branch_id: str) -> BranchResult:
        if branch_id not in self.branches:
            raise KeyError(f"Unknown speculative branch {branch_id!r}.")
        branch = self.branches[branch_id]
        branch.parent_snapshot.restore(self.world)
        branch.snapshot.restore(self.world)
        if self.trace:
            self.trace.record(
                "branch_promoted",
                branch_id=branch_id,
                action=branch.canonical_action.key,
                parent_snapshot_id=branch.parent_snapshot.snapshot_id,
                snapshot_id=branch.snapshot.snapshot_id,
            )
        return branch
