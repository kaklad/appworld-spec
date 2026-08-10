"""OpenAI Responses API clients for speculative AppWorld actors.

Luna proposes K distinct tool calls in one request. Terra chooses the canonical
actor action and can precompute the next action for every speculative branch.
The module accepts an injected client so tests need no API key or network.
"""

from __future__ import annotations

import asyncio
import copy
import json
import re
import threading
from dataclasses import dataclass
from typing import Any, Awaitable, Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, field_validator
from rich.console import Console, Group
from rich.json import JSON
from rich.text import Text

from appworld.speculative import (
    ActorContext,
    BranchContinuation,
    BranchResult,
    CanonicalToolCall,
    CredentialRef,
    ToolAction,
    ToolEffectManifest,
    ToolSchemaRegistry,
)
from appworld.speculative_trace import TraceRecorder, elapsed_ms, monotonic_ns, response_usage


ParsedModel = TypeVar("ParsedModel", bound=BaseModel)
AwaitedResult = TypeVar("AwaitedResult")
ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh", "max"]


class _ResponsesAPI(Protocol):
    async def parse(self, **kwargs: Any) -> Any: ...


class _OpenAIClient(Protocol):
    responses: _ResponsesAPI


async def _await_with_real_timeout(
    awaitable: Awaitable[AwaitedResult], timeout_seconds: float
) -> AwaitedResult:
    """Timeout an awaitable without using AppWorld's frozen event-loop clock."""

    loop = asyncio.get_running_loop()
    task = asyncio.ensure_future(awaitable)
    timed_out = threading.Event()

    def cancel_task() -> None:
        timed_out.set()
        loop.call_soon_threadsafe(task.cancel)

    timer = threading.Timer(timeout_seconds, cancel_task)
    timer.daemon = True
    timer.start()
    try:
        return await task
    except asyncio.CancelledError as error:
        if timed_out.is_set():
            raise TimeoutError from error
        raise
    finally:
        timer.cancel()


class ToolCallOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    app_name: str
    api_name: str
    arguments_json: str

    @field_validator("arguments_json")
    @classmethod
    def validate_arguments_json(cls, value: str) -> str:
        try:
            arguments = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("arguments_json must contain valid JSON.") from error
        if not isinstance(arguments, dict):
            raise ValueError("arguments_json must encode a JSON object.")
        return value

    @property
    def arguments(self) -> dict[str, Any]:
        return dict(json.loads(self.arguments_json))


class CandidateBatchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[ToolCallOutput]


class APIPredictionOutput(BaseModel):
    """Short-list of tools selected from names and descriptions only."""

    model_config = ConfigDict(extra="forbid")

    api_names: list[str]


@dataclass(frozen=True)
class OpenAISpeculativeConfig:
    actor_model: str = "gpt-5.6-terra"
    speculator_model: str = "gpt-5.6-luna"
    predictor_model: str = "gpt-5.6-luna"
    k: int = 3
    actor_reasoning_effort: ReasoningEffort = "low"
    speculator_reasoning_effort: ReasoningEffort = "none"
    predictor_reasoning_effort: ReasoningEffort = "none"
    max_predicted_tools: int = 20
    max_output_tokens: int = 4096
    llm_timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be at least 1.")
        if self.max_predicted_tools < 1:
            raise ValueError("max_predicted_tools must be at least 1.")
        if self.llm_timeout_seconds <= 0:
            raise ValueError("llm_timeout_seconds must be positive.")


class SpeculativeModelOutputError(RuntimeError):
    """The model returned a valid schema but an invalid speculative decision."""


class ManifestToolSelector:
    """Select tools whose manifest effects intersect the context's relevant apps.

    Explicit seed apps are the safest input. When omitted, app names are inferred
    conservatively from the task, working memory, and recent actor events.
    """

    def __init__(
        self,
        manifest: ToolEffectManifest | None = None,
        schemas: ToolSchemaRegistry | None = None,
        control_apps: tuple[str, ...] = ("supervisor",),
    ) -> None:
        self.manifest = manifest or ToolEffectManifest()
        self.schemas = schemas or ToolSchemaRegistry()
        self.control_apps = control_apps

    def infer_seed_apps(self, context: ActorContext) -> set[str]:
        searchable = " ".join(
            [
                context.task_instruction,
                *context.working_memory.credentials.keys(),
                *context.working_memory.entities.keys(),
                *context.working_memory.constraints.keys(),
                *(event.action.app_name for event in context.recent_events),
            ]
        ).lower()
        app_names = {doc["app_name"] for doc in self.schemas.tools.values()}
        app_names -= {"admin", "api_docs", *self.control_apps}
        return {
            app_name
            for app_name in app_names
            if re.search(
                rf"(?<![a-z0-9]){re.escape(app_name.replace('_', ' '))}(?![a-z0-9])",
                searchable,
            )
            or re.search(rf"(?<![a-z0-9]){re.escape(app_name)}(?![a-z0-9])", searchable)
        }

    def select(
        self,
        context: ActorContext,
        seed_apps: set[str] | None = None,
    ) -> tuple[str, ...]:
        """Return schema-known tools connected to seed apps by manifest DB effects."""

        seeds = set(seed_apps) if seed_apps is not None else self.infer_seed_apps(context)
        unknown = seeds - {doc["app_name"] for doc in self.schemas.tools.values()}
        if unknown:
            raise ValueError(f"Unknown seed apps for tool selection: {sorted(unknown)}")
        if not seeds:
            raise ValueError(
                "Could not infer a relevant app for this task. Pass seed_apps explicitly; "
                "refusing to send every AppWorld tool."
            )

        # Use only direct intersections. Transitive expansion through shared DBs
        # such as admin would reconnect otherwise unrelated apps.
        selected: set[str] = set()
        for tool_name in self.schemas.tools:
            if tool_name not in self.manifest.tools:
                continue
            app_name = tool_name.split(".", 1)[0]
            effect = self.manifest.effect(*tool_name.split(".", 1))
            tool_dbs = set(effect.read_dbs) | set(effect.write_dbs)
            if app_name in seeds or tool_dbs & seeds:
                selected.add(tool_name)

        selected.update(
            name
            for name, doc in self.schemas.tools.items()
            if doc["app_name"] in self.control_apps
        )
        return tuple(sorted(selected))

    def populate(
        self,
        context: ActorContext,
        seed_apps: set[str] | None = None,
    ) -> ActorContext:
        """Clone a context and fill its available tools without mutating the caller."""

        populated = copy.deepcopy(context)
        populated.available_tools = self.select(context, seed_apps)
        return populated


def create_async_openai_client() -> _OpenAIClient:
    """Create the default SDK client, reading ``OPENAI_API_KEY`` as usual."""

    try:
        from openai import AsyncOpenAI
    except ImportError as error:  # pragma: no cover - optional package
        raise RuntimeError(
            "Install the OpenAI integration with `pip install -e '.[speculative]'`."
        ) from error
    return AsyncOpenAI()


def _decode_credential_refs(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode_credential_refs(item) for item in value]
    if isinstance(value, dict):
        if "$credential" in value:
            extra = set(value) - {"$credential", "$principal"}
            if extra:
                raise SpeculativeModelOutputError(
                    f"Credential reference contains unexpected fields: {sorted(extra)}"
                )
            return CredentialRef(
                app_name=str(value["$credential"]),
                principal=str(value.get("$principal", "current_user")),
            )
        return {key: _decode_credential_refs(item) for key, item in value.items()}
    return value


class OpenAISpeculativeActors:
    """Luna speculator and Terra actor backed by the Responses API."""

    def __init__(
        self,
        client: _OpenAIClient | None = None,
        config: OpenAISpeculativeConfig | None = None,
        schemas: ToolSchemaRegistry | None = None,
        tool_selector: ManifestToolSelector | None = None,
        trace: TraceRecorder | None = None,
        show_progress: bool = True,
    ) -> None:
        self.client = client or create_async_openai_client()
        self.config = config or OpenAISpeculativeConfig()
        self.schemas = schemas or ToolSchemaRegistry()
        self.tool_selector = tool_selector or ManifestToolSelector(schemas=self.schemas)
        self.trace = trace
        self.show_progress = show_progress
        self.console = Console()

    def _progress_start(self, call_name: str, model: str, details: str) -> None:
        if not self.show_progress:
            return
        line = Text()
        line.append("▶ ", style="bold cyan")
        line.append(call_name, style="bold")
        line.append(f"  {model}  {details}", style="dim")
        self.console.print(line)

    def _progress_failed(self, call_name: str, error: str) -> None:
        if not self.show_progress:
            return
        line = Text()
        line.append("✗ ", style="bold red")
        line.append(call_name, style="bold")
        line.append(f"  {error}", style="red")
        self.console.print(line)

    def _progress_done(
        self,
        call_name: str,
        model: str,
        duration_ms: float,
        output: dict[str, Any],
        usage: dict[str, Any] | None = None,
    ) -> None:
        if not self.show_progress:
            return
        line = Text()
        line.append("✓ ", style="bold green")
        line.append(call_name, style="bold")
        line.append(f"  {model}  {duration_ms / 1000:.2f}s", style="dim")
        if usage:
            line.append(
                f"  tokens={usage.get('input_tokens', '?')}→{usage.get('output_tokens', '?')}",
                style="dim",
            )
        self.console.print(Group(line, JSON.from_data(output)))

    def _tool_documents(self, context: ActorContext) -> list[dict[str, Any]]:
        names = context.available_tools or self.tool_selector.select(context)
        missing = set(names) - set(self.schemas.tools)
        if missing:
            raise KeyError(f"ActorContext references unknown tools: {sorted(missing)}")
        return [self.schemas.tools[name] for name in names]

    def _input(self, context: ActorContext, instruction: str) -> list[dict[str, str]]:
        payload = {
            "actor_context": context.to_prompt_dict(),
            "tool_documents": self._tool_documents(context),
        }
        return [
            {
                "role": "developer",
                "content": (
                    "You control an AppWorld environment. Choose only from the supplied "
                    "tool documents and obey their parameter schemas. Credentials must be "
                    "obtained through tools; never invent usernames, passwords, or tokens. "
                    "If working_memory.credentials is empty, call "
                    "supervisor.show_account_passwords before an app login. A logical "
                    "reference such as {\"$credential\": \"venmo\", "
                    "\"$principal\": \"current_user\"} may be used only for an "
                    "access_token-like parameter when that credential already exists in "
                    "working_memory.credentials. Never add fields to a credential reference. "
                    "Put all tool parameters in arguments_json as a JSON object string; use "
                    "\"{}\" when the tool has no parameters. "
                    "supervisor.complete_task is the normal way to finish. " + instruction
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
            },
        ]

    async def predict_available_tools(self, context: ActorContext) -> tuple[str, ...]:
        """Run the benchmark-style API predictor before sending full schemas."""

        candidate_names = context.available_tools or tuple(sorted(self.schemas.tools))
        missing = set(candidate_names) - set(self.schemas.tools)
        if missing:
            raise KeyError(f"API predictor references unknown tools: {sorted(missing)}")
        descriptions = [
            {
                "api_name": name,
                "description": self.schemas.tools[name].get("description", ""),
            }
            for name in candidate_names
        ]
        request_input = [
            {
                "role": "developer",
                "content": (
                    "Select the APIs that may be needed to complete the task. Prioritize "
                    "high recall, but exclude clearly irrelevant APIs. Return at most "
                    f"{self.config.max_predicted_tools} exact api_names from the supplied "
                    "catalog. supervisor.complete_task must always be included."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"task": context.task_instruction, "api_catalog": descriptions},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        ]
        started_ns = monotonic_ns()
        call_name = "api-predictor"
        model = self.config.predictor_model
        self._progress_start(
            call_name,
            model,
            f"catalog={len(descriptions)} timeout={self.config.llm_timeout_seconds:g}s",
        )
        if self.trace:
            self.trace.record(
                "llm_call_started",
                model=model,
                context_id=context.context_id,
                output_type=APIPredictionOutput.__name__,
                call_name=call_name,
                catalog_count=len(descriptions),
                input_characters=sum(len(item["content"]) for item in request_input),
            )
        try:
            response = await _await_with_real_timeout(
                self.client.responses.parse(
                    model=model,
                    input=request_input,
                    text_format=APIPredictionOutput,
                    reasoning={"effort": self.config.predictor_reasoning_effort},
                    max_output_tokens=self.config.max_output_tokens,
                ),
                self.config.llm_timeout_seconds,
            )
        except Exception as error:
            timed_out = isinstance(error, TimeoutError)
            error_text = (
                f"OpenAI {model} call timed out after "
                f"{self.config.llm_timeout_seconds:g} seconds."
                if timed_out
                else f"{type(error).__name__}: {error}"
            )
            if self.trace:
                self.trace.record(
                    "llm_call_failed",
                    model=model,
                    context_id=context.context_id,
                    call_name=call_name,
                    duration_ms=elapsed_ms(started_ns),
                    error=error_text,
                    timed_out=timed_out,
                )
            self._progress_failed(call_name, error_text)
            if timed_out:
                raise TimeoutError(error_text) from error
            raise

        parsed = response.output_parsed
        if not isinstance(parsed, APIPredictionOutput):
            parsed = APIPredictionOutput.model_validate(parsed)
        allowed = set(candidate_names)
        predicted = []
        for name in parsed.api_names:
            if name in allowed and name not in predicted:
                predicted.append(name)
        required_control_tools = (
            "supervisor.complete_task",
            "supervisor.show_account_passwords",
        )
        for required_tool in reversed(required_control_tools):
            if required_tool in allowed and required_tool not in predicted:
                predicted.insert(0, required_tool)
        predicted = predicted[: self.config.max_predicted_tools]
        if not predicted:
            raise SpeculativeModelOutputError("API predictor returned no valid tool names.")
        context.available_tools = tuple(predicted)
        duration = elapsed_ms(started_ns)
        if self.trace:
            self.trace.record(
                "api_tools_predicted",
                model=model,
                context_id=context.context_id,
                duration_ms=duration,
                usage=response_usage(response),
                candidate_count=len(candidate_names),
                selected_tools=predicted,
            )
        self._progress_done(
            call_name,
            model,
            duration,
            {"selected_tools": predicted},
            response_usage(response),
        )
        return context.available_tools

    async def _parse(
        self,
        *,
        model: str,
        effort: ReasoningEffort,
        context: ActorContext,
        instruction: str,
        output_type: type[ParsedModel],
        call_name: str,
    ) -> ParsedModel:
        request_input = self._input(context, instruction)
        started_ns = monotonic_ns()
        self._progress_start(
            call_name,
            model,
            f"context={context.context_id[:12]} timeout={self.config.llm_timeout_seconds:g}s",
        )
        if self.trace:
            self.trace.record(
                "llm_call_started",
                model=model,
                context_id=context.context_id,
                environment_state_id=context.environment_state_id,
                output_type=output_type.__name__,
                tool_count=len(self._tool_documents(context)),
                input_characters=sum(len(item["content"]) for item in request_input),
            )
        try:
            response = await _await_with_real_timeout(
                self.client.responses.parse(
                    model=model,
                    input=request_input,
                    text_format=output_type,
                    reasoning={"effort": effort},
                    max_output_tokens=self.config.max_output_tokens,
                ),
                self.config.llm_timeout_seconds,
            )
        except TimeoutError as error:
            error_text = (
                f"OpenAI {model} call timed out after "
                f"{self.config.llm_timeout_seconds:g} seconds."
            )
            if self.trace:
                self.trace.record(
                    "llm_call_failed",
                    model=model,
                    context_id=context.context_id,
                    duration_ms=elapsed_ms(started_ns),
                    error=error_text,
                    timed_out=True,
                )
            self._progress_failed(call_name, error_text)
            raise TimeoutError(error_text) from error
        except Exception as error:
            if self.trace:
                self.trace.record(
                    "llm_call_failed",
                    model=model,
                    context_id=context.context_id,
                    duration_ms=elapsed_ms(started_ns),
                    error=f"{type(error).__name__}: {error}",
                )
            self._progress_failed(call_name, f"{type(error).__name__}: {error}")
            raise
        parsed = response.output_parsed
        if parsed is None:
            raise SpeculativeModelOutputError("The model did not return parsed structured output.")
        if not isinstance(parsed, output_type):
            parsed = output_type.model_validate(parsed)
        if self.trace:
            self.trace.record(
                "llm_call_completed",
                model=model,
                response_model=getattr(response, "model", None),
                response_id=getattr(response, "id", None),
                context_id=context.context_id,
                duration_ms=elapsed_ms(started_ns),
                usage=response_usage(response),
                output=parsed.model_dump(mode="json"),
            )
        self._progress_done(
            call_name,
            model,
            elapsed_ms(started_ns),
            parsed.model_dump(mode="json"),
            response_usage(response),
        )
        return parsed

    def _to_action(self, output: ToolCallOutput, branch_id: str = "") -> ToolAction:
        action = ToolAction(
            app_name=output.app_name,
            api_name=output.api_name,
            arguments=_decode_credential_refs(output.arguments),
            branch_id=branch_id,
        )
        CanonicalToolCall.from_action(action, self.schemas)
        return action

    async def propose_candidates(
        self, context: ActorContext, k: int | None = None
    ) -> list[ToolAction]:
        """Ask Luna once for exactly K distinct, schema-valid candidate calls."""

        k = self.config.k if k is None else k
        if k < 1:
            raise ValueError("k must be at least 1.")
        output = await self._parse(
            model=self.config.speculator_model,
            effort=self.config.speculator_reasoning_effort,
            context=context,
            instruction=(
                f"Return exactly {k} different likely next tool calls in candidates, "
                "ordered from most to least likely. This is one inference, not K inferences."
            ),
            output_type=CandidateBatchOutput,
            call_name="speculator",
        )
        if len(output.candidates) != k:
            raise SpeculativeModelOutputError(
                f"Expected {k} candidates from Luna, received {len(output.candidates)}."
            )
        actions = [
            self._to_action(candidate, branch_id=f"candidate-{index}")
            for index, candidate in enumerate(output.candidates)
        ]
        canonical = [CanonicalToolCall.from_action(action, self.schemas) for action in actions]
        if len(set(canonical)) != k:
            raise SpeculativeModelOutputError("Luna returned duplicate canonical candidates.")
        if self.trace:
            self.trace.record(
                "candidates_generated",
                context_id=context.context_id,
                candidate_count=k,
                candidates=[item.key for item in canonical],
            )
        return actions

    async def choose_action(
        self, context: ActorContext, call_name: str = "actor"
    ) -> CanonicalToolCall:
        """Ask the deterministic Terra actor for its next canonical action."""

        output = await self._parse(
            model=self.config.actor_model,
            effort=self.config.actor_reasoning_effort,
            context=context,
            instruction="Return the single best next tool call.",
            output_type=ToolCallOutput,
            call_name=call_name,
        )
        return CanonicalToolCall.from_action(self._to_action(output), self.schemas)

    async def precompute_continuations(
        self,
        parent_context: ActorContext,
        branch_results: list[BranchResult],
    ) -> list[BranchContinuation]:
        """Run one Terra call per speculative observation, concurrently."""

        continuations = [
            BranchContinuation.from_result(result, parent_context) for result in branch_results
        ]
        next_actions = await asyncio.gather(
            *(
                self.choose_action(
                    item.next_actor_context,
                    call_name=f"actor-continuation[{item.branch_id}]",
                )
                for item in continuations
            )
        )
        for continuation, next_action in zip(continuations, next_actions, strict=True):
            continuation.set_next_action(next_action)
            if self.trace:
                self.trace.record(
                    "continuation_ready",
                    branch_id=continuation.branch_id,
                    parent_context_id=continuation.parent_context_id,
                    next_context_id=continuation.next_actor_context.context_id,
                    next_action=next_action.key,
                )
        return continuations

    async def start_round(
        self, context: ActorContext, k: int | None = None
    ) -> tuple[CanonicalToolCall, list[ToolAction]]:
        """Start the current Terra actor and Luna candidate request together."""

        await self.predict_available_tools(context)
        actor_call = asyncio.create_task(self.choose_action(context))
        candidate_call = asyncio.create_task(self.propose_candidates(context, k))
        actor_action, candidates = await asyncio.gather(actor_call, candidate_call)
        if self.trace:
            self.trace.record(
                "round_llm_ready",
                context_id=context.context_id,
                actor_action=actor_action.key,
                candidate_count=len(candidates),
            )
        return actor_action, candidates

    def find_hit(
        self,
        actor_action: CanonicalToolCall,
        continuations: list[BranchContinuation],
        context: ActorContext,
    ) -> BranchContinuation | None:
        """Return and trace the fresh branch matching the authoritative action."""

        hit = next(
            (
                item
                for item in continuations
                if item.is_fresh_for(context) and item.is_hit(actor_action)
            ),
            None,
        )
        if self.trace:
            self.trace.record(
                "hit_checked",
                context_id=context.context_id,
                actor_action=actor_action.key,
                hit=hit is not None,
                branch_id=hit.branch_id if hit else None,
                candidate_count=len(continuations),
            )
        return hit
