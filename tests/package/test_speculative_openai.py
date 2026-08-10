from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

from appworld.speculative import ActorContext
from appworld.speculative_openai import (
    APIPredictionOutput,
    CandidateBatchOutput,
    ManifestToolSelector,
    OpenAISpeculativeActors,
    OpenAISpeculativeConfig,
    SpeculativeModelOutputError,
    ToolCallOutput,
)
from appworld.speculative_trace import TraceRecorder


class FakeResponses:
    def __init__(self, outputs: list[Any]) -> None:
        self.outputs = outputs
        self.calls: list[dict[str, Any]] = []

    async def parse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        await asyncio.sleep(0)
        output_type = kwargs["text_format"]
        for index, output in enumerate(self.outputs):
            if isinstance(output, output_type):
                return SimpleNamespace(output_parsed=self.outputs.pop(index))
        raise AssertionError(f"No fake output for {output_type}")


class SlowResponses:
    async def parse(self, **kwargs: Any) -> Any:
        await asyncio.sleep(1)
        raise AssertionError("unreachable")


def _context() -> ActorContext:
    return ActorContext(
        task_instruction="Inspect my profile",
        available_tools=("supervisor.show_profile", "supervisor.complete_task"),
        environment_state_id="S0",
    )


def test_terra_and_luna_start_together_without_unsupported_sampling_settings() -> None:
    responses = FakeResponses(
        [
            APIPredictionOutput(
                api_names=["supervisor.show_profile", "supervisor.complete_task"]
            ),
            ToolCallOutput(
                app_name="supervisor", api_name="show_profile", arguments_json="{}"
            ),
            CandidateBatchOutput(
                candidates=[
                    ToolCallOutput(
                        app_name="supervisor", api_name="show_profile", arguments_json="{}"
                    ),
                    ToolCallOutput(
                        app_name="supervisor",
                        api_name="complete_task",
                        arguments_json='{"answer":"done"}',
                    ),
                ]
            ),
        ]
    )
    actors = OpenAISpeculativeActors(
        client=SimpleNamespace(responses=responses),
        config=OpenAISpeculativeConfig(k=2),
    )

    actor_action, candidates = asyncio.run(actors.start_round(_context()))

    assert actor_action.api_name == "show_profile"
    assert [candidate.branch_id for candidate in candidates] == ["candidate-0", "candidate-1"]
    assert {call["model"] for call in responses.calls} == {
        "gpt-5.6-terra",
        "gpt-5.6-luna",
    }
    assert len(responses.calls) == 3
    assert all("temperature" not in call for call in responses.calls)
    assert all("reasoning" in call for call in responses.calls)


def test_luna_rejects_duplicate_canonical_candidates() -> None:
    duplicate = ToolCallOutput(
        app_name="supervisor", api_name="show_profile", arguments_json="{}"
    )
    responses = FakeResponses([CandidateBatchOutput(candidates=[duplicate, duplicate])])
    actors = OpenAISpeculativeActors(
        client=SimpleNamespace(responses=responses),
        config=OpenAISpeculativeConfig(k=2),
    )

    with pytest.raises(SpeculativeModelOutputError, match="duplicate canonical"):
        asyncio.run(actors.propose_candidates(_context()))


def test_api_predictor_replaces_catalog_with_short_list() -> None:
    responses = FakeResponses(
        [
            APIPredictionOutput(
                api_names=["supervisor.show_profile", "not_a_real.tool"]
            )
        ]
    )
    context = _context()
    context.available_tools = (
        *context.available_tools,
        "supervisor.show_account_passwords",
    )
    actors = OpenAISpeculativeActors(
        client=SimpleNamespace(responses=responses),
        config=OpenAISpeculativeConfig(max_predicted_tools=3),
        show_progress=False,
    )

    selected = asyncio.run(actors.predict_available_tools(context))

    assert selected == (
        "supervisor.complete_task",
        "supervisor.show_account_passwords",
        "supervisor.show_profile",
    )
    assert context.available_tools == selected


def test_manifest_selector_uses_task_app_and_keeps_control_tools() -> None:
    context = ActorContext(
        task_instruction="Remind my roommates about pending Venmo requests",
        environment_state_id="S0",
    )
    selected = ManifestToolSelector().select(context)

    assert "venmo.show_received_payment_requests" in selected
    assert "supervisor.complete_task" in selected
    assert not any(name.startswith("amazon.") for name in selected)
    assert len(selected) < 100


def test_manifest_selector_refuses_all_tools_fallback() -> None:
    context = ActorContext(task_instruction="Do the thing", environment_state_id="S0")

    with pytest.raises(ValueError, match="refusing to send every AppWorld tool"):
        ManifestToolSelector().select(context)


def test_llm_trace_records_duration_usage_and_output(tmp_path) -> None:
    responses = FakeResponses(
        [
            ToolCallOutput(
                app_name="supervisor", api_name="show_profile", arguments_json="{}"
            )
        ]
    )
    responses.parse_usage = None
    trace_path = tmp_path / "run.jsonl"
    actors = OpenAISpeculativeActors(
        client=SimpleNamespace(responses=responses),
        trace=TraceRecorder(trace_path, run_id="test-run"),
    )

    asyncio.run(actors.choose_action(_context()))

    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "llm_call_started",
        "llm_call_completed",
    ]
    assert events[1]["duration_ms"] >= 0
    assert events[1]["output"]["api_name"] == "show_profile"


def test_candidate_schema_is_compatible_with_strict_structured_outputs() -> None:
    schema = CandidateBatchOutput.model_json_schema()

    def assert_closed_objects(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("type") == "object":
                assert value.get("additionalProperties") is False
            for child in value.values():
                assert_closed_objects(child)
        elif isinstance(value, list):
            for child in value:
                assert_closed_objects(child)

    assert_closed_objects(schema)


def test_llm_call_times_out_and_traces_failure(tmp_path) -> None:
    trace_path = tmp_path / "timeout.jsonl"
    actors = OpenAISpeculativeActors(
        client=SimpleNamespace(responses=SlowResponses()),
        config=OpenAISpeculativeConfig(llm_timeout_seconds=0.01),
        trace=TraceRecorder(trace_path, run_id="timeout-run"),
    )

    with pytest.raises(TimeoutError, match="timed out after 0.01 seconds"):
        asyncio.run(actors.choose_action(_context()))

    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert events[-1]["event"] == "llm_call_failed"
    assert events[-1]["timed_out"] is True
