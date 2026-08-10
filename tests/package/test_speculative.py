from __future__ import annotations

import random
import json

from appworld import AppWorld
from appworld.speculative import (
    ActorContext,
    BranchContinuation,
    CanonicalToolCall,
    CredentialRef,
    Speculator,
    ToolAction,
    ToolEffectManifest,
    WorkingMemory,
)
from appworld.speculative_trace import TraceRecorder


TASK_ID = "4fab96f_3"


def test_manifest_effect_lookup() -> None:
    manifest = ToolEffectManifest()
    read_effect = manifest.effect("supervisor", "show_profile")
    write_effect = manifest.effect("supervisor", "complete_task")
    assert read_effect.read_dbs == ("supervisor",)
    assert read_effect.write_dbs == ()
    assert write_effect.write_dbs == ("supervisor",)


def test_canonical_tool_call_normalizes_defaults_types_and_credentials() -> None:
    implicit_defaults = CanonicalToolCall.from_action(
        ToolAction(
            "venmo",
            "show_received_payment_requests",
            {"access_token": CredentialRef("venmo"), "page_index": "0"},
        )
    )
    explicit_defaults = CanonicalToolCall.from_action(
        ToolAction(
            "venmo",
            "show_received_payment_requests",
            {
                "access_token": CredentialRef("venmo"),
                "query": "",
                "status": None,
                "page_index": 0,
                "page_limit": 5,
            },
        )
    )
    assert implicit_defaults == explicit_defaults
    assert implicit_defaults.arguments["access_token"] == {
        "$credential": "venmo",
        "$field": "access_token",
        "$principal": "current_user",
    }


def test_working_memory_records_and_resolves_login_credentials() -> None:
    with AppWorld(task_id=TASK_ID, experiment_name="test_credential_memory") as world:
        memory = WorkingMemory()
        memory.seed_usernames(["venmo"], world.task.supervisor.email)
        context = ActorContext(
            task_instruction=world.task.instruction,
            working_memory=memory,
            environment_state_id="S0",
        )
        speculator = Speculator(world)

        password_result = speculator.speculate(
            [ToolAction("supervisor", "show_account_passwords", branch_id="passwords")],
            working_memory=memory,
        )[0]
        password_context = BranchContinuation.from_result(
            password_result, context
        ).next_actor_context
        assert "venmo.password" in password_context.working_memory.credentials

        login_result = speculator.speculate(
            [
                ToolAction(
                    "venmo",
                    "login",
                    {
                        "username": CredentialRef("venmo", field="username"),
                        "password": CredentialRef("venmo", field="password"),
                    },
                    "login",
                )
            ],
            working_memory=password_context.working_memory,
        )[0]
        assert login_result.succeeded
        login_context = BranchContinuation.from_result(
            login_result, password_context
        ).next_actor_context
        assert "venmo.access_token" in login_context.working_memory.credentials
        assert login_context.working_memory.resolve_credentials(
            CredentialRef("venmo")
        ).count(".") == 2


def test_actor_context_branches_without_mutating_parent_memory() -> None:
    parent = ActorContext(
        task_instruction="Do the task",
        working_memory=WorkingMemory(credentials={"venmo": CredentialRef("venmo")}),
        environment_state_id="S0",
    )
    action = CanonicalToolCall.from_action(ToolAction("supervisor", "show_profile"))
    child = parent.after_tool(action, {"first_name": "Ada"}, True, "S1")
    assert parent.working_memory.completed_action_keys == []
    assert child.working_memory.completed_action_keys == [action.key]
    assert child.context_id != parent.context_id


def test_speculation_rolls_back_and_promotion_applies_branch() -> None:
    with AppWorld(task_id=TASK_ID, experiment_name="test_speculator") as world:
        speculator = Speculator(world)
        random_state = random.getstate()
        results = speculator.speculate(
            [
                ToolAction(
                    "supervisor", "show_profile", branch_id="read-profile"
                ),
                ToolAction(
                    "supervisor", "complete_task", branch_id="complete"
                ),
            ]
        )

        assert len(results) == 2
        assert results[0].succeeded
        assert results[0].status_code == 200
        assert results[0].effect.write_dbs == ()
        assert results[1].succeeded
        assert results[1].status_code == 200
        assert results[1].effect.write_dbs == ("supervisor",)
        assert not world.task_completed()
        assert random.getstate() == random_state

        parent_context = ActorContext(
            task_instruction=world.task.instruction,
            environment_state_id="parent",
        )
        continuation = BranchContinuation.from_result(
            results[1], parent_context, branch_id="complete"
        )
        assert continuation.is_hit(results[1].canonical_action)
        assert continuation.is_fresh_for(parent_context)
        continuation.set_next_action(results[0].canonical_action)
        assert continuation.precomputed_next_action == results[0].canonical_action

        # Promotion is based on the captured parent, not whatever state happens
        # to be live when promote() is called.
        world.requester.request(
            "supervisor", "complete_task", answer="unrelated", track=False
        )
        assert world.task_completed()
        promoted = speculator.promote("complete")
        assert promoted.action.api_name == "complete_task"
        assert world.task_completed()


def test_speculation_trace_records_snapshots_branches_and_promotion(tmp_path) -> None:
    trace_path = tmp_path / "speculation.jsonl"
    with AppWorld(task_id=TASK_ID, experiment_name="test_speculator_trace") as world:
        speculator = Speculator(
            world,
            trace=TraceRecorder(trace_path, run_id="test-run"),
        )
        speculator.speculate(
            [ToolAction("supervisor", "show_profile", branch_id="profile")]
        )
        speculator.promote("profile")

    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    names = [event["event"] for event in events]
    assert names == [
        "parent_snapshot_created",
        "branch_started",
        "branch_completed",
        "speculation_completed",
        "branch_promoted",
    ]
    assert events[2]["snapshot_apps"] == []
    assert events[2]["duration_ms"] >= 0
