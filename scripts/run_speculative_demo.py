#!/usr/bin/env python
"""Run one end-to-end speculative AppWorld round with OpenAI models."""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import traceback
import warnings
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TextIO

# AppWorld imports currently emit many framework deprecation warnings. Keep the
# live runner focused on actions, timings, and errors; exceptions remain visible.
warnings.filterwarnings("ignore")

from appworld import AppWorld
from appworld.speculative import ActorContext, Speculator
from appworld.speculative_openai import (
    ManifestToolSelector,
    OpenAISpeculativeActors,
    OpenAISpeculativeConfig,
)
from appworld.speculative_trace import TraceRecorder
from appworld.task import Task


REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class TeeOutput(AbstractContextManager["TeeOutput"]):
    """Mirror stdout and stderr to a plain-text log file."""

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.terminal_stdout = sys.stdout
        self.terminal_stderr = sys.stderr
        self.file: TextIO | None = None

    def write(self, text: str) -> int:
        self.terminal_stdout.write(text)
        if self.file is not None:
            self.file.write(ANSI_ESCAPE.sub("", text))
        return len(text)

    def flush(self) -> None:
        self.terminal_stdout.flush()
        if self.file is not None:
            self.file.flush()

    def isatty(self) -> bool:
        return self.terminal_stdout.isatty()

    def fileno(self) -> int:
        return self.terminal_stdout.fileno()

    @property
    def encoding(self) -> str:
        return self.terminal_stdout.encoding or "utf-8"

    def __getattr__(self, name: str):
        # Libraries such as prompt_toolkit inspect terminal-specific attributes
        # beyond the standard TextIO methods.
        return getattr(self.terminal_stdout, name)

    def __enter__(self) -> "TeeOutput":
        self.file = self.file_path.open("w", encoding="utf-8", buffering=1)
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        sys.stdout = self.terminal_stdout
        sys.stderr = self.terminal_stderr
        if self.file is not None:
            self.file.close()


def log_path(name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("log name may contain only letters, numbers, '.', '_' and '-'.")
    file_name = name if name.endswith(".log") else f"{name}.log"
    return Path("log") / file_name


async def run_llm_with_real_clock(world: AppWorld, awaitable):
    """Temporarily unfreeze AppWorld time for async network clients."""

    world.time_freezer.stop()
    try:
        return await awaitable
    finally:
        # Restore the task's deterministic datetime before any AppWorld API call.
        world.time_freezer.start()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", default="4fab96f_3")
    parser.add_argument("-k", type=int, default=3)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument(
        "--seed-app",
        action="append",
        default=[],
        help="Optionally restrict the API predictor catalog; repeat for multiple apps.",
    )
    parser.add_argument("--experiment-name", default="speculative-live")
    parser.add_argument("--trace", type=Path, default=None)
    parser.add_argument(
        "--log-name",
        default=None,
        help="Mirror terminal output to log/{name}.log (default: task ID).",
    )
    parser.add_argument("--actor-model", default="gpt-5.6-terra")
    parser.add_argument("--speculator-model", default="gpt-5.6-luna")
    parser.add_argument("--predictor-model", default="gpt-5.6-luna")
    parser.add_argument("--max-predicted-tools", type=int, default=20)
    parser.add_argument(
        "--llm-timeout-seconds",
        type=float,
        default=300.0,
        help="Timeout for each OpenAI request (default: 300 seconds).",
    )
    parser.add_argument(
        "--actor-reasoning-effort",
        choices=REASONING_EFFORTS,
        default="low",
        help="Reasoning effort for the Terra actor.",
    )
    parser.add_argument(
        "--speculator-reasoning-effort",
        choices=REASONING_EFFORTS,
        default="none",
        help="Reasoning effort for the Luna speculator.",
    )
    parser.add_argument(
        "--predictor-reasoning-effort",
        choices=REASONING_EFFORTS,
        default="none",
        help="Reasoning effort for the API predictor.",
    )
    return parser.parse_args()


async def run(args: argparse.Namespace) -> int:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in this shell.")
    if args.k < 1:
        raise ValueError("k must be at least 1.")
    if args.max_rounds < 1:
        raise ValueError("max_rounds must be at least 1.")

    trace_path = args.trace or Path("outputs/speculative") / f"{args.task_id}.jsonl"
    trace = TraceRecorder(trace_path, run_id=args.task_id)

    # Run the initial model calls before AppWorld starts freezegun. Task.load()
    # reads metadata and API docs but does not open the mutable execution world.
    task = Task.load(args.task_id, load_ground_truth=False)
    context = ActorContext(task_instruction=task.instruction, environment_state_id="S0")
    context.working_memory.seed_usernames(task.allowed_apps, task.supervisor.email)
    selector = ManifestToolSelector()
    if args.seed_app:
        context = selector.populate(context, seed_apps=set(args.seed_app))

    actors = OpenAISpeculativeActors(
        config=OpenAISpeculativeConfig(
            actor_model=args.actor_model,
            speculator_model=args.speculator_model,
            predictor_model=args.predictor_model,
            k=args.k,
            actor_reasoning_effort=args.actor_reasoning_effort,
            speculator_reasoning_effort=args.speculator_reasoning_effort,
            predictor_reasoning_effort=args.predictor_reasoning_effort,
            max_predicted_tools=args.max_predicted_tools,
            llm_timeout_seconds=args.llm_timeout_seconds,
        ),
        tool_selector=selector,
        trace=trace,
    )

    print(f"task: {task.instruction}")
    catalog_size = len(context.available_tools) or len(selector.schemas.tools)
    print(f"API predictor catalog: {catalog_size}")
    print(f"trace: {trace_path.resolve()}")
    print(f"starting Terra and Luna (timeout={args.llm_timeout_seconds:g}s each)...")

    actor_action, candidates = await actors.start_round(context, k=args.k)
    print(f"selected tools: {len(context.available_tools)}")
    print(f"actor: {actor_action.key}")
    for candidate in candidates:
        print(f"candidate[{candidate.branch_id}]: {candidate.tool_name}")

    with AppWorld(task_id=args.task_id, experiment_name=args.experiment_name) as world:
        speculator = Speculator(world, trace=trace)
        for round_number in range(1, args.max_rounds + 1):
            print(f"\n{'─' * 20} round {round_number}/{args.max_rounds} {'─' * 20}")
            branch_results = speculator.speculate(
                candidates,
                working_memory=context.working_memory,
            )
            for result in branch_results:
                print(
                    f"branch[{result.action.branch_id}]: status={result.status_code} "
                    f"success={result.succeeded} "
                    f"snapshot_apps={list(result.snapshot.app_names)}"
                )
                if result.error:
                    print(f"  error: {result.error}")

            continuations = await run_llm_with_real_clock(
                world,
                actors.precompute_continuations(context, branch_results),
            )
            hit = actors.find_hit(actor_action, continuations, context)
            if hit is not None:
                speculator.promote(hit.branch_id)
                context = hit.next_actor_context
                next_action = hit.precomputed_next_action
                print(f"result: HIT branch={hit.branch_id}")
            else:
                print("result: MISS; executing authoritative actor action")
                miss_branch_id = f"actor-miss-{round_number}"
                miss_result = speculator.speculate(
                    [actor_action.to_action(miss_branch_id)],
                    working_memory=context.working_memory,
                )[0]
                if not miss_result.succeeded:
                    raise RuntimeError(
                        f"Authoritative action failed: {miss_result.error or miss_result.observation}"
                    )
                speculator.promote(miss_branch_id)
                context = context.after_tool(
                    miss_result.canonical_action,
                    miss_result.observation,
                    True,
                    miss_result.snapshot.snapshot_id,
                )
                next_action = None

            if world.task_completed():
                print(f"task completed successfully in {round_number} rounds")
                return 0

            if next_action is None:
                next_action = await run_llm_with_real_clock(
                    world,
                    actors.choose_action(context, call_name="actor-after-miss"),
                )
            actor_action = next_action
            print(f"next actor action: {actor_action.key}")
            candidates = await run_llm_with_real_clock(
                world,
                actors.propose_candidates(context, args.k),
            )
            for candidate in candidates:
                print(f"candidate[{candidate.branch_id}]: {candidate.tool_name}")

        print(f"task incomplete after max_rounds={args.max_rounds}")
        return 2


def main() -> None:
    args = parse_args()
    output_log_path = log_path(args.log_name or args.task_id)
    with TeeOutput(output_log_path):
        print(f"console log: {output_log_path.resolve()}")
        try:
            exit_code = asyncio.run(run(args))
        except Exception:
            traceback.print_exc()
            exit_code = 1
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
