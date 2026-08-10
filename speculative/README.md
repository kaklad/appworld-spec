# Manifest-guided speculative execution

`appworld.speculative` provides a correctness-first prototype for branching at
the API-tool-call boundary. It uses `tool_effects.json` to snapshot only the
databases in a tool's effective write set.

```python
from appworld import AppWorld
from appworld.speculative import Speculator, ToolAction

with AppWorld(task_id=task_id, experiment_name="speculative-demo") as world:
    speculator = Speculator(world)
    branches = speculator.speculate(
        [
            ToolAction("venmo", "approve_payment_request", {"payment_request_id": 41}, "b0"),
            ToolAction("venmo", "deny_payment_request", {"payment_request_id": 41}, "b1"),
        ]
    )
    chosen = speculator.promote("b0")
```

For each speculative round, the implementation:

1. Captures one parent snapshot for the union of branch write sets.
2. Restores the parent before evaluating each tool action.
3. Executes the tool without adding it to the main request log.
4. Captures only that branch's write-set databases and runtime state.
5. Restores the parent after all branches have been evaluated.
6. Promotes a hit by applying its parent and branch snapshots atomically.

Runtime snapshots include Python RNG state, request-tracker state,
authentication blacklists, model hashes, and per-app SQL change trackers.
SQLite snapshots are full copies of only the selected app databases.

The current implementation evaluates branches sequentially in one process. It
is intended to validate snapshot and promotion semantics first. A parallel
version should run each branch in an isolated worker process while retaining
the same `ToolAction`, `BranchResult`, and per-app snapshot boundaries.

## Actor state and hit matching

The actor-facing types deliberately keep raw database state out of the model
prompt:

- `WorkingMemory` stores compact, observation-derived facts, plan progress,
  pagination state, and logical credential handles.
- `ActorContext` combines the task, working memory, a short recent-event
  window, relevant tools, and opaque environment/context IDs.
- `CanonicalToolCall` fills API defaults, normalizes schema types and JSON key
  order, and provides the exact identity used for hit matching.
- `BranchContinuation` holds one branch observation, its derived actor context,
  and the next action produced by the structured-output large actor.

```python
from appworld.speculative import (
    ActorContext,
    BranchContinuation,
    CanonicalToolCall,
    CredentialRef,
    ToolAction,
    WorkingMemory,
)

memory = WorkingMemory(credentials={"venmo": CredentialRef("venmo")})
context = ActorContext(
    task_instruction=world.task.instruction,
    working_memory=memory,
    available_tools=("venmo.show_received_payment_requests",),
    environment_state_id="S0",
)

actor_action = CanonicalToolCall.from_action(
    ToolAction(
        "venmo",
        "show_received_payment_requests",
        {"access_token": CredentialRef("venmo")},
    )
)

continuation = BranchContinuation.from_result(branch_result, context)
if continuation.is_fresh_for(context) and continuation.is_hit(actor_action):
    speculator.promote(continuation.branch_id)
```

`CredentialRef` is intentionally a logical handle. The prompt and canonical
action should not contain volatile JWT strings. A later executor layer must
resolve these handles to the actual branch-local token immediately before the
tool is called.

## OpenAI actor calls

`appworld.speculative_openai.OpenAISpeculativeActors` uses the Responses API
with structured outputs. A benchmark-style API predictor first sees only tool
names and short descriptions and selects at most 20 tools. Only those tools'
full schemas are sent to the actor and speculator. GPT-5.6 Luna then produces
all K candidates in one call;
GPT-5.6 Terra produces the authoritative current action. The two requests are
started together. These models reject the Responses API `temperature` field,
so determinism relies on explicit reasoning settings, structured output, and
canonical action matching rather than sampling parameters. After branch
execution, the K Terra continuation requests can also be issued concurrently.

Install the optional SDK dependency and set the API key:

```bash
pip install -e '.[speculative]'
export OPENAI_API_KEY='...'
```

```python
import asyncio

from appworld.speculative_openai import OpenAISpeculativeActors

actors = OpenAISpeculativeActors()
actor_action, candidates = asyncio.run(actors.start_round(context, k=3))
branch_results = speculator.speculate(candidates)
continuations = asyncio.run(
    actors.precompute_continuations(context, branch_results)
)
```

Tests inject a fake Responses client and therefore do not need an API key or
network access.

Run one live speculative round with:

```bash
python scripts/run_speculative_demo.py \
  --task-id 4fab96f_3 \
  -k 3 \
  --llm-timeout-seconds 300 \
  --max-predicted-tools 20 \
  --log-name venmo-demo \
  --actor-reasoning-effort low \
  --speculator-reasoning-effort none \
  --predictor-reasoning-effort none
```

Terminal output and exceptions are mirrored to `log/venmo-demo.log`. ANSI
color codes are removed from the file; the structured JSONL trace remains a
separate artifact for programmatic analysis.

The runner suppresses Python warnings by default. Exceptions and trace failure
events are still reported. It temporarily releases AppWorld's frozen task clock
during OpenAI network calls, then restores it before executing AppWorld APIs;
this prevents frozen `asyncio`/HTTP client timers from hanging.

When `ActorContext.available_tools` is empty, the OpenAI actor uses
`ManifestToolSelector`. It infers seed apps from the task, working memory, and
recent actions, then includes only tools whose manifest read/write databases
intersect those apps. Supervisor control tools are retained. If no seed app can
be inferred, it raises instead of silently sending all 473 tools. Ambiguous
tasks should provide explicit seeds:

```python
selector = ManifestToolSelector()
context = selector.populate(context, seed_apps={"venmo"})
```

## Execution trace

Use one `TraceRecorder` for the LLM and snapshot layers. It writes JSONL events
with UTC timestamps, monotonic durations, model usage, actions, observations,
snapshot DB sets, hit checks, and promotions.

```python
from appworld.speculative_trace import TraceRecorder

trace = TraceRecorder("outputs/speculative/run.jsonl", run_id="task-4fab96f_3")
actors = OpenAISpeculativeActors(trace=trace)
speculator = Speculator(world, trace=trace)
```
