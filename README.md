# Skyhook

Bind SkyRL environments and their reward functions with decorators. One small
module, with SkyRL Gym as its only direct runtime dependency.

## Install

Requires Python 3.10 or newer and
[uv](https://docs.astral.sh/uv/getting-started/installation/).
Run from the project directory:

```sh
uv sync
```

This creates `.venv` and installs Skyhook and its dependencies. Use `uv run`
to run commands in that environment without manually activating it.

The distribution is named `skyhook-rl`; the Python import is `skyhook`.
The package includes type annotations for type checkers.

## Use

```python
import skyrl_gym
from skyrl_gym.envs.base_text_env import BaseTextEnv
from skyhook import env


@env.textarena_wordle
class WordleEnv(BaseTextEnv):
    def __init__(self, target="crane"):
        super().__init__()
        self.target = target

    def step(self, action):
        return {
            "observations": [{"role": "user", "content": action}],
            "reward": env.textarena_wordle.sum_rewards(self, action),
            "done": action == self.target,
            "metadata": {},
            "postprocessed_action": action,
        }


@env.textarena_wordle.reward
def correct_word(environment, action):
    return 1.0 if action == environment.target else 0.0


@env.textarena_wordle.reward
def turn_cost(environment, action):
    return -0.05


game = skyrl_gym.make("textarena_wordle", target="crane")
game.init([{"role": "user", "content": "Guess the word."}])
assert game.step("crane")["reward"] == 0.95
game.close()
```

The class decorator calls
`skyrl_gym.register("textarena_wordle", entry_point=WordleEnv)`. Both decorators return their inputs unchanged.

`sum_rewards(*args, **kwargs)` calls each registered reward once,
in registration order, forwarding the same arguments to every function, then
passes the collected results to Python's built-in `sum()`. No rewards means the
integer `0`. Results must support `0 + value` and addition with the running total.
`Decimal`, `Fraction`, and custom addition-compatible types keep their normal
addition behavior; incompatible combinations raise Python's usual `TypeError`.

There is no float conversion or finiteness check, and floating-point precision
follows the Python version's built-in `sum()`. Your environment's `step()` remains
responsible for producing the finite scalar reward expected by SkyRL. Exceptions
raised by reward functions or addition propagate unchanged. Async functions are
rejected at registration; coroutine results from other callables are closed and
rejected, never awaited.

To call just one reward inside `step`, look it up by its function name:

```python
reward = env.textarena_wordle.rewards["correct_word"](self, action)
```

`rewards` is a read-only, insertion-ordered mapping of names to the original
callables. A single lookup calls only that function; `sum_rewards` still calls
them all. Direct calls return the original result without aggregation.
Missing names raise `KeyError`, and registering a duplicate name in the
same environment raises `ValueError` instead of replacing or double-counting it.

For a custom name or a callable without `__name__`, register it explicitly:

```python
env.textarena_wordle.reward(lambda game, action: 0.1, name="bonus")
```

Rewards can be registered before or after the class, including after environment
construction. Only rewards registered before a call to `sum_rewards` participate.
Use `env["my-env-v0"]` instead of attribute access for names containing punctuation.

Skyhook does not wrap `step`, replace existing rewards, or install another base
class. Call `sum_rewards` where your environment calculates its reward; add it to
an existing base reward explicitly if needed. Duplicate environment IDs raise
SkyRL's registration error rather than replacing another environment.

Registration happens at import time and is process-local. Import the modules
containing your decorated classes **and** rewards in each SkyRL/Ray worker before
calling `skyrl_gym.make`. There is no automatic module discovery or worker sync.

## Run

```sh
uv run python examples/wordle.py
uv run python tests/test_skyhook.py
```

The example is a tiny word-guessing environment, not the TextArena game engine.
The check exercises the real SkyRL registry and `make` API without a test framework.

GitHub Actions runs the check on every push and pull request using Python 3.10
and 3.12. CI tests a non-editable installation in isolated Python mode, so imports
come from the installed package rather than the source checkout:

```sh
uv run --locked --no-editable --reinstall-package skyhook-rl python -I tests/test_skyhook.py
```

`--locked` enforces `uv.lock`; regenerate it with `uv lock` after changing
dependencies. `--reinstall-package` rebuilds Skyhook so source edits are tested
instead of an older cached wheel.
