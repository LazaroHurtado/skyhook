"""Run with: uv run python tests/test_skyhook.py."""

from decimal import Decimal
from fractions import Fraction
from functools import partial
from importlib.resources import files
from inspect import CORO_CLOSED, getcoroutinestate
from math import isclose, isnan
from operator import setitem

import skyrl_gym
from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput

from skyhook import env


def check() -> None:
    assert files("skyhook").joinpath("py.typed").is_file()
    binding = env.skyhook_check
    assert binding is env.skyhook_check is env["skyhook_check"]
    assert binding.sum_rewards() == 0
    assert type(binding.sum_rewards()) is int

    calls = []

    @binding.reward
    def correct(environment, action, *, scale=1.0):
        calls.append(("correct", action))
        return scale if action == environment.target else 0.0

    class GuessEnv(BaseTextEnv):
        def __init__(self, target: str) -> None:
            super().__init__()
            self.target = target

        def step(self, action: str) -> BaseTextEnvStepOutput:
            return {
                "observations": [{"role": "user", "content": action}],
                "reward": binding.sum_rewards(self, action, scale=2.0),
                "done": action == self.target,
                "metadata": {"target": self.target},
                "postprocessed_action": action,
            }

    original_step = GuessEnv.step
    assert binding(GuessEnv) is GuessEnv
    assert GuessEnv.step is original_step
    assert skyrl_gym.spec("skyhook_check").entry_point is GuessEnv
    game = skyrl_gym.make("skyhook_check", target="crane")
    assert isinstance(game, GuessEnv)

    @binding.reward
    def penalty(environment, action, *, scale=1.0):
        calls.append(("penalty", action))
        return -0.25

    expected_rewards = {"correct": correct, "penalty": penalty}
    assert binding.rewards == expected_rewards
    assert list(binding.rewards) == ["correct", "penalty"]
    assert binding.rewards["correct"] is correct
    result = game.step("crane")
    assert isclose(result["reward"], 1.75)
    assert result["observations"] == [{"role": "user", "content": "crane"}]
    assert result["done"] is True
    assert result["metadata"] == {"target": "crane"}
    assert result["postprocessed_action"] == "crane"
    assert calls == [("correct", "crane"), ("penalty", "crane")]
    assert game.step("wrong")["reward"] == -0.25

    calls.clear()
    assert binding.rewards["correct"](game, "crane", scale=2.0) == 2.0
    assert calls == [("correct", "crane")]
    assert binding.rewards["penalty"](game, "crane") == -0.25

    @env["skyhook-other-v0"].reward
    def unrelated():
        return 100.0

    assert env["skyhook-other-v0"].sum_rewards() == 100.0
    assert env.skyhook_empty.rewards == {}
    assert env.skyhook_empty.sum_rewards() == 0
    assert binding.rewards == expected_rewards

    env.skyhook_separate.reward(correct)
    assert env.skyhook_separate.rewards == {"correct": correct}
    assert env.skyhook_separate.sum_rewards(game, "crane", scale=3.0) == 3.0

    constant = partial(float, "2.5")
    assert env.skyhook_named.reward(constant, name="bonus") is constant
    assert env.skyhook_named.rewards["bonus"]() == 2.5
    assert env.skyhook_named.sum_rewards() == 2.5

    class CallableReward:
        def __call__(self):
            return 0.5

    callable_reward = CallableReward()
    assert env.skyhook_callable.reward(callable_reward, name="bonus") is callable_reward
    assert env.skyhook_callable.sum_rewards() == 0.5

    @env.skyhook_snapshot.reward
    def register_during_evaluation():
        env.skyhook_snapshot.reward(
            lambda: 2.0, name=f"late_{len(env.skyhook_snapshot.rewards)}"
        )
        return 1.0

    assert env.skyhook_snapshot.sum_rewards() == 1.0
    assert env.skyhook_snapshot.sum_rewards() == 3.0

    precision_values = (1e16, 1.0, -1e16)
    for index, value in enumerate(precision_values):
        env.skyhook_precision.reward(lambda value=value: value, name=f"term_{index}")
    assert env.skyhook_precision.sum_rewards() == sum(precision_values)

    for name, values, expected in (
        ("integer", (10**400, 10**400), 2 * 10**400),
        ("decimal", (Decimal("0.1"), Decimal("0.2")), Decimal("0.3")),
        ("fraction", (Fraction(1, 3), Fraction(1, 6)), Fraction(1, 2)),
        ("complex", (1 + 2j, 3 + 4j), 4 + 6j),
        ("infinite", (float("inf"), 1.0), float("inf")),
    ):
        numeric = env[f"skyhook_{name}"]
        for index, value in enumerate(values):
            numeric.reward(lambda value=value: value, name=f"term_{index}")
        total = numeric.sum_rewards()
        assert total == expected
        assert type(total) is type(expected)

    env.skyhook_nan.reward(lambda: float("nan"))
    assert isnan(env.skyhook_nan.sum_rewards())

    class AddOnly:
        def __init__(self, value):
            self.value = value

        def __add__(self, other):
            if isinstance(other, AddOnly):
                return AddOnly(self.value + other.value)
            return NotImplemented

        def __radd__(self, other):
            return self if other == 0 else NotImplemented

    for index, value in enumerate((AddOnly(2), AddOnly(3))):
        env.skyhook_add_only.reward(lambda value=value: value, name=f"term_{index}")
    total = env.skyhook_add_only.sum_rewards()
    assert isinstance(total, AddOnly)
    assert total.value == 5

    class FloatOnly:
        def __float__(self):
            raise AssertionError("Aggregation must not convert rewards to floats.")

    for name, values in (
        ("string", ("1.0",)),
        ("none", (None,)),
        ("list", ([1],)),
        ("float_only", (FloatOnly(),)),
        ("mixed", (Decimal("0.1"), Fraction(1, 2))),
    ):
        invalid = env[f"skyhook_invalid_{name}"]
        for index, value in enumerate(values):
            invalid.reward(lambda value=value: value, name=f"term_{index}")
        try:
            invalid.sum_rewards()
        except TypeError:
            pass
        else:
            raise AssertionError(f"Incompatible addition must fail for {name}.")

    async def async_reward():
        return 1.0

    async def async_generator():
        yield 1.0

    class AsyncReward:
        async def __call__(self):
            return 1.0

    for function in (async_reward, partial(async_reward), async_generator, AsyncReward()):
        try:
            binding.reward(function, name="async")
        except TypeError as error:
            assert "synchronous" in str(error)
        else:
            raise AssertionError("Async reward registration must fail.")

    coroutine = async_reward()
    env.skyhook_coroutine_result.reward(lambda: coroutine)
    try:
        env.skyhook_coroutine_result.sum_rewards()
    except TypeError as error:
        assert "skyhook_coroutine_result" in str(error)
    else:
        raise AssertionError("Coroutine results must be rejected, not awaited.")
    assert getcoroutinestate(coroutine) == CORO_CLOSED

    failure = RuntimeError("reward failed")

    @env.skyhook_failure.reward
    def broken():
        raise failure

    @env.skyhook_failure.reward
    def not_reached():
        raise AssertionError("A failed reward must stop evaluation immediately.")

    error_cases = [
        (skyrl_gym.error.RegistrationError, lambda: binding(GuessEnv)),
        (TypeError, lambda: env.skyhook_not_env(object)),
        (TypeError, lambda: env.skyhook_not_class(lambda: None)),
        (TypeError, lambda: binding.reward(None)),
        (TypeError, lambda: binding.reward(constant)),
        (TypeError, lambda: binding.reward(correct, name=123)),
        (ValueError, lambda: binding.reward(correct, name="")),
        (ValueError, lambda: binding.reward(correct, name="   ")),
        (ValueError, lambda: binding.reward(correct)),
        (ValueError, lambda: binding.reward(penalty, name="correct")),
        (KeyError, lambda: binding.rewards["missing"]),
        (TypeError, lambda: setitem(binding.rewards, "correct", penalty)),
        (TypeError, lambda: env[123]),
        (ValueError, lambda: env[""]),
        (ValueError, lambda: env["   "]),
        (AttributeError, lambda: getattr(env, "__missing__")),
        (TypeError, lambda: binding.sum_rewards()),
        (RuntimeError, lambda: env.skyhook_failure.sum_rewards()),
        (RuntimeError, lambda: env.skyhook_failure.rewards["broken"]()),
    ]
    for expected, operation in error_cases:
        try:
            operation()
        except expected as error:
            if expected is RuntimeError:
                assert error is failure
        else:
            raise AssertionError(f"Expected {expected.__name__}")

    assert binding.rewards == expected_rewards
    game.close()
    print("Skyhook checks passed.")


if __name__ == "__main__":
    check()
