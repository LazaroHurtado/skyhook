"""Bind environments and reward functions to SkyRL with decorators."""

from collections.abc import Callable, Mapping
from inspect import isasyncgenfunction, iscoroutine, iscoroutinefunction
from types import MappingProxyType
from typing import Any, TypeVar

import skyrl_gym

__all__ = ["env"]

EnvType = TypeVar("EnvType", bound=skyrl_gym.Env)
RewardType = TypeVar("RewardType", bound=Callable[..., Any])


def _validate_name(name: object, kind: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"{kind} names must be strings.")
    if not name.strip():
        raise ValueError(f"{kind} names must not be empty or whitespace.")
    return name


class _Environment:
    def __init__(self, name: str) -> None:
        self.name = name
        self._rewards: dict[str, Callable[..., Any]] = {}

    def __call__(self, cls: type[EnvType]) -> type[EnvType]:
        """Register a SkyRL environment class without modifying it."""
        if not isinstance(cls, type) or not issubclass(cls, skyrl_gym.Env):
            raise TypeError("Environment decorators require a skyrl_gym.Env subclass.")
        skyrl_gym.register(self.name, entry_point=cls)
        return cls

    def reward(self, function: RewardType, *, name: str | None = None) -> RewardType:
        """Register a callable by name (default: __name__) and return it unchanged."""
        if not callable(function):
            raise TypeError("Reward decorators require a callable.")
        for target in (function, type(function).__call__):
            if iscoroutinefunction(target) or isasyncgenfunction(target):
                raise TypeError("Reward functions must be synchronous.")

        name = name if name is not None else getattr(function, "__name__", None)
        name = _validate_name(name, "Reward")
        if name in self._rewards:
            raise ValueError(f"Reward {name!r} is already registered for {self.name!r}.")

        self._rewards[name] = function
        return function

    @property
    def rewards(self) -> Mapping[str, Callable[..., Any]]:
        """Read-only name-to-callable mapping, in registration order."""
        return MappingProxyType(self._rewards)

    def sum_rewards(self, *args: Any, **kwargs: Any) -> Any:
        """Sum reward results using Python's addition rules, starting from 0."""
        values = []
        for name, function in tuple(self._rewards.items()):
            value = function(*args, **kwargs)
            if iscoroutine(value):
                value.close()
                raise TypeError(
                    f"Reward {name!r} for {self.name!r} must be synchronous."
                )
            values.append(value)
        return sum(values)


class _Registry:
    def __init__(self) -> None:
        self._environments: dict[str, _Environment] = {}

    def __getitem__(self, name: str) -> _Environment:
        name = _validate_name(name, "Environment")
        if name not in self._environments:
            self._environments[name] = _Environment(name)
        return self._environments[name]

    def __getattr__(self, name: str) -> _Environment:
        if name.startswith("_"):
            raise AttributeError(name)
        return self[name]


env: _Registry = _Registry()
