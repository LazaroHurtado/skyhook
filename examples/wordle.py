"""A tiny word-guessing example, not a TextArena implementation."""

import skyrl_gym
from skyrl_gym.envs.base_text_env import (
    BaseTextEnv,
    BaseTextEnvStepOutput,
    ConversationType,
)

from skyhook import env


@env.textarena_wordle
class WordleEnv(BaseTextEnv):
    def __init__(self, target: str = "crane") -> None:
        super().__init__()
        self.target = target
        self.max_turns = 6
        self.done = False

    def init(self, prompt: ConversationType) -> tuple[ConversationType, dict]:
        self.turns = 0
        self.done = False
        return super().init(prompt)

    def step(self, action: str) -> BaseTextEnvStepOutput:
        if self.done:
            raise RuntimeError("The episode is finished; call init() to start again.")
        self.turns += 1
        won = action == self.target
        self.done = won or self.turns >= self.max_turns

        feedback = "Try again."
        if self.done:
            feedback = "Correct!" if won else "Out of turns."

        return {
            "observations": [
                {"role": "user", "content": feedback}
            ],
            "reward": env.textarena_wordle.sum_rewards(self, action),
            "done": self.done,
            "metadata": {"turns": self.turns},
            "postprocessed_action": action,
        }


@env.textarena_wordle.reward
def correct_word(environment: WordleEnv, action: str) -> float:
    return 1.0 if action == environment.target else 0.0


@env.textarena_wordle.reward
def turn_cost(environment: WordleEnv, action: str) -> float:
    return -0.05


if __name__ == "__main__":
    with skyrl_gym.make("textarena_wordle", target="crane") as game:
        game.init([{"role": "user", "content": "Guess the five-letter word."}])
        print(game.step("slate"))
        print(game.step("crane"))
        print(env.textarena_wordle.rewards["correct_word"](game, "crane"))
