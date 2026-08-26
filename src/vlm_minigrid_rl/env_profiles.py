"""Per-environment action/prompt/horizon profiles for Empty MiniGrid.

`--env-size 8|16` resolves Empty. Optional `--env-profile` / `--env-id` must
name the same Empty stand.

Step horizons (train != eval):
  train_max_steps = L_max (worst-case BFS-optimal action count).
  eval_max_steps  = L_max + max(4, p95_steps_to_see).
  GRPO pins MiniGrid ``env.unwrapped.max_steps`` to the train horizon so
  reward ``1 - 0.9 * t / max_steps`` prefers shorter wins.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from .training_utils import ACTION_NAMES, ACTION_TO_ID, build_navigation_prompt


@dataclass(frozen=True)
class EnvProfile:
    name: str
    gym_id: str
    env_size: int
    action_names: tuple[str, ...]
    action_to_id: MappingProxyType
    train_max_steps: int
    eval_max_steps: int
    dataset_slug: str
    kind: str
    val_split: float

    @property
    def max_steps(self) -> int:
        """Eval horizon (test_models / env-eval default)."""
        return self.eval_max_steps

    @property
    def id_to_action(self) -> dict[int, str]:
        return {int(idx): name for name, idx in self.action_to_id.items()}

    def prompt(self, goal_color: str = "green") -> str:
        return build_navigation_prompt(goal_color)


EMPTY_8X8 = EnvProfile(
    name="empty-8x8",
    gym_id="MiniGrid-Empty-8x8-v0",
    env_size=8,
    action_names=tuple(ACTION_NAMES),
    action_to_id=MappingProxyType(dict(ACTION_TO_ID)),
    train_max_steps=12,
    eval_max_steps=16,
    dataset_slug="8x8",
    kind="empty",
    val_split=0.1,
)

EMPTY_16X16 = EnvProfile(
    name="empty-16x16",
    gym_id="MiniGrid-Empty-16x16-v0",
    env_size=16,
    action_names=tuple(ACTION_NAMES),
    action_to_id=MappingProxyType(dict(ACTION_TO_ID)),
    train_max_steps=28,
    eval_max_steps=38,
    dataset_slug="16x16",
    kind="empty",
    val_split=0.01,
)

PROFILES: dict[str, EnvProfile] = {
    EMPTY_8X8.name: EMPTY_8X8,
    EMPTY_16X16.name: EMPTY_16X16,
}

_GYM_ID_TO_PROFILE = {profile.gym_id: profile for profile in PROFILES.values()}
_ENV_SIZE_TO_EMPTY = {8: EMPTY_8X8, 16: EMPTY_16X16}


def profile_for_env_size(env_size: int) -> EnvProfile:
    try:
        return _ENV_SIZE_TO_EMPTY[int(env_size)]
    except KeyError as exc:
        raise ValueError(f"No Empty profile for env_size={env_size!r}") from exc


def resolve_profile(
    env_size: int = 8,
    env_profile: str | None = None,
    env_id: str | None = None,
) -> EnvProfile:
    if env_profile:
        key = str(env_profile).strip()
        if key not in PROFILES:
            raise ValueError(
                f"Unknown --env-profile {key!r}. Expected one of: {sorted(PROFILES)}"
            )
        return PROFILES[key]
    if env_id:
        gym_id = str(env_id).strip()
        if gym_id not in _GYM_ID_TO_PROFILE:
            raise ValueError(f"Unknown --env-id {gym_id!r}")
        return _GYM_ID_TO_PROFILE[gym_id]
    return profile_for_env_size(env_size)


def add_profile_cli_args(parser) -> None:
    parser.add_argument(
        "--env-profile",
        default=None,
        choices=sorted(PROFILES),
        help="Env profile name. Default: Empty from --env-size.",
    )
    parser.add_argument(
        "--env-id",
        default=None,
        help="Gymnasium id override, e.g. MiniGrid-Empty-8x8-v0.",
    )
