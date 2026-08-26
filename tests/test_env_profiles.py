"""Empty env profiles freeze train/eval horizons and gym ids."""

import unittest
from unittest.mock import patch

from vlm_minigrid_rl.env_profiles import EMPTY_8X8, EMPTY_16X16, profile_for_env_size, resolve_profile
from vlm_minigrid_rl.minigrid_utils import (
    create_minigrid_env,
    default_eval_max_steps,
    default_max_steps,
    default_train_max_steps,
)
from vlm_minigrid_rl.training_utils import ACTION_NAMES, ACTION_TO_ID, build_navigation_prompt


class EmptyProfileFreezeTest(unittest.TestCase):
    def test_empty_8x8_matches_frozen_stand(self):
        p = profile_for_env_size(8)
        self.assertEqual(p.name, "empty-8x8")
        self.assertIs(p, EMPTY_8X8)
        self.assertEqual(p.gym_id, "MiniGrid-Empty-8x8-v0")
        self.assertEqual(p.env_size, 8)
        self.assertEqual(list(p.action_names), ACTION_NAMES)
        self.assertEqual(dict(p.action_to_id), ACTION_TO_ID)
        self.assertEqual(p.train_max_steps, 12)
        self.assertEqual(p.eval_max_steps, 16)
        self.assertEqual(p.max_steps, 16)
        self.assertEqual(p.dataset_slug, "8x8")
        self.assertEqual(p.kind, "empty")
        self.assertEqual(p.val_split, 0.1)
        self.assertEqual(p.prompt("green"), build_navigation_prompt("green"))
        self.assertEqual(default_train_max_steps(8), 12)
        self.assertEqual(default_eval_max_steps(8), 16)
        self.assertEqual(default_max_steps(8), 16)

    def test_empty_16x16_matches_frozen_stand(self):
        p = profile_for_env_size(16)
        self.assertEqual(p.name, "empty-16x16")
        self.assertIs(p, EMPTY_16X16)
        self.assertEqual(p.gym_id, "MiniGrid-Empty-16x16-v0")
        self.assertEqual(p.train_max_steps, 28)
        self.assertEqual(p.eval_max_steps, 38)
        self.assertEqual(p.max_steps, 38)
        self.assertEqual(p.dataset_slug, "16x16")
        self.assertEqual(p.prompt("red"), build_navigation_prompt("red"))
        self.assertEqual(default_train_max_steps(16), 28)
        self.assertEqual(default_eval_max_steps(16), 38)
        self.assertEqual(default_max_steps(16), 38)

    def test_resolve_profile_defaults_to_empty_from_env_size(self):
        self.assertIs(resolve_profile(env_size=8), EMPTY_8X8)
        self.assertIs(resolve_profile(env_size=16), EMPTY_16X16)

    def test_resolve_by_env_profile_and_env_id(self):
        self.assertIs(resolve_profile(env_profile="empty-8x8"), EMPTY_8X8)
        self.assertIs(resolve_profile(env_id="MiniGrid-Empty-16x16-v0"), EMPTY_16X16)

    def test_unknown_profile_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_profile(env_profile="doorkey-6x6")

    def test_create_minigrid_env_uses_profile_gym_id(self):
        with patch("vlm_minigrid_rl.minigrid_utils.gym.make") as make:
            with patch("vlm_minigrid_rl.minigrid_utils.RGBImgPartialObsWrapper") as wrap:
                wrap.side_effect = lambda env, tile_size=32: env
                create_minigrid_env(8)
                make.assert_called_with("MiniGrid-Empty-8x8-v0", render_mode="rgb_array")
                create_minigrid_env(16)
                make.assert_called_with("MiniGrid-Empty-16x16-v0", render_mode="rgb_array")

    def test_create_minigrid_env_pins_horizon(self):
        env = create_minigrid_env(8)
        self.assertEqual(env.unwrapped.max_steps, EMPTY_8X8.eval_max_steps)
        env.close()
        env = create_minigrid_env(8, max_steps=EMPTY_8X8.train_max_steps)
        self.assertEqual(env.unwrapped.max_steps, EMPTY_8X8.train_max_steps)
        env.close()
        env = create_minigrid_env(16)
        self.assertEqual(env.unwrapped.max_steps, EMPTY_16X16.eval_max_steps)
        env.close()


if __name__ == "__main__":
    unittest.main()
