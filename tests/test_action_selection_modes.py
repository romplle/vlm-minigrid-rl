"""Unit tests for generate vs argmax3 action selection."""

import inspect
import unittest
from unittest.mock import MagicMock, patch

import torch

from vlm_minigrid_rl.minigrid_utils import evaluate_model_in_env
from vlm_minigrid_rl.model_utils import select_action_argmax3
from vlm_minigrid_rl.training_utils import ACTION_NAMES, ACTION_TO_ID


class ActionSelectionModesTest(unittest.TestCase):
    def test_select_action_argmax3_picks_highest_of_three(self):
        model = MagicMock()
        tokenizer = MagicMock()
        image_processor = MagicMock()
        fake_logits = torch.tensor([1.0, 3.0, 2.0])

        with (
            patch("vlm_minigrid_rl.model_utils.build_inference_inputs") as build_inputs,
            patch("vlm_minigrid_rl.model_utils.action_token_ids") as token_ids,
            patch("vlm_minigrid_rl.model_utils.single_token_action_ids") as single_ids,
            patch("vlm_minigrid_rl.model_utils.score_action_logits", return_value=fake_logits),
        ):
            build_inputs.return_value = (MagicMock(), MagicMock(), MagicMock())
            token_ids.return_value = [[1], [2], [3]]
            single_ids.return_value = [1, 2, 3]

            action_name, action_idx, generated = select_action_argmax3(
                model, tokenizer, image_processor, image=None, prompt="x", device="cpu"
            )

        self.assertEqual(action_name, "right")
        self.assertEqual(action_idx, ACTION_TO_ID["right"])
        self.assertEqual(generated, "")
        self.assertIn(action_name, ACTION_NAMES)

    def test_evaluate_model_in_env_default_action_mode_is_generate(self):
        sig = inspect.signature(evaluate_model_in_env)
        self.assertEqual(sig.parameters["action_mode"].default, "generate")


if __name__ == "__main__":
    unittest.main()
