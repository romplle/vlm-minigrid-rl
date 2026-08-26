"""Leading-space Empty action tokens must be single ids, distinct from bare words."""

import unittest

from vlm_minigrid_rl.env_profiles import EMPTY_8X8

try:
    from vlm_minigrid_rl.model_utils import (
        action_token_ids,
        action_token_name_by_id,
        action_token_texts,
        load_project_tokenizer,
        single_token_action_ids,
    )
except ModuleNotFoundError:
    action_token_ids = None  # type: ignore[assignment]


@unittest.skipIf(action_token_ids is None, "nanoVLM checkout is required for tokenizer tests")
class ActionTokenizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = load_project_tokenizer()

    def test_empty_three_way_single_tokens(self):
        action_names = EMPTY_8X8.action_names
        texts = action_token_texts(action_names)
        ids_list = action_token_ids(self.tokenizer, action_names)
        single = single_token_action_ids(ids_list)
        self.assertIsNotNone(
            single,
            f"Expected one token per leading-space action, got {list(zip(texts, ids_list))}",
        )
        by_id = action_token_name_by_id(self.tokenizer)
        self.assertEqual(len(by_id), len(action_names))
        for name, ids, token_id in zip(action_names, ids_list, single):
            self.assertEqual(ids, [token_id])
            self.assertEqual(by_id[token_id], name)
            bare = self.tokenizer.encode(name, add_special_tokens=False)
            self.assertNotEqual(
                token_id,
                bare[0] if bare else None,
                f"{name!r} must not share the leading-space token id",
            )

    def test_eos_after_action_is_separate(self):
        eos = self.tokenizer.eos_token
        for name in EMPTY_8X8.action_names:
            ids = self.tokenizer.encode(f" {name}{eos}", add_special_tokens=False)
            action_ids = self.tokenizer.encode(f" {name}", add_special_tokens=False)
            self.assertEqual(len(action_ids), 1)
            self.assertEqual(ids[:1], action_ids)
            self.assertGreater(len(ids), 1)


if __name__ == "__main__":
    unittest.main()
