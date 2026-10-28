import tempfile
import unittest
from pathlib import Path

from vlm_minigrid_rl.handoff_sweep import (
    HEADLINE_GRPO_HP,
    build_delta_row,
    build_grpo_train_command,
    epochs_missing_from_picks,
    eval_json_path,
    extract_policy_metrics,
    grpo_episode_dir,
    grpo_handoff_root,
    list_sft_epochs,
    load_grpo_picks,
    merge_grpo_picks_on_disk,
    pick_best_grpo_episode,
    select_ood_sft_epochs,
    should_skip_grpo_train,
    sft_epoch_dir,
)


class HandoffSweepHelpersTest(unittest.TestCase):
    def test_list_sft_epochs_sorted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("epoch-10", "epoch-2", "epoch-1", "not-an-epoch"):
                d = root / name
                d.mkdir()
                if name.startswith("epoch-"):
                    (d / "adapter_config.json").write_text("{}", encoding="utf-8")
            self.assertEqual(list_sft_epochs(root), [1, 2, 10])

    def test_paths(self):
        root = Path("checkpoints/sft_adapter_8x8_bs32_e10")
        self.assertEqual(sft_epoch_dir(root, 5), root / "epoch-5")
        grpo = grpo_handoff_root(8, 5, checkpoints_root=Path("checkpoints"))
        self.assertEqual(grpo, Path("checkpoints/grpo_adapter_8x8_handoff_from_e10_e5"))
        self.assertEqual(
            grpo_handoff_root(16, 3, checkpoints_root=Path("checkpoints"), train_seed=42),
            Path("checkpoints/grpo_adapter_16x16_handoff_from_e10_e3"),
        )
        self.assertEqual(
            grpo_handoff_root(16, 3, checkpoints_root=Path("checkpoints"), train_seed=123),
            Path("checkpoints/grpo_adapter_16x16_handoff_from_e10_e3_seed123"),
        )

    def test_eval_json_path(self):
        run = Path("runs/2026-07-handoff-empty")
        self.assertEqual(
            eval_json_path(run, kind="sft", env_size=8, sft_epoch=3, seed=42),
            run / "8x8" / "sft_e3_seed42.json",
        )
        self.assertEqual(
            eval_json_path(
                run, kind="grpo", env_size=8, sft_epoch=3, seed=42, grpo_episode=100
            ),
            run / "8x8" / "grpo_e3_ep100_seed42.json",
        )

    def test_extract_and_delta(self):
        payload = {
            "policies": [
                {
                    "name": "SFT",
                    "metrics": {
                        "success_rate": 90.0,
                        "timeouts": 10,
                        "invalid_action_episodes": 1,
                    },
                },
                {
                    "name": "GRPO",
                    "metrics": {
                        "success_rate": 95.0,
                        "timeouts": 5,
                        "invalid_action_episodes": 0,
                    },
                },
            ]
        }
        sft = extract_policy_metrics(payload, "SFT")
        self.assertEqual(sft["success_rate"], 90.0)
        row = build_delta_row(
            sft_epoch=3,
            sft_rates=[90.0, 92.0, 91.0],
            grpo_rates=[95.0, 96.0, 94.0],
            sft_timeouts=[10, 8, 9],
            grpo_timeouts=[5, 4, 6],
            sft_invalids=[1, 1, 1],
            grpo_invalids=[0, 1, 0],
        )
        self.assertAlmostEqual(row["sft_mean"], 91.0)
        self.assertAlmostEqual(row["grpo_mean"], 95.0)
        self.assertAlmostEqual(row["delta_pp"], 4.0)
        self.assertEqual(HEADLINE_GRPO_HP["lr"], 5e-6)

    def test_select_ood_epochs_under_mid_over(self):
        rows = [
            build_delta_row(1, [70, 70, 70], [80, 80, 80], [1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0]),
            build_delta_row(3, [90, 90, 90], [95, 95, 95], [1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0]),
            build_delta_row(5, [94, 94, 94], [96, 96, 96], [1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0]),
            build_delta_row(9, [98, 98, 98], [98.5, 98.5, 98.5], [1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0]),
        ]
        self.assertEqual(select_ood_sft_epochs(rows, k=3), [1, 5, 9])

    def test_pick_best_grpo_episode(self):
        self.assertEqual(pick_best_grpo_episode({25: 90.0, 50: 96.0, 75: 95.0, 100: 96.0}), 100)

    def test_load_grpo_picks_merges_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            picks_file = Path(tmp) / "grpo_episode_picks.json"
            self.assertEqual(
                load_grpo_picks(picks_file, epochs=[1, 3, 9], default_episode=100),
                {1: 100, 3: 100, 9: 100},
            )

            picks_file.write_text('{"1": 50, "3": 75}', encoding="utf-8")
            self.assertEqual(
                load_grpo_picks(picks_file, epochs=[1, 3, 9], default_episode=100),
                {1: 50, 3: 75, 9: 100},
            )

            picks_file.write_text('{"1": 50, "3": 75, "9": 25, "99": 999}', encoding="utf-8")
            self.assertEqual(
                load_grpo_picks(picks_file, epochs=[1, 3, 9], default_episode=100),
                {1: 50, 3: 75, 9: 25},
            )

    def test_build_grpo_train_command(self):
        argv = build_grpo_train_command(
            python="python",
            grpo_script=Path("scripts/grpo.py"),
            env_size=8,
            dataset=Path("datasets/dataset_8x8"),
            sft_adapter=Path("checkpoints/sft_adapter_8x8_bs32_e10/epoch-5"),
            output_dir=Path("checkpoints/grpo_adapter_8x8_handoff_from_e10_e5"),
            train_episodes=100,
            checkpoint_interval=25,
            max_steps=12,
            seed=42,
        )
        joined = " ".join(argv)
        self.assertTrue("5e-06" in argv or "5e-6" in argv)
        self.assertIn("handoff_from_e10_e5", joined)
        self.assertIn("--epsilon", argv)
        self.assertIn("0.1", argv)
        self.assertIn("--lora-dropout", argv)
        self.assertIn("0.0", argv)

    def test_should_skip_grpo_train_only_when_target_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            grpo_root = Path(tmp) / "grpo_adapter_8x8_handoff_from_e10_e1"
            grpo_root.mkdir()
            partial = grpo_episode_dir(grpo_root, 25)
            partial.mkdir()
            (partial / "adapter_config.json").write_text("{}", encoding="utf-8")

            self.assertFalse(should_skip_grpo_train(grpo_root, 100, skip_existing=True))

            complete = grpo_episode_dir(grpo_root, 100)
            complete.mkdir()
            (complete / "adapter_config.json").write_text("{}", encoding="utf-8")
            self.assertTrue(should_skip_grpo_train(grpo_root, 100, skip_existing=True))
            self.assertFalse(should_skip_grpo_train(grpo_root, 100, skip_existing=False))

    def test_epochs_missing_from_picks(self):
        with tempfile.TemporaryDirectory() as tmp:
            picks_file = Path(tmp) / "grpo_episode_picks.json"
            self.assertEqual(epochs_missing_from_picks(picks_file, [1, 3, 9]), [1, 3, 9])

            picks_file.write_text('{"1": 50, "3": 75}', encoding="utf-8")
            self.assertEqual(epochs_missing_from_picks(picks_file, [1, 3, 9]), [9])
            self.assertEqual(epochs_missing_from_picks(picks_file, [1, 3]), [])

    def test_merge_grpo_picks_on_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            picks_file = Path(tmp) / "grpo_episode_picks.json"
            picks_file.write_text('{"1": 50, "3": 75}', encoding="utf-8")
            merged = merge_grpo_picks_on_disk(picks_file, {3: 100, 9: 25})
            self.assertEqual(merged, {"1": 50, "3": 100, "9": 25})


if __name__ == "__main__":
    unittest.main()
