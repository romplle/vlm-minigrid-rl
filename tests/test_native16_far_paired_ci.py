"""CPU tests for the native 16x16 paired far interval. No VLM."""

import json
import unittest
from pathlib import Path

from vlm_minigrid_rl.native16_far_paired_ci import (
    FROZEN_GRPO_42,
    N_EPISODES,
    PHRASE_COVERS,
    PHRASE_COVERS_ABOVE,
    PHRASE_HOLDS,
    PHRASE_INVALID,
    PHRASE_BELOW,
    PRIOR_EXPERIMENT,
    SFT_EPISODE_CELLS,
    InvalidAlign,
    assert_gpu_aligned,
    assert_seed42_aligned,
    assert_writable_run_dir,
    classify_interval,
    compact_episode_bits,
    interval_from_deltas,
    paired_far_deltas_pp,
    seed42_interval,
)
from vlm_minigrid_rl.native16_grpo_evalseed import FROZEN_FAR_RUN, TRAIN_SEEDS


def _episode(stratum: str, success: bool, *, pos: tuple[int, int] = (1, 2), direction: int = 0) -> dict:
    visible = stratum == "visible"
    any_dir = stratum != "far"
    return {
        "success": success,
        "stratum": stratum,
        "start_goal_visible": visible,
        "start_visible_any_dir": any_dir,
        "start_pos": [pos[0], pos[1]],
        "start_dir": direction,
        "actions": ["forward", "left"],
    }


def _column(strata: list[str], successes: list[bool]) -> list[dict]:
    return [_episode(stratum, success, pos=(index, 1), direction=index % 4) for index, (stratum, success) in enumerate(zip(strata, successes))]


def _pad(strata: list[str], successes: list[bool]) -> tuple[list[str], list[bool]]:
    extra = N_EPISODES - len(strata)
    return strata + ["visible"] * extra, successes + [True] * extra


class WritableDirTest(unittest.TestCase):
    def test_refuse_frozen_and_prior_summary(self) -> None:
        with self.assertRaisesRegex(ValueError, FROZEN_FAR_RUN):
            assert_writable_run_dir(Path("runs") / FROZEN_FAR_RUN)
        with self.assertRaisesRegex(ValueError, PRIOR_EXPERIMENT):
            assert_writable_run_dir(Path("runs") / PRIOR_EXPERIMENT)
        assert_writable_run_dir(Path("runs/2026-09-native16-far-paired-ci"))


class CompactBitsTest(unittest.TestCase):
    def test_compact_record_drops_the_rollout(self) -> None:
        bits = compact_episode_bits([_episode("far", True)])
        self.assertEqual(set(bits[0]), {"index", "success", "stratum", "start_pos", "start_dir"})
        self.assertNotIn("actions", bits[0])
        self.assertEqual(bits[0]["stratum"], "far")
        self.assertTrue(bits[0]["success"])


class AlignTest(unittest.TestCase):
    def test_stratum_mismatch_is_invalid_align(self) -> None:
        strata, successes = _pad(["far", "facing_miss"], [True, False])
        sft = _column(strata, successes)
        grpo = {train: _column(strata, successes) for train in TRAIN_SEEDS}
        flipped, flipped_success = _pad(["facing_miss", "far"], [True, False])
        grpo[42] = _column(flipped, flipped_success)
        with self.assertRaisesRegex(InvalidAlign, PHRASE_INVALID):
            assert_seed42_aligned(sft, grpo)

    def test_start_mismatch_is_invalid_align(self) -> None:
        strata, successes = _pad(["far"], [True])
        sft = _column(strata, successes)
        bits = {
            train: compact_episode_bits(_column(strata, successes))
            for train in TRAIN_SEEDS
        }
        bits[123][0]["start_pos"] = [9, 9]
        with self.assertRaisesRegex(InvalidAlign, PHRASE_INVALID):
            assert_gpu_aligned(sft, bits)


class PhraseTest(unittest.TestCase):
    def test_three_interval_phrases(self) -> None:
        holds = interval_from_deltas([10.0] * 40)
        self.assertEqual(holds["phrase"], PHRASE_HOLDS)
        self.assertGreater(holds["lo"], 0.0)

        withdrawn = interval_from_deltas([-10.0] * 40)
        self.assertEqual(withdrawn["phrase"], PHRASE_BELOW)
        self.assertLess(withdrawn["hi"], 0.0)

        above = interval_from_deltas([30.0] * 101 + [-30.0] * 99)
        self.assertEqual(above["phrase"], PHRASE_COVERS_ABOVE)
        self.assertGreater(above["point"], 0.0)
        self.assertLessEqual(above["lo"], 0.0)
        self.assertGreaterEqual(above["hi"], 0.0)

        covers = interval_from_deltas([30.0] * 100 + [-30.0] * 100)
        self.assertEqual(covers["phrase"], PHRASE_COVERS)
        self.assertLessEqual(covers["point"], 0.0)

    def test_classify_bounds(self) -> None:
        self.assertEqual(classify_interval(1.0, 0.1, 2.0), PHRASE_HOLDS)
        self.assertEqual(classify_interval(-1.0, -2.0, -0.1), PHRASE_BELOW)
        self.assertEqual(classify_interval(0.2, -1.0, 1.0), PHRASE_COVERS_ABOVE)
        self.assertEqual(classify_interval(0.0, -1.0, 1.0), PHRASE_COVERS)
        self.assertEqual(classify_interval(-0.2, -1.0, 1.0), PHRASE_COVERS)


class DeltaTest(unittest.TestCase):
    def test_far_mean_is_grpo_minus_sft(self) -> None:
        strata = ["far", "far", "visible"]
        sft = _column(strata, [True, False, True])
        grpo = {
            42: _column(strata, [False, False, True]),
            123: _column(strata, [False, False, False]),
            456: _column(strata, [True, False, True]),
        }
        deltas = paired_far_deltas_pp(sft, grpo)
        self.assertEqual(len(deltas), 2)
        self.assertAlmostEqual(deltas[0], 100.0 * ((0 + 0 + 1) / 3.0 - 1.0))
        self.assertAlmostEqual(deltas[1], 100.0 * ((0 + 0 + 0) / 3.0 - 0.0))


def _repo_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "runs" / FROZEN_FAR_RUN).is_dir():
            return parent
    return None


class FrozenSeed42Test(unittest.TestCase):
    def test_frozen_bits_align_and_match_published_point(self) -> None:
        root = _repo_root()
        if root is None:
            self.skipTest("frozen far cells are not in this tree")
        sft = json.loads((root / SFT_EPISODE_CELLS[42]).read_text(encoding="utf-8"))
        grpo = {
            train: json.loads((root / rel).read_text(encoding="utf-8"))["episodes"]
            for train, rel in FROZEN_GRPO_42.items()
        }
        block = seed42_interval(sft["episodes"], grpo)
        config_delta = (
            sum(float(json.loads((root / rel).read_text(encoding="utf-8"))["config"]["success_far"]) for rel in FROZEN_GRPO_42.values())
            / 3.0
            - float(sft["config"]["success_far"])
        )
        self.assertEqual(block["n_far"], 142)
        self.assertAlmostEqual(block["point"], config_delta, places=6)
        self.assertLess(block["point"], 0.0)


if __name__ == "__main__":
    unittest.main()
