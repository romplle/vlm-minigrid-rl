"""CPU tests for SFT goal-red stratum verdict. No VLM."""

import unittest
from pathlib import Path

from vlm_minigrid_rl.c1_goalred import C1_EVAL_8X8, METHOD_G1
from vlm_minigrid_rl.sft_goalred_visible import (
    EVAL_SEEDS,
    FROZEN_GOALRED_RUN,
    FROZEN_OOD_RUN,
    VERDICT_INVALID,
    VERDICT_OTHER,
    VERDICT_UNLOCALIZED,
    VERDICT_VISIBLE,
    assert_not_frozen_run_dir,
    build_summary,
    cell_out_path,
    decide_verdict,
    planned_cells,
    row_from_payload,
    stratum_change,
)


def _arm(
    policy: str,
    epoch: int,
    success: float,
    visible: float,
    facing: float,
    far: float,
    *,
    n_visible: int = 100,
    n_facing: int = 140,
    n_far: int = 10,
) -> list[dict]:
    return [
        {
            "policy": policy,
            "epoch": epoch,
            "seed": seed,
            "success_rate": success,
            "success_visible": visible,
            "success_facing_miss": facing,
            "success_far": far,
            "n_visible": n_visible,
            "n_facing_miss": n_facing,
            "n_far": n_far,
        }
        for seed in EVAL_SEEDS
    ]


class FrozenDirTest(unittest.TestCase):
    def test_refuse_frozen_catalogs(self) -> None:
        for frozen in (FROZEN_GOALRED_RUN, FROZEN_OOD_RUN):
            with self.assertRaisesRegex(ValueError, frozen):
                assert_not_frozen_run_dir(Path("runs") / frozen)
            with self.assertRaisesRegex(ValueError, frozen):
                assert_not_frozen_run_dir(Path("I:/VS Code/vlm-minigrid-rl/runs") / frozen / "nested")
        assert_not_frozen_run_dir(Path("runs/2026-09-sft-goalred-visible"))

    def test_twelve_g1_cells(self) -> None:
        cells = planned_cells()
        self.assertEqual(len(cells), 12)
        self.assertEqual(cell_out_path(Path("runs/x"), "SFT", 1, 42).name, "sft_e1_s42.json")
        self.assertEqual(cell_out_path(Path("runs/x"), "GRPO", 9, 456).name, "grpo_e9_s456.json")


class RowTest(unittest.TestCase):
    def test_row_rejects_paint_and_wrong_cap(self) -> None:
        payload = {
            "config": {
                "policy": "SFT",
                "sft_epoch": 9,
                "seed": 42,
                "method": "PAINT",
                "eval_max_steps": C1_EVAL_8X8,
            },
            "policies": [{"name": "SFT", "metrics": {"success_rate": 17.87}}],
            "strata": {},
        }
        with self.assertRaisesRegex(ValueError, "G1"):
            row_from_payload(payload)
        payload["config"]["method"] = METHOD_G1
        payload["config"]["eval_max_steps"] = 40
        with self.assertRaisesRegex(ValueError, "12"):
            row_from_payload(payload)


class VerdictTest(unittest.TestCase):
    def _decide(
        self,
        *,
        e5_visible: float,
        e9_visible: float,
        e5_facing: float = 40.0,
        e9_facing: float = 40.0,
        e5_far: float = 10.0,
        e9_far: float = 10.0,
        n_visible: int = 100,
        n_facing: int = 140,
        n_far: int = 10,
        sft_e9: float = 17.87,
        grpo_e9: float = 2.40,
    ) -> dict:
        return decide_verdict(
            sft_e5=_arm(
                "SFT",
                5,
                40.27,
                e5_visible,
                e5_facing,
                e5_far,
                n_visible=n_visible,
                n_facing=n_facing,
                n_far=n_far,
            ),
            sft_e9=_arm(
                "SFT",
                9,
                sft_e9,
                e9_visible,
                e9_facing,
                e9_far,
                n_visible=n_visible,
                n_facing=n_facing,
                n_far=n_far,
            ),
            grpo_e9=_arm("GRPO", 9, grpo_e9, 5.0, 1.0, 0.0),
        )

    def test_visible_carries(self) -> None:
        decision = self._decide(e5_visible=70.0, e9_visible=50.0, e5_facing=40.0, e9_facing=10.0)
        self.assertEqual(decision["verdict"], VERDICT_VISIBLE)
        self.assertEqual(decision["carrier_strata"], ["visible"])
        self.assertTrue(decision["suite_ok"])

    def test_other_stratum_carries(self) -> None:
        decision = self._decide(
            e5_visible=80.0,
            e9_visible=78.0,
            e5_facing=50.0,
            e9_facing=20.0,
            n_facing=125,
        )
        change = decision["strata_e9_minus_e5"]["facing_miss"]
        self.assertAlmostEqual(change["contribution_pp"], (125 / 250) * (20.0 - 50.0))
        self.assertEqual(decision["verdict"], VERDICT_OTHER)
        self.assertEqual(decision["carrier_strata"], ["facing_miss"])

    def test_joint_other_when_neither_stratum_alone_hits_ten(self) -> None:
        decision = self._decide(
            e5_visible=80.0,
            e9_visible=80.0,
            e5_facing=40.0,
            e9_facing=25.0,
            e5_far=40.0,
            e9_far=10.0,
            n_facing=100,
            n_far=50,
        )
        self.assertEqual(decision["verdict"], VERDICT_OTHER)
        self.assertEqual(decision["carrier_strata"], ["facing_miss", "far"])

    def test_unlocalized_when_visible_drop_is_between_gates(self) -> None:
        decision = self._decide(e5_visible=40.0, e9_visible=34.0)
        self.assertEqual(decision["verdict"], VERDICT_UNLOCALIZED)

    def test_unlocalized_when_nonvisible_contribution_is_small(self) -> None:
        decision = self._decide(
            e5_visible=40.0,
            e9_visible=39.0,
            e5_facing=50.0,
            e9_facing=45.0,
            n_facing=100,
        )
        self.assertEqual(decision["verdict"], VERDICT_UNLOCALIZED)

    def test_invalid_suite_blocks_strata(self) -> None:
        decision = self._decide(e5_visible=70.0, e9_visible=50.0, sft_e9=30.0)
        self.assertEqual(decision["verdict"], VERDICT_INVALID)
        self.assertFalse(decision["suite_ok"])
        self.assertEqual(decision["carrier_strata"], [])

    def test_gate_is_inclusive_at_three_pp(self) -> None:
        inside = self._decide(e5_visible=40.0, e9_visible=40.0, sft_e9=20.87, grpo_e9=5.40)
        self.assertTrue(inside["suite_ok"])
        outside = self._decide(e5_visible=40.0, e9_visible=40.0, sft_e9=20.88)
        self.assertEqual(outside["verdict"], VERDICT_INVALID)

    def test_summary_keeps_e1_out_of_the_verdict(self) -> None:
        rows = (
            _arm("SFT", 1, 30.93, 90.0, 20.0, 0.0)
            + _arm("SFT", 5, 40.27, 80.0, 50.0, 10.0, n_facing=125)
            + _arm("SFT", 9, 17.87, 78.0, 20.0, 10.0, n_facing=125)
            + _arm("GRPO", 9, 2.40, 5.0, 1.0, 0.0)
        )
        summary = build_summary(rows)
        self.assertEqual(summary["verdict"], VERDICT_OTHER)
        self.assertIn("sft_e1", summary["by_arm"])
        delta = stratum_change(
            _arm("SFT", 5, 40.27, 80.0, 50.0, 10.0, n_facing=125),
            _arm("SFT", 9, 17.87, 78.0, 20.0, 10.0, n_facing=125),
            rate_key="success_facing_miss",
            n_key="n_facing_miss",
            episodes=250,
        )
        self.assertAlmostEqual(delta["delta"], -30.0)


if __name__ == "__main__":
    unittest.main()
