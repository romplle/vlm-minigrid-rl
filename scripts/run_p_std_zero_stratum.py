"""CPU p_all_fail by stratum: join C1 GRPO logs + spin/search control."""

from _bootstrap import bootstrap

bootstrap()

from vlm_minigrid_rl.p_std_zero_stratum import cli_main


if __name__ == "__main__":
    cli_main()
