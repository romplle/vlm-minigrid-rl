"""CPU memoryless Bayes audit. C1 Empty caps 12/40. No VLM."""

from _bootstrap import bootstrap

bootstrap()

from vlm_minigrid_rl.memoryless_bayes import cli_main


if __name__ == "__main__":
    cli_main()
