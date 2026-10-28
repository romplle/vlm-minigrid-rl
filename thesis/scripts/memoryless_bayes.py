"""CPU memoryless Bayes audit (spin vs search, C1 caps 12/40). No VLM.

Run from repo root with PYTHONPATH=src, or use scripts/run_memoryless_bayes.py.
"""

from vlm_minigrid_rl.memoryless_bayes import cli_main


if __name__ == "__main__":
    cli_main()
