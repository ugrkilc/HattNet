"""
Run a single HattNet ablation mode.

Place this file at the project root (next to the 'hattnet/' folder).

Examples:
    python run_single.py                          # default: coord_attention
    python run_single.py --mode cbam
    python run_single.py --mode no_mask
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hattnet import AVAILABLE_MODES, SEED, set_seed, run_mode


def main():
    parser = argparse.ArgumentParser(description="HattNet - single mode runner")
    parser.add_argument("--mode", type=str, default="coord_attention",
                        choices=AVAILABLE_MODES,
                        help="Ablation mode to run (default: coord_attention)")
    args = parser.parse_args()

    set_seed(SEED)

    summary_path = "ablation_summary.txt"
    if not os.path.exists(summary_path):
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("HattNet Ablation Summary\n")
            f.write("=" * 100 + "\n")
            f.write(f"{'Mode':<32} | {'Accuracy':<22} | Metrics                          | Time\n")
            f.write("-" * 100 + "\n")

    run_mode(args.mode, summary_path=summary_path)


if __name__ == "__main__":
    main()
