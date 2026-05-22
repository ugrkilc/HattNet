"""
Run all HattNet ablation modes sequentially.

Place this file at the project root (next to the 'hattnet/' folder).

Usage:
    python run_all.py
"""

import os
import sys
import traceback


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hattnet import AVAILABLE_MODES, SEED, set_seed, run_mode


def main():
    set_seed(SEED)

    summary_path = "ablation_summary.txt"
    if not os.path.exists(summary_path):
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("HattNet Ablation Summary\n")
            f.write("=" * 100 + "\n")
            f.write(f"{'Mode':<32} | {'Accuracy':<22} | Metrics                          | Time\n")
            f.write("-" * 100 + "\n")

    print("Running ALL ablation modes...\n")
    for mode in AVAILABLE_MODES:
        print(f"\n\n{'#' * 60}")
        print(f"# Mode: {mode}")
        print(f"{'#' * 60}\n")
        try:
            run_mode(mode, summary_path=summary_path)
        except Exception as e:
            print(f"ERROR in {mode}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
