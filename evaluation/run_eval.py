import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.run_eval import run_benchmark

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Support Agent Benchmarks")
    parser.add_argument("--samples", type=int, default=None,
                        help="Number of samples to evaluate (default: full golden set)")
    args = parser.parse_args()

    run_benchmark(samples_limit=args.samples)