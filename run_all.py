"""Run Day 22 lab steps individually or sequentially."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = {
    "1": "01_langsmith_rag_pipeline.py",
    "2": "02_prompt_hub_ab_routing.py",
    "3": "03_ragas_evaluation.py",
    "4": "04_guardrails_validator.py",
}


def run_step(step: str) -> None:
    script = STEPS[step]
    print("\n" + "=" * 72)
    print(f"Running step {step}: {script}")
    print("=" * 72)
    subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Day 22 LangSmith lab scripts.")
    parser.add_argument("--step", choices=STEPS.keys(), help="Run only one step.")
    args = parser.parse_args()

    selected = [args.step] if args.step else list(STEPS.keys())
    for step in selected:
        run_step(step)


if __name__ == "__main__":
    main()

