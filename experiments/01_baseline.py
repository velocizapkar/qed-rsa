"""Experiment 01: QED-Nano baseline on AIME 2025.

Generate 8 independent solutions per problem to measure:
  - pass@1: accuracy of a single random sample
  - pass@8: did any of the 8 samples get the correct answer

This tells us the ceiling that RSA is trying to reach.
"""

import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.eval import extract_answer, is_correct, majority_vote, population_accuracy, pass_at_n
from core.generate import get_client, generate_batch
from core.prompts import solve_prompt

# ── Config ──────────────────────────────────────────────────────────────
N_SAMPLES = 8
MAX_TOKENS = 16384
TEMPERATURE = 0.7
PORT = 30000
MODEL = "default"
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "aime_2025.json")
RESULTS_DIR = "/workspace/results"
# ────────────────────────────────────────────────────────────────────────


async def run_baseline():
    with open(DATA_PATH) as f:
        problems = json.load(f)

    client = get_client(port=PORT)
    all_results = []
    total_correct_pass1 = 0
    total_correct_passn = 0
    total_correct_majority = 0
    t_start = time.time()

    for i, p in enumerate(problems):
        print(f"[{i+1}/{len(problems)}] {p['id']}...", flush=True)
        t0 = time.time()

        # Generate N independent solutions
        messages_list = [solve_prompt(p["problem"]) for _ in range(N_SAMPLES)]
        completions = await generate_batch(
            client, messages_list,
            model=MODEL, max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE, max_concurrent=N_SAMPLES,
        )

        elapsed = time.time() - t0
        answers = [extract_answer(c) for c in completions]
        correct_flags = [is_correct(c, p["answer"]) for c in completions]
        acc = population_accuracy(completions, p["answer"])
        pan = pass_at_n(completions, p["answer"])
        mv = majority_vote(completions)
        mv_correct = (mv == p["answer"])

        total_correct_pass1 += acc
        total_correct_passn += int(pan)
        total_correct_majority += int(mv_correct)

        result = {
            "problem_id": p["id"],
            "ground_truth": p["answer"],
            "answers": answers,
            "correct_flags": correct_flags,
            "pass_at_1": acc,
            "pass_at_n": pan,
            "majority_answer": mv,
            "majority_correct": mv_correct,
            "elapsed_seconds": round(elapsed, 1),
        }
        all_results.append(result)

        status = "✓" if mv_correct else "✗"
        print(f"  {status} acc={acc:.2f} pass@{N_SAMPLES}={'Y' if pan else 'N'} "
              f"majority={mv} (gt={p['answer']}) [{elapsed:.1f}s]")

    total_time = time.time() - t_start
    n_problems = len(problems)

    summary = {
        "experiment": "01_baseline",
        "model": "QED-Nano (lm-provers/QED-Nano)",
        "config": {
            "n_samples": N_SAMPLES,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
        },
        "metrics": {
            "n_problems": n_problems,
            "mean_pass_at_1": round(total_correct_pass1 / n_problems, 4),
            "pass_at_n": round(total_correct_passn / n_problems, 4),
            "majority_vote_accuracy": round(total_correct_majority / n_problems, 4),
        },
        "timing": {
            "total_seconds": round(total_time, 1),
            "seconds_per_problem": round(total_time / n_problems, 1),
        },
        "per_problem": all_results,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "01_baseline.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"\n=== Summary ===")
    print(f"  Mean pass@1:       {summary['metrics']['mean_pass_at_1']:.4f}")
    print(f"  pass@{N_SAMPLES}:           {summary['metrics']['pass_at_n']:.4f}")
    print(f"  Majority vote acc: {summary['metrics']['majority_vote_accuracy']:.4f}")
    print(f"  Total time:        {total_time:.0f}s")


if __name__ == "__main__":
    asyncio.run(run_baseline())
