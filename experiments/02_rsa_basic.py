"""Experiment 02: RSA (Recursive Self-Aggregation) on AIME 2025.

RSA with N=8, K=3, T=5. Log pass@1 (population accuracy) at each step t=0..5.
This shows whether iterative aggregation improves accuracy over steps.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rsa import RSAConfig, run_rsa

# ── Config ──────────────────────────────────────────────────────────────
CONFIG = RSAConfig(
    N=8,
    K=3,
    T=5,
    max_tokens=16384,
    temperature=0.7,
    port=30000,
    model="default",
    max_concurrent=32,
)
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "aime_2025.json")
RESULTS_DIR = "/workspace/results"
# ────────────────────────────────────────────────────────────────────────


def main():
    with open(DATA_PATH) as f:
        problems = json.load(f)

    t_start = time.time()
    results = run_rsa(problems, CONFIG)
    total_time = time.time() - t_start

    # Aggregate per-step metrics across all problems
    n_problems = len(results)
    n_steps = CONFIG.T + 1  # steps 0..T

    step_metrics = []
    for t in range(n_steps):
        accs = [r.steps[t].accuracy for r in results]
        pass_ns = [r.steps[t].pass_at_n for r in results]
        maj_corrects = [r.steps[t].majority_correct for r in results]
        step_metrics.append({
            "step": t,
            "mean_accuracy": round(sum(accs) / n_problems, 4),
            "pass_at_n": round(sum(pass_ns) / n_problems, 4),
            "majority_vote_accuracy": round(sum(maj_corrects) / n_problems, 4),
        })

    # Per-problem details
    per_problem = []
    for r in results:
        problem_data = {
            "problem_id": r.problem_id,
            "ground_truth": r.ground_truth,
            "final_majority_answer": r.final_majority_answer,
            "final_majority_correct": r.final_majority_correct,
            "elapsed_seconds": round(r.elapsed_seconds, 1),
            "steps": [],
        }
        for s in r.steps:
            problem_data["steps"].append({
                "step": s.step,
                "answers": s.answers,
                "accuracy": round(s.accuracy, 4),
                "pass_at_n": s.pass_at_n,
                "majority_answer": s.majority_answer,
                "majority_correct": s.majority_correct,
            })
        per_problem.append(problem_data)

    final_correct = sum(1 for r in results if r.final_majority_correct)

    summary = {
        "experiment": "02_rsa_basic",
        "model": "QED-Nano (lm-provers/QED-Nano)",
        "config": {
            "N": CONFIG.N,
            "K": CONFIG.K,
            "T": CONFIG.T,
            "max_tokens": CONFIG.max_tokens,
            "temperature": CONFIG.temperature,
        },
        "metrics": {
            "n_problems": n_problems,
            "final_majority_vote_accuracy": round(final_correct / n_problems, 4),
            "per_step": step_metrics,
        },
        "timing": {
            "total_seconds": round(total_time, 1),
            "seconds_per_problem": round(total_time / n_problems, 1),
        },
        "per_problem": per_problem,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "02_rsa_basic.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"\n=== Step-by-step metrics ===")
    for sm in step_metrics:
        print(f"  Step {sm['step']}: mean_acc={sm['mean_accuracy']:.4f}  "
              f"pass@N={sm['pass_at_n']:.4f}  majority={sm['majority_vote_accuracy']:.4f}")
    print(f"\n  Final majority vote accuracy: {final_correct}/{n_problems} = "
          f"{final_correct/n_problems:.4f}")
    print(f"  Total time: {total_time:.0f}s")


if __name__ == "__main__":
    main()
