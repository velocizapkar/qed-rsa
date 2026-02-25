"""Experiment 03: RSA with self-verify + Boltzmann selection (placeholder).

This extends the basic RSA loop with:
  1. Self-verification: after aggregation, ask the model to verify each candidate
  2. Boltzmann selection: use verification scores as weights when subsampling

TODO: Implement the following:
  - verify_prompt(problem, candidate) -> "True" or "False"
  - Score each candidate with multiple verify calls, average to get p(correct)
  - Use softmax(scores / tau) as sampling weights instead of uniform random.sample
  - Compare against 02_rsa_basic to see if verification improves convergence
"""

import json
import os
import sys
import time
import asyncio
import math
import random
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import AsyncOpenAI
from core.eval import extract_answer, majority_vote, population_accuracy, pass_at_n
from core.generate import get_client, generate_batch, generate_one
from core.prompts import solve_prompt, aggregation_prompt
from core.rsa import RSAConfig

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
VERIFY_SAMPLES = 3    # number of verify calls per candidate
BOLTZMANN_TAU = 1.0   # temperature for Boltzmann selection
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "aime_2025.json")
RESULTS_DIR = "/workspace/results"
# ────────────────────────────────────────────────────────────────────────


VERIFY_SYSTEM = (
    "You are a math verification assistant. "
    "Determine if the given solution to the problem is correct."
)


def verify_prompt(problem: str, candidate: str) -> List[dict]:
    """Build a chat prompt asking the model to verify a candidate solution."""
    user_content = (
        "You are given a problem and a candidate solution. "
        "Verify whether the candidate solution is correct. "
        "If the solution is correct, output only True. "
        "If it is incorrect, output only False. "
        "Do not generate anything else.\n\n"
        f"Problem:\n{problem.strip()}\n\n"
        f"Candidate solution:\n{candidate.strip()}\n\n"
        'Now verify if the solution is True or False. Only output "True" or "False".'
    )
    return [
        {"role": "system", "content": VERIFY_SYSTEM},
        {"role": "user", "content": user_content},
    ]


def parse_verify(response: str) -> bool:
    """Parse a verification response into a boolean."""
    text = response.strip().lower()
    if "true" in text:
        return True
    return False


async def verify_candidates(
    client: AsyncOpenAI,
    problem: str,
    candidates: List[str],
    config: RSAConfig,
    n_verify: int = 3,
) -> List[float]:
    """Score each candidate by running n_verify verification calls.

    Returns a list of scores in [0, 1] for each candidate.
    """
    all_messages = []
    for cand in candidates:
        for _ in range(n_verify):
            all_messages.append(verify_prompt(problem, cand))

    responses = await generate_batch(
        client, all_messages,
        model=config.model, max_tokens=16,
        temperature=0.1, max_concurrent=config.max_concurrent,
    )

    scores = []
    idx = 0
    for _ in candidates:
        correct_count = 0
        for _ in range(n_verify):
            if parse_verify(responses[idx]):
                correct_count += 1
            idx += 1
        scores.append(correct_count / n_verify)
    return scores


def boltzmann_sample(candidates: List[str], scores: List[float], k: int, tau: float) -> List[str]:
    """Sample k candidates using Boltzmann (softmax) weights."""
    if tau <= 0 or all(s == 0 for s in scores):
        return random.sample(candidates, k)

    # Compute softmax weights
    max_score = max(scores)
    exp_scores = [math.exp((s - max_score) / tau) for s in scores]
    total = sum(exp_scores)
    weights = [e / total for e in exp_scores]

    # Weighted sampling without replacement
    indices = list(range(len(candidates)))
    selected = []
    remaining_weights = list(weights)
    remaining_indices = list(indices)

    for _ in range(k):
        total_w = sum(remaining_weights)
        if total_w <= 0:
            # Fallback to uniform
            idx = random.choice(remaining_indices)
        else:
            probs = [w / total_w for w in remaining_weights]
            idx = random.choices(remaining_indices, weights=probs, k=1)[0]
        selected.append(candidates[idx])
        pos = remaining_indices.index(idx)
        remaining_indices.pop(pos)
        remaining_weights.pop(pos)

    return selected


async def rsa_verify_one_problem(
    client: AsyncOpenAI,
    problem_id: str,
    problem_text: str,
    ground_truth: int,
    config: RSAConfig,
) -> dict:
    """Run RSA with verification + Boltzmann selection on one problem."""
    t0 = time.time()
    steps_log = []

    # Step 0: generate initial population
    init_messages = [solve_prompt(problem_text) for _ in range(config.N)]
    population = await generate_batch(
        client, init_messages,
        model=config.model, max_tokens=config.max_tokens,
        temperature=config.temperature, max_concurrent=config.max_concurrent,
    )

    answers = [extract_answer(c) for c in population]
    acc = population_accuracy(population, ground_truth)
    mv = majority_vote(population)
    steps_log.append({
        "step": 0, "answers": answers, "accuracy": round(acc, 4),
        "majority_answer": mv, "majority_correct": mv == ground_truth,
        "verify_scores": None,
    })

    for t in range(1, config.T + 1):
        # Verify current population
        scores = await verify_candidates(
            client, problem_text, population, config,
            n_verify=VERIFY_SAMPLES,
        )

        # Aggregate with Boltzmann-weighted subsampling
        agg_messages = []
        for _ in range(config.N):
            subset = boltzmann_sample(population, scores, config.K, BOLTZMANN_TAU)
            agg_messages.append(aggregation_prompt(problem_text, subset))

        population = await generate_batch(
            client, agg_messages,
            model=config.model, max_tokens=config.max_tokens,
            temperature=config.temperature, max_concurrent=config.max_concurrent,
        )

        answers = [extract_answer(c) for c in population]
        acc = population_accuracy(population, ground_truth)
        mv = majority_vote(population)
        steps_log.append({
            "step": t, "answers": answers, "accuracy": round(acc, 4),
            "majority_answer": mv, "majority_correct": mv == ground_truth,
            "verify_scores": [round(s, 3) for s in scores],
        })

    final_mv = steps_log[-1]["majority_answer"]
    return {
        "problem_id": problem_id,
        "ground_truth": ground_truth,
        "final_majority_answer": final_mv,
        "final_majority_correct": final_mv == ground_truth,
        "elapsed_seconds": round(time.time() - t0, 1),
        "steps": steps_log,
    }


async def run_all():
    with open(DATA_PATH) as f:
        problems = json.load(f)

    client = get_client(port=CONFIG.port)
    t_start = time.time()
    all_results = []

    for i, p in enumerate(problems):
        print(f"[{i+1}/{len(problems)}] {p['id']}...", flush=True)
        r = await rsa_verify_one_problem(
            client, p["id"], p["problem"], p["answer"], CONFIG,
        )
        for s in r["steps"]:
            status = "correct" if s["majority_correct"] else "wrong"
            verify_info = ""
            if s["verify_scores"] is not None:
                verify_info = f" verify_scores={s['verify_scores']}"
            print(f"  Step {s['step']}: acc={s['accuracy']:.2f} "
                  f"majority={s['majority_answer']} ({status}){verify_info}")
        all_results.append(r)

    total_time = time.time() - t_start
    n_problems = len(problems)

    # Per-step aggregated metrics
    n_steps = CONFIG.T + 1
    step_metrics = []
    for t in range(n_steps):
        accs = [r["steps"][t]["accuracy"] for r in all_results]
        majs = [r["steps"][t]["majority_correct"] for r in all_results]
        step_metrics.append({
            "step": t,
            "mean_accuracy": round(sum(accs) / n_problems, 4),
            "majority_vote_accuracy": round(sum(majs) / n_problems, 4),
        })

    final_correct = sum(1 for r in all_results if r["final_majority_correct"])

    summary = {
        "experiment": "03_rsa_verify",
        "model": "QED-Nano (lm-provers/QED-Nano)",
        "config": {
            "N": CONFIG.N,
            "K": CONFIG.K,
            "T": CONFIG.T,
            "max_tokens": CONFIG.max_tokens,
            "temperature": CONFIG.temperature,
            "verify_samples": VERIFY_SAMPLES,
            "boltzmann_tau": BOLTZMANN_TAU,
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
        "per_problem": all_results,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "03_rsa_verify.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"\n=== Step-by-step metrics ===")
    for sm in step_metrics:
        print(f"  Step {sm['step']}: mean_acc={sm['mean_accuracy']:.4f}  "
              f"majority={sm['majority_vote_accuracy']:.4f}")
    print(f"\n  Final majority vote accuracy: {final_correct}/{n_problems} = "
          f"{final_correct/n_problems:.4f}")
    print(f"  Total time: {total_time:.0f}s")


if __name__ == "__main__":
    asyncio.run(run_all())
