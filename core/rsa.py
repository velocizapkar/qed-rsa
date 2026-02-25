"""RSA (Recursive Self-Aggregation) loop implementation.

Follows the RSA paper (arxiv 2509.26626):
  P_0 = [generate(problem) for _ in range(N)]
  for t in range(1, T+1):
      P_t = []
      for i in range(N):
          S = random.sample(P_{t-1}, K)
          new = generate(aggregation_prompt(problem, S))
          P_t.append(new)
      log pass@1 for P_t
  return majority_vote(P_T)
"""

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional

from openai import AsyncOpenAI

from core.eval import extract_answer, is_correct, majority_vote, population_accuracy, pass_at_n
from core.generate import get_client, generate_batch
from core.prompts import solve_prompt, aggregation_prompt


@dataclass
class RSAConfig:
    N: int = 8           # population size
    K: int = 3           # subsample size for aggregation
    T: int = 5           # number of RSA steps
    max_tokens: int = 16384
    temperature: float = 0.7
    port: int = 30000
    model: str = "default"
    max_concurrent: int = 32


@dataclass
class StepResult:
    """Metrics for one RSA step on one problem."""
    step: int
    completions: List[str]
    answers: List[Optional[int]]
    correct_answer: int
    accuracy: float          # fraction of population with correct answer
    pass_at_n: bool          # did any get it right
    majority_answer: Optional[int]
    majority_correct: bool


@dataclass
class ProblemResult:
    """Full RSA trace for a single problem."""
    problem_id: str
    problem_text: str
    ground_truth: int
    steps: List[StepResult] = field(default_factory=list)
    final_majority_answer: Optional[int] = None
    final_majority_correct: bool = False
    elapsed_seconds: float = 0.0


async def rsa_one_problem(
    client: AsyncOpenAI,
    problem_id: str,
    problem_text: str,
    ground_truth: int,
    config: RSAConfig,
) -> ProblemResult:
    """Run the full RSA loop on a single problem."""
    result = ProblemResult(
        problem_id=problem_id,
        problem_text=problem_text,
        ground_truth=ground_truth,
    )
    t0 = time.time()

    # Step 0: generate initial population
    init_messages = [solve_prompt(problem_text) for _ in range(config.N)]
    population = await generate_batch(
        client, init_messages,
        model=config.model, max_tokens=config.max_tokens,
        temperature=config.temperature, max_concurrent=config.max_concurrent,
    )

    # Log step 0
    answers = [extract_answer(c) for c in population]
    acc = population_accuracy(population, ground_truth)
    pan = pass_at_n(population, ground_truth)
    mv = majority_vote(population)
    result.steps.append(StepResult(
        step=0, completions=population, answers=answers,
        correct_answer=ground_truth, accuracy=acc, pass_at_n=pan,
        majority_answer=mv, majority_correct=(mv == ground_truth),
    ))

    # RSA steps 1..T
    for t in range(1, config.T + 1):
        agg_messages = []
        for _ in range(config.N):
            subset = random.sample(population, config.K)
            agg_messages.append(aggregation_prompt(problem_text, subset))

        population = await generate_batch(
            client, agg_messages,
            model=config.model, max_tokens=config.max_tokens,
            temperature=config.temperature, max_concurrent=config.max_concurrent,
        )

        answers = [extract_answer(c) for c in population]
        acc = population_accuracy(population, ground_truth)
        pan = pass_at_n(population, ground_truth)
        mv = majority_vote(population)
        result.steps.append(StepResult(
            step=t, completions=population, answers=answers,
            correct_answer=ground_truth, accuracy=acc, pass_at_n=pan,
            majority_answer=mv, majority_correct=(mv == ground_truth),
        ))

    result.final_majority_answer = result.steps[-1].majority_answer
    result.final_majority_correct = result.steps[-1].majority_correct
    result.elapsed_seconds = time.time() - t0
    return result


async def rsa_all_problems(
    problems: List[dict],
    config: RSAConfig,
) -> List[ProblemResult]:
    """Run RSA on all problems sequentially.

    Each problem is processed one at a time to keep GPU utilization manageable
    (each problem already generates N concurrent requests per step).
    """
    client = get_client(port=config.port)
    results = []
    for i, p in enumerate(problems):
        print(f"[RSA] Problem {i+1}/{len(problems)}: {p['id']}")
        r = await rsa_one_problem(
            client,
            problem_id=p["id"],
            problem_text=p["problem"],
            ground_truth=p["answer"],
            config=config,
        )
        # Print progress
        for step in r.steps:
            status = "correct" if step.majority_correct else "wrong"
            print(f"  Step {step.step}: acc={step.accuracy:.2f}, "
                  f"majority={step.majority_answer} ({status})")
        results.append(r)
    return results


def run_rsa(problems: List[dict], config: RSAConfig) -> List[ProblemResult]:
    """Synchronous entry point for RSA."""
    return asyncio.run(rsa_all_problems(problems, config))
