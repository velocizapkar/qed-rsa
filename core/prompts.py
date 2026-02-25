"""Prompt templates for QED-Nano + RSA.

Aggregation prompt adapted from the RSA paper (arxiv 2509.26626) and the
official RSA repo (https://github.com/HyperPotatoNeo/RSA).
"""

from typing import List


SOLVE_SYSTEM = (
    "You are a helpful math assistant. "
    "Solve the problem step by step and put your final integer answer in \\boxed{}."
)


def solve_prompt(problem: str) -> List[dict]:
    """Build a chat-format prompt for initial solution generation."""
    return [
        {"role": "system", "content": SOLVE_SYSTEM},
        {"role": "user", "content": problem},
    ]


def aggregation_prompt(problem: str, candidates: List[str]) -> List[dict]:
    """Build a chat-format prompt that asks the model to aggregate K candidates.

    Follows the RSA paper's approach: present the problem and K candidate
    solutions, ask the model to produce a single improved solution.
    """
    parts: list[str] = []

    if len(candidates) == 1:
        parts.append(
            "You are given a math problem and a candidate solution. "
            "The candidate may be incomplete or contain errors. "
            "Refine this trajectory and produce an improved, higher-quality solution. "
            "If it is entirely wrong, attempt a new strategy. "
            "End with the final result in \\boxed{}.\n"
        )
    else:
        parts.append(
            "You are given a math problem and several candidate solutions. "
            "Some candidates may be incorrect or contain errors. "
            "Aggregate the useful ideas and produce a single, high-quality solution. "
            "Reason carefully; if candidates disagree, choose the correct path. "
            "If all are incorrect, then attempt a different strategy. "
            "End with the final result in \\boxed{}.\n"
        )

    parts.append("Problem:\n")
    parts.append(problem.strip() + "\n")

    if len(candidates) == 1:
        parts.append("Candidate solution (may contain mistakes):\n")
        parts.append(f"---- Candidate ----\n{candidates[0].strip()}\n")
        parts.append(
            "Now refine the candidate into an improved solution. "
            "Provide clear reasoning and end with the final answer in \\boxed{}."
        )
    else:
        parts.append("Candidate solutions (may contain mistakes):\n")
        for i, cand in enumerate(candidates, 1):
            parts.append(f"---- Solution {i} ----\n{cand.strip()}\n")
        parts.append(
            "Now write a single improved solution. "
            "Provide clear reasoning and end with the final answer in \\boxed{}."
        )

    user_content = "\n".join(parts)
    return [
        {"role": "system", "content": SOLVE_SYSTEM},
        {"role": "user", "content": user_content},
    ]
