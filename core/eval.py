"""AIME answer extraction and evaluation.

AIME answers are integers 000-999. We extract from \\boxed{} and do exact
integer comparison.
"""

import re
from typing import Optional, List
from collections import Counter


def extract_boxed(text: str) -> Optional[str]:
    """Extract the content of the last \\boxed{} in text.

    QED-Nano is a thinking model that produces <think>...</think> blocks.
    We look for the last \\boxed{} which is the final answer after reasoning.
    Handles nested braces.
    """
    # Find all \boxed{ positions
    pattern = r"\\boxed\{"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None

    # Take the last match and extract content handling nested braces
    last = matches[-1]
    start = last.end()  # position after \boxed{
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    if depth != 0:
        return None

    return text[start : i - 1].strip()


def parse_integer_answer(boxed_content: str) -> Optional[int]:
    """Parse an integer from boxed content.

    Handles formats like "42", "042", "7", possibly with surrounding whitespace
    or dollar signs.
    """
    if boxed_content is None:
        return None
    # Strip LaTeX math delimiters and whitespace
    cleaned = boxed_content.strip().strip("$").strip()
    # Remove leading zeros but keep "0" as 0
    try:
        return int(cleaned)
    except ValueError:
        # Try to find a number in the string
        m = re.search(r"-?\d+", cleaned)
        if m:
            return int(m.group())
        return None


def extract_answer(completion: str) -> Optional[int]:
    """Extract an integer answer from a model completion."""
    boxed = extract_boxed(completion)
    if boxed is None:
        return None
    return parse_integer_answer(boxed)


def is_correct(completion: str, ground_truth: int) -> bool:
    """Check if the model's answer matches the ground truth."""
    answer = extract_answer(completion)
    if answer is None:
        return False
    return answer == ground_truth


def majority_vote(completions: List[str]) -> Optional[int]:
    """Return the most common extracted answer from a list of completions."""
    answers = []
    for c in completions:
        a = extract_answer(c)
        if a is not None:
            answers.append(a)
    if not answers:
        return None
    counter = Counter(answers)
    return counter.most_common(1)[0][0]


def population_accuracy(completions: List[str], ground_truth: int) -> float:
    """Fraction of population members with the correct answer (pass@1 estimate)."""
    if not completions:
        return 0.0
    correct = sum(1 for c in completions if is_correct(c, ground_truth))
    return correct / len(completions)


def pass_at_n(completions: List[str], ground_truth: int) -> bool:
    """Did any member of the population get the correct answer?"""
    return any(is_correct(c, ground_truth) for c in completions)
