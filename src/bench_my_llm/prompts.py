"""Built-in prompt suites for benchmarking."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Prompt:
    """A single benchmark prompt with optional reference answer."""

    text: str
    category: str
    reference: str = ""
    max_tokens: int = 512


@dataclass
class PromptSuite:
    """A named collection of prompts."""

    name: str
    description: str
    prompts: list[Prompt] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Reasoning suite
# ---------------------------------------------------------------------------
REASONING = PromptSuite(
    name="reasoning",
    description="Logic, math, and step-by-step reasoning tasks",
    prompts=[
        Prompt(
            text="A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left? Explain your reasoning step by step.",
            category="reasoning",
            reference="9 sheep. 'All but 9' means 9 remain.",
        ),
        Prompt(
            text="If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets? Think carefully.",
            category="reasoning",
            reference="5 minutes. Each machine makes 1 widget in 5 minutes.",
        ),
        Prompt(
            text="I have a 3-gallon jug and a 5-gallon jug. How do I measure exactly 4 gallons of water? Show your steps.",
            category="reasoning",
            reference="Fill 5-gallon, pour into 3-gallon (leaves 2 in 5-gallon), empty 3-gallon, pour 2 into 3-gallon, fill 5-gallon, pour from 5 into 3-gallon (1 gallon fits), 5-gallon now has 4.",
        ),
        Prompt(
            text="Three friends split a $30 dinner bill equally. They each pay $10 to the waiter. The waiter realizes the bill is only $25 and returns $5. They each take $1 back, tipping $2 total. They each paid $9 (totaling $27) plus $2 tip = $29. Where is the missing dollar? Explain.",
            category="reasoning",
            reference="There is no missing dollar. The $27 paid includes the $25 bill plus $2 tip. Adding the $2 tip to $27 is double-counting.",
        ),
        Prompt(
            text="What is the next number in the sequence: 1, 1, 2, 3, 5, 8, 13, ...? Explain the pattern.",
            category="reasoning",
            reference="21. This is the Fibonacci sequence where each number is the sum of the two preceding numbers.",
        ),
    ],
)

# ---------------------------------------------------------------------------
# Coding suite
# ---------------------------------------------------------------------------
CODING = PromptSuite(
    name="coding",
    description="Code generation and explanation tasks",
    prompts=[
        Prompt(
            text="Write a Python function that checks if a string is a palindrome. Include type hints and handle edge cases.",
            category="coding",
            reference="def is_palindrome(s: str) -> bool: cleaned = ''.join(c.lower() for c in s if c.isalnum()); return cleaned == cleaned[::-1]",
        ),
        Prompt(
            text="Implement binary search in Python. The function should return the index of the target or -1 if not found. Include type hints.",
            category="coding",
            reference="def binary_search(arr: list[int], target: int) -> int",
        ),
        Prompt(
            text="Write a Python function to flatten a nested list of arbitrary depth. For example, [1, [2, [3, 4], 5]] becomes [1, 2, 3, 4, 5].",
            category="coding",
            reference="def flatten(lst): result = []; for item in lst: result.extend(flatten(item) if isinstance(item, list) else [item]); return result",
        ),
        Prompt(
            text="Write a Python decorator that retries a function up to N times with exponential backoff if it raises an exception.",
            category="coding",
            reference="A decorator using functools.wraps with a loop, try/except, and time.sleep(2**attempt) logic.",
        ),
        Prompt(
            text="Explain the difference between a list and a tuple in Python. When should you use each? Give code examples.",
            category="coding",
            reference="Lists are mutable (append, remove), tuples are immutable. Use tuples for fixed collections, dict keys, and function returns.",
        ),
    ],
)

# ---------------------------------------------------------------------------
# Creative suite
# ---------------------------------------------------------------------------
CREATIVE = PromptSuite(
    name="creative",
    description="Creative writing and storytelling tasks",
    prompts=[
        Prompt(
            text="Write a haiku about debugging code at 3 AM.",
            category="creative",
            max_tokens=128,
        ),
        Prompt(
            text="In exactly 50 words, tell a complete story with a beginning, middle, and end about a robot learning to paint.",
            category="creative",
            max_tokens=256,
        ),
        Prompt(
            text="Write a product description for a time machine that fits in your pocket. Make it sound like an Apple product launch.",
            category="creative",
        ),
        Prompt(
            text="Create a short dialogue between a semicolon and an em dash arguing about which punctuation mark is more useful.",
            category="creative",
        ),
        Prompt(
            text="Write 3 metaphors that explain how a neural network learns, aimed at a 10-year-old audience.",
            category="creative",
            max_tokens=256,
        ),
    ],
)

# ---------------------------------------------------------------------------
# Factual suite
# ---------------------------------------------------------------------------
FACTUAL = PromptSuite(
    name="factual",
    description="Factual recall and knowledge tasks",
    prompts=[
        Prompt(
            text="What are the three laws of thermodynamics? Explain each in one sentence.",
            category="factual",
            reference="1st: Energy cannot be created or destroyed. 2nd: Entropy of an isolated system always increases. 3rd: Entropy approaches zero as temperature approaches absolute zero.",
        ),
        Prompt(
            text="List the planets in our solar system in order from the Sun, and state which ones are gas giants.",
            category="factual",
            reference="Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune. Gas giants: Jupiter, Saturn. Ice giants: Uranus, Neptune.",
        ),
        Prompt(
            text="Who invented the World Wide Web, in what year, and at which institution?",
            category="factual",
            reference="Tim Berners-Lee, 1989, CERN.",
        ),
        Prompt(
            text="What is the Big O time complexity of quicksort in the average case and the worst case? Explain why.",
            category="factual",
            reference="Average: O(n log n) with good pivot selection. Worst: O(n^2) when pivot is always the smallest or largest element.",
        ),
        Prompt(
            text="Explain what DNS is and describe what happens when you type a URL into a browser, in 5 steps or fewer.",
            category="factual",
            reference="DNS translates domain names to IP addresses. Steps: browser checks cache, queries DNS resolver, resolver queries root/TLD/authoritative servers, IP returned, browser connects via HTTP/HTTPS.",
        ),
    ],
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
SUITES: dict[str, PromptSuite] = {
    "reasoning": REASONING,
    "coding": CODING,
    "creative": CREATIVE,
    "factual": FACTUAL,
}

ALL_SUITE = PromptSuite(
    name="all",
    description="All built-in prompts combined",
    prompts=[p for suite in SUITES.values() for p in suite.prompts],
)
SUITES["all"] = ALL_SUITE


def get_suite(name: str) -> PromptSuite:
    """Get a prompt suite by name. Raises KeyError if not found."""
    if name not in SUITES:
        available = ", ".join(SUITES.keys())
        raise KeyError(f"Unknown suite '{name}'. Available: {available}")
    return SUITES[name]
