"""Tests for new features: suites command, models command, expanded cost table."""

from __future__ import annotations

import subprocess
import sys

from click.testing import CliRunner

from bench_my_llm.cli import cli
from bench_my_llm.metrics import (
    COST_TABLE,
    DEFAULT_COST,
    _lookup_cost,
    estimate_cost,
    score_quality,
)


# ---------------------------------------------------------------------------
# CLI: suites command
# ---------------------------------------------------------------------------

class TestSuitesCommand:
    """Tests for the 'suites' CLI command."""

    def test_suites_exits_zero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["suites"])
        assert result.exit_code == 0

    def test_suites_lists_all_suites(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["suites"])
        for name in ("reasoning", "coding", "creative", "factual", "all"):
            assert name in result.output

    def test_suites_shows_descriptions(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["suites"])
        assert "Logic" in result.output or "reasoning" in result.output
        assert "Code" in result.output or "coding" in result.output

    def test_suites_shows_prompt_counts(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["suites"])
        # Each individual suite has 5 prompts
        assert "5" in result.output
        # "all" suite has 20
        assert "20" in result.output

    def test_suites_subprocess(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "bench_my_llm", "suites"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "reasoning" in result.stdout


# ---------------------------------------------------------------------------
# CLI: models command
# ---------------------------------------------------------------------------

class TestModelsCommand:
    """Tests for the 'models' CLI command."""

    def test_models_exits_zero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["models"])
        assert result.exit_code == 0

    def test_models_lists_known_models(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["models"])
        assert "gpt-4o" in result.output
        assert "claude-opus-4" in result.output
        assert "llama-4-maverick" in result.output

    def test_models_shows_pricing(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["models"])
        # Should have dollar signs for pricing
        assert "$" in result.output

    def test_models_shows_default_fallback(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["models"])
        assert "Unknown models" in result.output or "default" in result.output.lower()

    def test_models_subprocess(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "bench_my_llm", "models"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "gpt-4" in result.stdout


# ---------------------------------------------------------------------------
# Expanded cost table
# ---------------------------------------------------------------------------

class TestExpandedCostTable:
    """Tests for the expanded model cost table."""

    def test_gpt_4_1_pricing(self) -> None:
        inp, out = _lookup_cost("gpt-4.1")
        assert inp == 0.002
        assert out == 0.008

    def test_gpt_4_1_mini_pricing(self) -> None:
        inp, out = _lookup_cost("gpt-4.1-mini")
        assert inp == 0.0004
        assert out == 0.0016

    def test_gpt_4_1_nano_pricing(self) -> None:
        inp, out = _lookup_cost("gpt-4.1-nano")
        assert inp == 0.0001
        assert out == 0.0004

    def test_o3_pricing(self) -> None:
        inp, out = _lookup_cost("o3")
        assert inp == 0.01
        assert out == 0.04

    def test_o3_mini_pricing(self) -> None:
        inp, out = _lookup_cost("o3-mini")
        assert inp == 0.0011
        assert out == 0.0044

    def test_o4_mini_pricing(self) -> None:
        inp, out = _lookup_cost("o4-mini")
        assert inp == 0.0011
        assert out == 0.0044

    def test_o1_pricing(self) -> None:
        inp, out = _lookup_cost("o1")
        assert inp == 0.015
        assert out == 0.06

    def test_claude_opus_4_pricing(self) -> None:
        inp, out = _lookup_cost("claude-opus-4")
        assert inp == 0.015
        assert out == 0.075

    def test_claude_sonnet_4_pricing(self) -> None:
        inp, out = _lookup_cost("claude-sonnet-4")
        assert inp == 0.003
        assert out == 0.015

    def test_claude_3_5_sonnet_pricing(self) -> None:
        inp, out = _lookup_cost("claude-3.5-sonnet")
        assert inp == 0.003
        assert out == 0.015

    def test_claude_3_5_haiku_pricing(self) -> None:
        inp, out = _lookup_cost("claude-3.5-haiku")
        assert inp == 0.0008
        assert out == 0.004

    def test_llama_4_maverick_pricing(self) -> None:
        inp, out = _lookup_cost("llama-4-maverick")
        assert inp == 0.0005
        assert out == 0.0015

    def test_llama_4_scout_pricing(self) -> None:
        inp, out = _lookup_cost("llama-4-scout")
        assert inp == 0.00018
        assert out == 0.0005

    def test_mistral_large_pricing(self) -> None:
        inp, out = _lookup_cost("mistral-large")
        assert inp == 0.002
        assert out == 0.006

    def test_gemini_2_5_pro_pricing(self) -> None:
        inp, out = _lookup_cost("gemini-2.5-pro")
        assert inp == 0.00125
        assert out == 0.01

    def test_gemini_2_5_flash_pricing(self) -> None:
        inp, out = _lookup_cost("gemini-2.5-flash")
        assert inp == 0.00015
        assert out == 0.0006

    def test_deepseek_r1_pricing(self) -> None:
        inp, out = _lookup_cost("deepseek-r1")
        assert inp == 0.00055
        assert out == 0.0022

    def test_deepseek_v3_pricing(self) -> None:
        inp, out = _lookup_cost("deepseek-v3")
        assert inp == 0.00027
        assert out == 0.0011

    def test_unknown_model_uses_default(self) -> None:
        inp, out = _lookup_cost("totally-unknown-model-xyz")
        assert (inp, out) == DEFAULT_COST

    def test_case_insensitive_lookup(self) -> None:
        """Lookup should be case-insensitive."""
        inp, out = _lookup_cost("GPT-4.1-MINI")
        assert inp == 0.0004

    def test_estimate_cost_with_new_model(self) -> None:
        """Cost estimation works with the new models."""
        cost = estimate_cost("gpt-4.1-nano", prompt_tokens=1000, completion_tokens=500)
        expected = (1000 / 1000) * 0.0001 + (500 / 1000) * 0.0004
        assert abs(cost - expected) < 1e-6

    def test_cost_table_has_no_duplicates(self) -> None:
        """Each key in the cost table should be unique."""
        keys = list(COST_TABLE.keys())
        assert len(keys) == len(set(keys))

    def test_all_costs_positive(self) -> None:
        """All cost entries should be positive numbers."""
        for model, (inp, out) in COST_TABLE.items():
            assert inp > 0, f"{model} input cost should be positive"
            assert out > 0, f"{model} output cost should be positive"


# ---------------------------------------------------------------------------
# Quality scoring edge cases
# ---------------------------------------------------------------------------

class TestQualityScoringEdgeCases:
    """Edge cases for the quality scoring function."""

    def test_identical_strings(self) -> None:
        score = score_quality("the quick brown fox", "the quick brown fox")
        assert score == 1.0

    def test_completely_different(self) -> None:
        score = score_quality("alpha beta gamma", "one two three four")
        assert score == 0.0

    def test_partial_overlap(self) -> None:
        score = score_quality("the quick brown fox", "the slow brown cat")
        assert 0.0 < score < 1.0

    def test_empty_response_with_reference(self) -> None:
        score = score_quality("", "expected answer here")
        assert score == 0.0

    def test_whitespace_only_reference(self) -> None:
        score = score_quality("anything here", "   ")
        assert score == 1.0

    def test_punctuation_stripped(self) -> None:
        """Punctuation should not affect matching."""
        s1 = score_quality("hello, world!", "hello world")
        s2 = score_quality("hello world", "hello world")
        assert s1 == s2

    def test_case_insensitive(self) -> None:
        """Matching should be case-insensitive."""
        s1 = score_quality("The Quick Brown Fox", "the quick brown fox")
        assert s1 == 1.0
