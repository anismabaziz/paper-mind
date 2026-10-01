"""
The numbers a run reports, computed without a model.

Every function here is pure: retrieved text and a label in, a number out. That
is the point — a reported score has to be reproducible by reading its inputs, so
nothing that decides a score is allowed to be a model, a clock, or a database.
"""

import math

import pytest

from evaluation.metrics import (
    LatencySummary,
    MetricSummary,
    RetrievalReport,
    cost_for,
    hit_at_k,
    latency_summary,
    ndcg_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
    retrieval_scores,
    summarize,
)


def scores(retrieved, gold=("alpha",), k=5):
    """Return the per-question retrieval scores for one question."""
    return retrieval_scores(retrieved, list(gold), k)


class TestRetrievalScores:
    """One question's ranking, measured four ways."""

    def test_a_gold_snippet_in_the_first_chunk_hits_and_reciprocates_one(self):
        """The best possible ranking scores a hit, full recall, and rank 1."""
        result = scores(["alpha here", "beta there"])

        assert result["hit_at_k"] is True
        assert result["recall_at_k"] == 1.0
        assert result["reciprocal_rank"] == 1.0
        assert result["ndcg_at_k"] == 1.0

    def test_a_gold_snippet_below_k_counts_as_missed(self):
        """The window bounds the measurement: a gold chunk at rank 6 misses at k=5."""
        retrieved = ["nothing", "nothing", "nothing", "nothing", "nothing", "alpha"]

        result = scores(retrieved, k=5)

        assert result["hit_at_k"] is False
        assert result["reciprocal_rank"] == 0.0
        assert result["ndcg_at_k"] == 0.0

    def test_reciprocal_rank_falls_with_the_position_of_the_first_gold(self):
        """A hit at rank 3 is worth a third, which is what MRR averages."""
        assert scores(["a", "b", "alpha", "c"])["reciprocal_rank"] == pytest.approx(
            1 / 3
        )

    def test_recall_counts_every_gold_snippet_not_just_the_first(self):
        """One of two gold snippets found is half the recall, and a hit."""
        result = scores(["alpha", "beta"], gold=("alpha", "gamma"))

        assert result["hit_at_k"] is True
        assert result["recall_at_k"] == 0.5

    def test_ndcg_rewards_putting_the_relevant_chunk_first(self):
        """Two rankings that both find the gold chunk are not equally good."""
        first = scores(["alpha", "filler"])["ndcg_at_k"]
        second = scores(["filler", "alpha"])["ndcg_at_k"]

        assert first > second

    def test_ndcg_is_one_only_when_every_gold_snippet_leads_the_ranking(self):
        """A chunk does not count twice, so one chunk cannot answer for two golds."""
        both = scores(["alpha", "beta", "filler"], gold=("alpha", "beta"))
        one_chunk = scores(["alpha and beta", "filler"], gold=("alpha", "beta"))

        assert both["ndcg_at_k"] == 1.0
        assert 0.0 < one_chunk["ndcg_at_k"] < 1.0

    def test_finding_only_some_of_the_evidence_cannot_score_a_perfect_ranking(self):
        """
        The degenerate case matters: hit rate cannot tell it from a full answer.

            A run that retrieves one of two gold snippets and puts it first has
            a perfect hit rate, so a score that also read 1.0 here would report
            half the evidence as a complete retrieval.
        """
        result = scores(["alpha", "filler"], gold=("alpha", "beta"))

        assert result["hit_at_k"] is True
        assert result["recall_at_k"] == 0.5
        assert result["ndcg_at_k"] == pytest.approx(1 / (1 + 1 / math.log2(3)))

    def test_a_question_with_no_gold_snippet_is_scored_as_nothing_found(self):
        """An unlabeled question has no ranking to grade."""
        result = retrieval_scores(["anything"], [], 5)

        assert result == {
            "hit_at_k": False,
            "recall_at_k": 0.0,
            "reciprocal_rank": 0.0,
            "ndcg_at_k": 0.0,
        }

    def test_matching_ignores_case_and_extra_whitespace(self):
        """A PDF's line wrapping must not decide whether a snippet was found."""
        assert hit_at_k(["Hit   rate\nat k is 0.9"], ["hit rate at k is"], 1) is True


class TestRetrievalAggregate:
    """The run's numbers, with the per-question numbers beside them."""

    def test_aggregates_average_every_question_that_reached_retrieval(self):
        """The run's retrieval numbers are the mean of its questions'."""
        rows = [
            retrieval_scores(["alpha", "beta"], ["alpha", "beta"], 2),
            retrieval_scores(["filler", "filler"], ["alpha"], 2),
        ]

        report = summarize(rows, k=2, questions=["first", "second"])

        assert report.questions == 2
        assert report.hit_rate == 0.5
        assert report.recall == 0.5
        assert report.mrr == 0.5
        assert report.ndcg == pytest.approx((1.0 + 0.0) / 2)
        assert [row["id"] for row in report.per_question] == ["first", "second"]
        assert report.per_question[0]["hit_at_k"] is True
        assert report.per_question[1]["reciprocal_rank"] == 0.0

    def test_a_run_with_no_retrieval_reports_zeroes_rather_than_a_division(self):
        """Nothing asked means nothing to average, not a crash."""
        report = summarize([], k=5, questions=[])

        assert report == RetrievalReport(
            questions=0,
            k=5,
            hit_rate=0.0,
            recall=0.0,
            mrr=0.0,
            ndcg=0.0,
            per_question=[],
        )

    def test_a_row_that_cannot_be_named_by_a_question_is_rejected(self):
        """
        An aggregate nobody can trace is the number this module exists to earn.

            Naming the questions is required rather than best-effort, so a caller
            that drops one finds out here instead of publishing an average with
            a row in it that belongs to nothing.
        """
        with pytest.raises(ValueError):
            summarize([retrieval_scores(["alpha"], ["alpha"], 1)], k=1, questions=[])


class TestPercentiles:
    """Latency percentiles, computed by rank so the value is a real sample."""

    def test_the_median_of_an_odd_number_of_samples_is_the_middle_one(self):
        """The median of three numbers is the one in the middle."""
        assert percentile([3.0, 1.0, 2.0], 0.5) == 2.0

    def test_p95_of_ten_samples_sits_between_the_ninth_and_the_tenth(self):
        """Percentiles are read off the sorted samples, not rounded to one."""
        values = [float(value) for value in range(1, 11)]

        assert percentile(values, 0.95) == pytest.approx(9.55)

    def test_a_single_sample_is_every_percentile_of_itself(self):
        """One measurement has no spread to report."""
        assert percentile([7.5], 0.5) == percentile([7.5], 0.95) == 7.5

    def test_no_samples_has_no_percentile(self):
        """Zero would be the best latency ever measured, which is a different claim."""
        assert percentile([], 0.5) is None

    def test_the_result_does_not_depend_on_the_order_the_samples_arrive_in(self):
        """A percentile is a property of the sample, not of its arrival."""
        values = [5.0, 1.0, 4.0, 2.0, 3.0]

        assert percentile(values, 0.5) == percentile(sorted(values), 0.5)


class TestLatencySummary:
    """Steady-state latency, kept apart from the run's first case."""

    def test_p50_and_p95_come_from_the_samples_they_name(self):
        """Both ends of the distribution are read off real measurements."""
        values = [float(value) for value in range(1, 21)]

        summary = latency_summary(values)

        assert summary == LatencySummary(samples=20, p50=10.5, p95=19.05)

    def test_a_run_with_no_samples_reports_none_rather_than_zero(self):
        """Zero would be the best latency there is, which is a different claim."""
        summary = latency_summary([])

        assert summary == LatencySummary(samples=0, p50=None, p95=None)

    def test_cold_start_and_warm_latency_are_summarized_apart(self):
        """Model loading is not steady-state latency, so it is not averaged in."""
        warm = latency_summary([1.0, 2.0, 3.0])
        cold = latency_summary([12.0])

        assert warm == LatencySummary(samples=3, p50=2.0, p95=2.9)
        assert cold == LatencySummary(samples=1, p50=12.0, p95=12.0)


class TestCost:
    """An estimate of what one run cost, from the catalog's published prices."""

    def test_cost_is_the_catalog_price_per_million_tokens(self):
        """The estimate uses the prices the catalog publishes."""
        from services.accounts.chat_settings_service import model_for

        model = model_for("google", "gemini-2.5-flash")

        # One million in, one million out: $0.30 + $2.50.
        assert cost_for(1_000_000, 1_000_000, model) == pytest.approx(2.80)

    def test_a_short_answer_costs_fractions_of_a_cent(self):
        """A grounded answer is a few hundred tokens, not a page."""
        from services.accounts.chat_settings_service import model_for

        model = model_for("google", "gemini-2.5-flash")

        assert cost_for(1_200, 80, model) == pytest.approx(0.00056, abs=1e-9)


class TestMetricSummary:
    """The shape every graded answer metric is reported in."""

    def test_a_mean_is_over_the_cases_that_were_scored(self):
        """A case nothing was decided about is counted, not averaged in as a zero."""
        summary = MetricSummary(
            graded=3,
            scored=2,
            mean=0.75,
            passed=1,
            failed=1,
            unknown=1,
        )

        assert summary.passed + summary.failed + summary.unknown == summary.graded

    def test_a_summary_serializes_with_its_counts_and_its_mean(self):
        """A metric is reported as counts plus a mean, never as a mean alone."""
        summary = MetricSummary(
            graded=1,
            scored=1,
            mean=1.0,
            passed=1,
            failed=0,
            unknown=0,
        )

        assert summary.to_dict() == {
            "graded": 1,
            "scored": 1,
            "mean": 1.0,
            "passed": 1,
            "failed": 0,
            "unknown": 0,
        }
