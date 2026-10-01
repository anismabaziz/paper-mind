"""
The versions of a run that get compared to each other.

A comparison is only honest if the choices behind it were made away from the
numbers being reported, and if the configuration behind each column is written
down rather than being whatever the machine happened to be set to. These tests
are about those two properties: what an experiment is allowed to change, and
which split its choice may be made on.
"""

import pytest

from evaluation import experiments
from evaluation.dataset import REPORTED, TUNING
from settings import Settings


@pytest.fixture
def settings_obj():
    """Return the pinned settings the suite runs with."""
    import settings as settings_module

    return settings_module.get_settings()


def an_experiment(**changes):
    """Return an experiment with the given knobs set."""
    defaults = {
        "id": "variant",
        "family": "retrieval-method",
        "question": "Does this choice change what the reader gets?",
    }
    return experiments.Experiment(**{**defaults, **changes})


class TestTheBaseline:
    """The baseline is the configuration the application ships."""

    def test_the_baseline_changes_nothing(self, settings_obj):
        """The baseline is measured under the settings the app ships."""
        assert experiments.BASELINE.settings(settings_obj) == settings_obj

    def test_the_baseline_leaves_retrieval_to_the_application(self):
        """An unset knob is the app's own choice, not a second one."""
        assert experiments.BASELINE.retrieval() == {
            "method": None,
            "candidate_depth": None,
            "rerank": None,
            "query_expansion": None,
        }

    def test_the_baseline_needs_no_other_index(self):
        """The baseline runs against the index a normal run builds."""
        assert experiments.BASELINE.needs_fresh_index is False


class TestWhatAnExperimentMayChange:
    """An experiment changes one thing, and says which thing."""

    def test_a_variant_records_the_knobs_it_sets(self):
        """A report can publish the configuration behind a column."""
        experiment = an_experiment(
            family="chunking", chunk_size_tokens=256, chunk_overlap_tokens=25
        )

        assert experiment.knobs() == {
            "chunk_size_tokens": 256,
            "chunk_overlap_tokens": 25,
        }

    def test_a_variant_that_changes_nothing_is_refused(self):
        """A column that differs nowhere is the baseline's column."""
        with pytest.raises(ValueError, match="changes nothing"):
            an_experiment()

    def test_a_variant_is_not_mislabelled_by_its_family(self):
        """A chunking change under a retrieval label would misread the table."""
        with pytest.raises(ValueError, match="retrieval-method"):
            an_experiment(chunk_size_tokens=256)

    def test_an_unknown_family_is_refused(self):
        """Only the declared families can be compared."""
        with pytest.raises(ValueError, match="family"):
            an_experiment(family="vibes", method="dense")


class TestWhichSplitAVariantWasChosenOn:
    """A choice is made on the tuning split, never on the reported one."""

    def test_a_variant_chosen_on_the_reported_split_is_refused(self):
        """The reported split is quoted from, so it cannot also be tuned on."""
        with pytest.raises(ValueError, match="validation"):
            an_experiment(decided_on=REPORTED, method="dense")

    def test_a_variant_records_the_split_it_was_chosen_on(self):
        """Every experiment says where its value was picked."""
        assert an_experiment(method="dense").decided_on == TUNING


class TestTheRegistry:
    """The registry is the versioned set of experiments a report compares."""

    def test_every_retrieval_method_is_measured(self):
        """Dense, sparse, and hybrid each get a column of their own."""
        methods = {
            experiment.retrieval()["method"]
            for experiment in experiments.REGISTRY
            if experiment.family == "retrieval-method"
        }

        assert methods == {"dense", "sparse", "hybrid"}

    def test_chunking_candidate_depth_and_expansion_are_compared(self):
        """The four adjustable families are all present."""
        families = {
            experiment.family
            for experiment in experiments.REGISTRY
            if experiment.family != "baseline"
        }

        assert families == {
            "retrieval-method",
            "rerank",
            "candidate-depth",
            "query-expansion",
            "chunking",
        }

    def test_the_baseline_is_the_first_experiment(self):
        """The comparison is read against the first column."""
        assert experiments.REGISTRY[0] is experiments.BASELINE

    def test_every_experiment_has_its_own_id(self):
        """Ids are the filenames results are stored under."""
        ids = [experiment.id for experiment in experiments.REGISTRY]

        assert len(ids) == len(set(ids))

    def test_a_registry_with_a_repeated_id_is_refused(self):
        """Two columns answering the same question cannot be read."""
        with pytest.raises(ValueError, match="variant"):
            experiments.validated([experiments.BASELINE, experiments.BASELINE])

    def test_a_chunking_experiment_indexes_the_documents_again(self, settings_obj):
        """Different chunks are different vectors, so they are reindexed."""
        chunking = experiments.experiment("chunk-256-25")

        assert chunking.needs_fresh_index is True
        assert chunking.settings(settings_obj).chunking.chunk_size_tokens == 256
        assert chunking.settings(settings_obj).chunking.chunk_overlap_tokens == 25

    def test_a_retrieval_experiment_reuses_the_index_it_has(self, settings_obj):
        """Retrieval arguments are measured against the existing index."""
        reranking = experiments.experiment("rerank-on")

        assert reranking.needs_fresh_index is False
        assert reranking.settings(settings_obj) == settings_obj
