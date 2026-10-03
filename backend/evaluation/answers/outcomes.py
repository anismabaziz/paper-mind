"""
What a case can end as.

The vocabulary lives on its own because two modules need it and neither is
above the other. A case in the evaluation set declares the outcome it has to
come out as, and a run records the outcome a case did come out as. When those
two are the same list, a set cannot ask for something the run is unable to name.
"""

ANSWERED = "answered"
ABSTAINED = "abstained"
PROVIDER_ERROR = "provider_error"
CITATION_ERROR = "citation_error"
PERSISTENCE_ERROR = "persistence_error"
CANCELLED = "cancelled"

#: The question was not asked at all: the Document could not be asked about.
REFUSED = "refused"
#: The stored text is a failed call quoting the evidence back, not an answer.
CONTEXT_FALLBACK = "context_fallback"

#: Every way a case can end. A run reports all of them, because a case that
#: abstained is a different result from a case that answered badly, and neither
#: is the same as a case that never reached the model.
OUTCOMES = (
    ANSWERED,
    ABSTAINED,
    PROVIDER_ERROR,
    CITATION_ERROR,
    PERSISTENCE_ERROR,
    CANCELLED,
    REFUSED,
    CONTEXT_FALLBACK,
)

#: The outcomes a case in the evaluation set is allowed to require. The two a
#: question can honestly end as, and the two failures worth asserting that a
#: run reports rather than grades.
REQUIRED_OUTCOMES = (ANSWERED, ABSTAINED, PROVIDER_ERROR, CITATION_ERROR)
