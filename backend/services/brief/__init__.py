"""
The Research Brief: a bounded, tool-using run over two Documents.

A chat answer retrieves once and answers from what it found. A brief lets the
model choose what to read next, within a pair of Documents it was given and a
budget it cannot exceed. The pieces are separate because each is a separate
decision: the scope is what the run may reach, the tools are how, the budget is
how much it may spend, and the ledger is what it ended up holding.
"""

from services.brief.budget import BriefBudget, BriefLimits
from services.brief.evidence import Evidence, EvidenceLedger
from services.brief.scope import (
    BRIEF_DOCUMENT_COUNT,
    BriefScope,
    ScopedDocument,
    resolve_scope,
)
from services.brief.service import BriefRequest, BriefService, ResolvedBrief
