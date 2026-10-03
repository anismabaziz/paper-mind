# Grading follows the retrieval word set; compare stays independent

Chosen: one canonical stopword set and word splitter shared by sparse
retrieval and grading, while evidence comparison keeps its own.

## Grading adopts the retrieval set verbatim

Sparse retrieval and grading each filtered common words with their own list,
nine words apart despite a comment saying to keep them in sync. A Passage
that retrieval ignores for being all stopwords should be a Passage grading
ignores too, so both now read the same set and the same splitter from one
module. The alternative — a union set — would change indexed sparse vectors
and force every existing index stale through the tokenizer version. Nine
words are not worth a reindex wave, so the set is the retrieval one
unchanged: no version bump, no stale indexes, and a small accepted shift in
grading overlap. Token counting (chunking and context budgets) shares one
`tiktoken` encoding for the same reason: two instances can only ever agree
by accident.

## Topic-term extraction stays independent

Evidence comparison keeps its own larger stopword set and its own splitter.
Its exclusions are load-bearing for contrast detection: negation and
contrast cues must stay visible in the term arithmetic or cue-carrying pairs
are never compared. Merging it into the shared set would trade a working
comparison for a tidier import, so it is explicitly out of the shared
module and must not be re-suggested on tidiness grounds.

Rejected: a union word set with a tokenizer version bump. Correct in the
abstract, but the price is every Document in every library going stale at
once for a change no reader could observe.

Rejected: folding the comparison set into the shared one. The comparison
needs words the shared set removes, so one set cannot serve both without
lying to one of them.
