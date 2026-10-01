import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowUp, ChevronDown, CornerDownLeft, Loader2, RefreshCw, RotateCw, Settings, FileText } from "lucide-react";
import { chatStream, classifyChatRequestError, isSettingsFailureMessage, type ChatAbstentionReason, type ChatOutcomeFailure, type IClaim, type IRetrievalResult, type ISource } from "@/services/files";
import { useFileStatus, useFileMessages, useReindex } from "@/hooks/useFiles";
import { MarkdownRenderer } from "./MarkdownRenderer";
import { FailureNotice } from "./FailureNotice";
import usePdfStore from "@/store/pdf-state";
import useSettingsUi from "@/store/settings-ui";
import { cn } from "@/lib/utils";
import { isIndexStale, isIngestionActive, type DocumentIndex } from "@/types/db";

type ChatMessage = {
  id: string;
  text: string;
  sender: "user" | "bot";
  sources?: ISource[];
  claims?: IClaim[];
  /** At least one claim named a passage the app actually supplied. */
  grounded?: boolean;
  retrieval?: IRetrievalResult;
  failed?: boolean;
  needsSettings?: boolean;
  /** The answer was cut short by the model's output budget. */
  truncated?: boolean;
  /** True once the user walked away from this answer. */
  cancelled?: boolean;
  /** Why the answer was stopped, when the server said. */
  stopReason?: string;
  /** Which outcome ended this answer, for styling and for tests. */
  failure?: ChatOutcomeFailure;
  /** The app answered without a model call: there was no evidence. */
  abstained?: boolean;
  /** Why it abstained, which decides what the reader can do next. */
  abstentionReason?: ChatAbstentionReason;
  /** The question this answer was started for, so a failure can offer a retry. */
  question?: string;
};

/**
 * What the reader can do about an abstention.
 *
 * Nothing to read is not a fault: the paper simply does not cover the question.
 * Matches that could not be used are a different case, and a reindex is the
 * only thing that clears them.
 */
const ABSTENTION_HINTS: Record<ChatAbstentionReason, string> = {
  no_evidence: "Nothing in this paper answers that. Try a different question, or one of the prompts above.",
  evidence_unusable: "The passages this question matched could not be read. Reindex the paper to search it again.",
};

/** How an answer ended, in one place: the machine-readable outcome and the word for it. */
type Outcome = { outcome: string | undefined; label: string };

const GROUNDED: Outcome = { outcome: undefined, label: "grounded" };

/** Retrieval scores are a ranking, not a chance of being right, so only a claim that cites earns this. */
const UNGROUNDED: Outcome = { outcome: "ungrounded", label: "ungrounded" };

/** Nothing has been decided yet: no claim has been read, so nothing is claimed. */
const ANSWERING: Outcome = { outcome: "answering", label: "answering" };

function describe(m: ChatMessage): Outcome {
  if (m.failure) return { outcome: m.failure, label: "failed" };
  if (m.cancelled) return { outcome: "cancelled", label: "stopped" };
  if (m.abstained) return { outcome: m.abstentionReason, label: "abstained" };
  if (m.truncated) return { outcome: "truncated", label: "truncated" };
  // The terminal event is what settles whether anything is cited, so an answer
  // still arriving is not called grounded before it is.
  if (m.sources === undefined) return ANSWERING;
  if (m.grounded === false) return UNGROUNDED;
  return GROUNDED;
}

/** Settings failures are decided once, where the refusal reasons are named. */
const isSettingsError = isSettingsFailureMessage;

/**
 * What each failure means and what the reader can do next.
 *
 * Every failed answer keeps this shape: a heading the server did not write,
 * the server's safe message, one recovery action, and no citation list. A
 * failed answer never renders as grounded, however much text arrived first.
 */
const FAILURE_COPY: Record<ChatOutcomeFailure, { heading: string; hint: string; retryLabel: string }> = {
  provider: {
    heading: "The model could not answer",
    hint: "The paper was searched but the model call failed. Your question is kept — try it again.",
    retryLabel: "Ask again",
  },
  timeout: {
    heading: "The answer took too long",
    hint: "The model call was stopped before it finished. A narrower question usually completes faster.",
    retryLabel: "Try again",
  },
  empty_output: {
    heading: "The model returned nothing",
    hint: "The model call finished with no text. Rephrase the question and try again.",
    retryLabel: "Try again",
  },
  citations: {
    heading: "The answer cited evidence it was not given",
    hint: "The model named a passage outside what was retrieved, so the answer was withheld rather than shown ungrounded. Ask again — nothing here is cited.",
    retryLabel: "Ask again",
  },
  retrieval_unavailable: {
    heading: "Search is unavailable",
    hint: "The paper could not be searched right now, so no model was asked. Wait a moment, then try the question again.",
    retryLabel: "Retry search",
  },
  persistence: {
    heading: "The answer could not be saved",
    hint: "The answer arrived but storing it failed, so this Conversation does not include it. Ask the question again to store a fresh answer.",
    retryLabel: "Ask again",
  },
  interrupted: {
    heading: "The answer was interrupted",
    hint: "The connection broke before the answer finished. What arrived is not saved as a complete answer — try the question again.",
    retryLabel: "Try again",
  },
  settings: {
    heading: "No chat provider is configured",
    hint: "The question was never sent to a model. Verify a provider and API key in App Settings, then ask the question again.",
    retryLabel: "Ask again",
  },
};

const suggestedPrompts = [
  "What is the main topic?",
  "Summarize key findings",
  "Methodology used?",
];

function formatManifestValue(value: string | number | boolean | null): string {
  if (typeof value === "boolean") return value ? "on" : "off";
  return value === null || value === "" ? "unpinned" : String(value);
}

function StaleIndexNotice({ index }: { index: DocumentIndex }) {
  const file = usePdfStore((s) => s.file);
  const reindex = useReindex();
  return (
    <div role="status" className="rounded-sm border border-destructive/30 bg-destructive/5 p-5" data-testid="stale-index-notice">
      <div className="flex items-center gap-2">
        <RefreshCw className="size-3.5 text-destructive" />
        <h4 className="font-mono text-[0.68rem] font-semibold uppercase tracking-widest text-destructive">
          Index needs reindexing
        </h4>
      </div>
      <p className="mt-2 font-serif text-xs leading-relaxed text-ink-soft">
        These settings changed after this paper was indexed, so its passages no longer match how
        the app searches. Reindex to ask questions again.
      </p>
      <dl className="mt-3 space-y-1.5">
        {index.change_details.map((change) => (
          <div key={change.field} className="flex flex-wrap items-baseline gap-x-2 text-[0.65rem]">
            <dt className="font-mono text-ink-faint">{change.label}</dt>
            <dd className="text-ink-soft">
              {change.indexed === null
                ? "not recorded"
                : `${formatManifestValue(change.indexed)} → ${formatManifestValue(change.current)}`}
            </dd>
          </div>
        ))}
      </dl>
      <p className="label-meta mt-3">
        Active index generation {index.manifest?.index_generation ?? "?"}
      </p>
      {reindex.isError && (
        <p className="mt-2 text-[0.65rem] text-destructive">
          {reindex.error instanceof Error ? reindex.error.message : "The reindex could not be queued."}
        </p>
      )}
      <button
        type="button"
        onClick={() => file && reindex.mutate(file.name)}
        disabled={reindex.isPending || !file}
        className="mt-3 inline-flex items-center gap-1.5 border border-ink bg-ink px-3 py-1.5 font-mono text-[0.65rem] text-paper hover:bg-ink/90 disabled:opacity-40"
        data-testid="chat-reindex"
      >
        {reindex.isPending ? <Loader2 className="size-3 animate-spin" /> : <RefreshCw className="size-3" />}
        {reindex.isPending ? "Queueing reindex…" : "Reindex this paper"}
      </button>
    </div>
  );
}

/** A claim's citation, which opens the page that supports it when it has one. */
function ClaimCitation({ source }: { source: ISource }) {
  const setCitationTarget = usePdfStore((s) => s.setCitationTarget);
  const hasPage = source.page != null;
  return (
    <button
      type="button"
      disabled={!hasPage}
      onClick={() => hasPage && setCitationTarget(source.page as number)}
      title={hasPage ? `Open page ${source.page}` : "This passage has no page to open"}
      aria-label={
        hasPage
          ? `Open page ${source.page} for citation ${source.source_id}`
          : `Citation ${source.source_id}, no page to open`
      }
      data-testid={`citation-${source.source_id}`}
      className={cn(
        "font-mono text-[0.62rem] tracking-wide",
        hasPage ? "cursor-pointer text-marker hover:underline" : "text-ink-faint",
      )}
    >
      {source.source_id}
    </button>
  );
}

function ClaimList({ claims, sources }: { claims: IClaim[]; sources: ISource[] }) {
  const byId = new Map(sources.map((source) => [source.source_id, source]));
  return (
    <div className="mt-4 border-t border-rule pt-3">
      <p className="label-meta">Claims this answer makes</p>
      <ol className="mt-2.5 space-y-1.5">
        {claims.map((claim, index) => (
          <li key={index} className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
            <span className="font-serif text-[0.85rem] leading-snug text-ink-soft">{claim.claim}</span>
            {claim.sources.length > 0 ? (
              <span className="flex items-baseline gap-1">
                {claim.sources.map((sourceId) => {
                  const source = byId.get(sourceId);
                  return source ? (
                    <ClaimCitation key={sourceId} source={source} />
                  ) : null;
                })}
              </span>
            ) : (
              <span className="font-mono text-[0.6rem] text-ink-faint">not supported</span>
            )}
          </li>
        ))}
      </ol>
    </div>
  );
}

const METHOD_LABELS: Record<IRetrievalResult["method"], string> = {
  dense: "dense retrieval",
  sparse: "sparse retrieval",
  hybrid: "hybrid retrieval",
};

function SourceList({ sources, retrieval }: { sources: ISource[]; retrieval?: IRetrievalResult }) {
  const [open, setOpen] = useState(true);
  const setCitationTarget = usePdfStore((s) => s.setCitationTarget);
  const listId = `source-list-${sources[0]?.source_id ?? "empty"}`;
  return (
    <div className="mt-4 border-t border-rule pt-3">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-controls={listId}
        aria-label={`${open ? "Hide" : "Show"} ${sources.length} cited passages`}
        className="flex w-full items-center justify-between text-ink-faint hover:text-ink"
      >
        <span className="label-meta">
          {sources.length} passages{retrieval ? ` · ${METHOD_LABELS[retrieval.method]}` : ""}
        </span>
        <ChevronDown className={cn("size-3 transition-transform", open && "rotate-180")} />
      </button>
      {open && (
        <ol id={listId} className="mt-3 space-y-3">
          {sources.map((s, idx) => {
            const hasPage = s.page != null;
            const citationButton = (
              <span className="font-mono text-[0.62rem] tracking-wide text-ink-faint">
                <span className="text-marker">[{s.source_id}]</span> {s.document} · chunk {s.chunk_index}
                {hasPage ? ` · p. ${s.page}` : ""}
              </span>
            );
            return (
              <li key={idx}>
                <button
                  type="button"
                  disabled={!hasPage}
                  data-testid={`source-jump-${s.source_id}`}
                  onClick={() => {
                    if (hasPage) setCitationTarget(s.page);
                  }}
                  className={cn(
                    "group block w-full border-l-2 py-0.5 pl-3 text-left transition-colors",
                    hasPage ? "border-rule hover:border-marker cursor-pointer" : "border-rule cursor-default",
                  )}
                  title={hasPage ? `Jump to page ${s.page}` : undefined}
                >
                  <span className="flex items-baseline justify-between gap-2">
                    {citationButton}
                    <span className="font-mono text-[0.6rem] text-ink-faint">
                      Rank {s.rank ?? idx + 1}
                    </span>
                  </span>
                  <span className="mt-1 block font-serif text-[0.85rem] leading-snug text-ink-soft italic">“{s.content.slice(0, 220)}”</span>
                </button>
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}

function OpenSettingsLink() {
  const openSettings = useSettingsUi((s) => s.open);
  return (
    <button
      onClick={openSettings}
      className="mt-2 inline-flex items-center gap-1 font-mono text-[0.62rem] font-semibold tracking-wide text-destructive underline underline-offset-2 hover:text-destructive/80"
    >
      <Settings className="size-3" /> Open Settings
    </button>
  );
}

export function ChatPane() {
  const { file } = usePdfStore();
  const [value, setValue] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [thinking, setThinking] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const stickToBottomRef = useRef(true);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const streamControllerRef = useRef<AbortController | null>(null);
  // Identity of the document a stream was started for. Late tokens from a
  // previous document are ignored so they never land in the wrong
  // conversation, even if the abort races the next chunk.
  const fileIdRef = useRef<string | null>(null);

  const checkProcessedQuery = useFileStatus(file);
  const ingestionJob = checkProcessedQuery.data?.ingestion;
  const documentIndex = checkProcessedQuery.data?.index;
  const isStaleIndex = isIndexStale(documentIndex);
  const isIndexing = isIngestionActive(ingestionJob?.state);
  const messagesQuery = useFileMessages(
    file,
    checkProcessedQuery.data?.is_processed === true && !isIndexing
  );
  const reindex = useReindex();
  const watchedFileId = file?.id ?? null;

  useEffect(() => {
    fileIdRef.current = watchedFileId;
  }, [watchedFileId]);

  useEffect(() => {
    return () => {
      streamControllerRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    streamControllerRef.current?.abort();
    streamControllerRef.current = null;
  }, [watchedFileId]);

  // Drop the previous conversation the moment the document changes instead
  // of flashing its history until the new query resolves.
  useEffect(() => {
    setMessages([]);
    stickToBottomRef.current = true;
  }, [watchedFileId]);

  useEffect(() => {
    if (watchedFileId && messagesQuery.data?.messages) {
      setMessages(
        messagesQuery.data.messages.map((m) => ({
          id: m.id,
          text: m.text,
          sender: m.sender,
          sources: m.sources,
          claims: m.claims,
          // An answer is only as grounded as the claims that cite something:
          // a stored answer whose claims name no passage is not grounded.
          grounded:
            m.sender === "bot" && m.turn_status === "answered"
              ? (m.claims ?? []).some((claim) => claim.sources.length > 0)
              : undefined,
          // An abstention stored in history is still an abstention, not an
          // answer that came back empty.
          abstained: m.turn_status === "abstained",
          abstentionReason: m.turn_abstention_reason ?? undefined,
        })),
      );
    }
  }, [watchedFileId, messagesQuery.data]);

  useEffect(() => {
    if (stickToBottomRef.current) endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, thinking]);

  useEffect(() => {
    inputRef.current?.focus();
  }, [file]);

  const send = useCallback(
    async (text: string) => {
    const body = text.trim();
    if (!body || !file || thinking || isIndexing) return;
    setValue("");
    setThinking(true);
    const botId = crypto.randomUUID();
    setMessages((m) => [...m, { id: crypto.randomUUID(), text: body, sender: "user" }, { id: botId, text: "", sender: "bot", question: body }]);

    streamControllerRef.current?.abort();
    const controller = new AbortController();
    streamControllerRef.current = controller;
    const activeFileId = file.id;
    // Late chunks from a superseded stream are dropped so tokens never land
    // in the wrong conversation, even if the abort races the next chunk.
    const isStale = () => fileIdRef.current !== activeFileId || controller.signal.aborted;

    try {
      await chatStream(
        body,
        file.name,
        {
          onToken: (t) => {
            if (isStale()) return;
            setMessages((prev) => prev.map((msg) => (msg.id === botId ? { ...msg, text: msg.text + t } : msg)));
          },
          onProviderError: (message, category) => {
            if (isStale()) return;
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === botId
                  ? { ...msg, text: message, failed: true, failure: category, needsSettings: isSettingsError(message) }
                  : msg,
              ),
            );
          },
          onPersistenceError: (message) => {
            if (isStale()) return;
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === botId ? { ...msg, text: message, failed: true, failure: "persistence" } : msg,
              ),
            );
          },
          onCancelled: (reason) => {
            if (isStale()) return;
            // The answer the user already read is not thrown away because it
            // was never stored. It is marked as not saved, the same as a stop.
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === botId
                  ? {
                      ...msg,
                      cancelled: true,
                      stopReason: reason,
                      text: msg.text
                        ? `${msg.text}\n\n_Stopped before this answer was saved._`
                        : "",
                    }
                  : msg,
              ),
            );
          },
          onAbstained: ({ message, reason, retrieval }) => {
            if (isStale()) return;
            // Not a failure and not an empty answer: the app knows there is
            // nothing to ground one in, so it says so and cites nothing.
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === botId
                  ? { ...msg, text: message, abstained: true, abstentionReason: reason, retrieval }
                  : msg,
              ),
            );
          },
          onDone: ({ sources, claims, grounded, retrieval, truncated }) => {
            if (isStale()) return;
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === botId ? { ...msg, sources, claims, grounded, retrieval, truncated } : msg,
              ),
            );
          },
        },
        { signal: controller.signal },
      );
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") {
        // The user left this answer. The server recorded the turn as
        // cancelled, so what is on screen is not what history will replay —
        // it is marked stopped rather than presented as a stored answer.
        setMessages((prev) =>
          prev.flatMap((msg) => {
            if (msg.id !== botId) return [msg];
            if (!msg.text) return [];
            return [
              {
                ...msg,
                cancelled: true,
                text: `${msg.text}\n\n_Stopped before this answer was saved._`,
              },
            ];
          }),
        );
        setThinking(false);
        return;
      }
      if (isStale()) {
        setThinking(false);
        return;
      }
      // A rejected request is never an empty answer: classify it so a dead
      // search, missing App Settings, and a broken stream each read
      // differently and offer the right recovery.
      const { failure, message, needsSettings } = classifyChatRequestError(e);
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === botId
            ? { ...msg, text: message, failed: true, failure, needsSettings }
            : msg,
        ),
      );
    } finally {
      setThinking(false);
      inputRef.current?.focus();
    }
    },
    [file, thinking, isIndexing],
  );

  const isProcessed = checkProcessedQuery.data?.is_processed;
  const inputDisabled = !file || !isProcessed || isIndexing || isStaleIndex;

  return (
    <section className="flex w-full min-w-0 shrink-0 flex-col border-l border-rule bg-background lg:w-[26rem]" data-testid="chat-pane">
      <header className="flex h-14 items-center border-b border-rule px-5">
        <div>
          <p className="text-[0.82rem] font-medium">Reading companion</p>
          <p className="label-meta">Answers cite this paper only</p>
        </div>
      </header>

      <div
        ref={scrollContainerRef}
        data-testid="chat-messages"
        onScroll={() => {
          const el = scrollContainerRef.current;
          if (!el) return;
          stickToBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
        }}
        className="scroll-slim flex-1 space-y-7 overflow-y-auto px-5 py-6"
      >
        {!file && (
          <div className="flex flex-col items-center py-16 text-center">
            <div className="grid size-10 place-items-center border border-rule bg-paper text-ink-faint">
              <FileText className="size-4" />
            </div>
            <h4 className="mt-4 font-serif text-sm font-medium">No Active Session</h4>
            <p className="mt-1 max-w-[22ch] font-serif text-xs leading-relaxed text-ink-faint">
              Select a research paper from your library to start an interactive analysis session.
            </p>
          </div>
        )}

        {file && checkProcessedQuery.isError && !checkProcessedQuery.data && (
          <FailureNotice
            testId="chat-status-error"
            title="Could not check this paper"
            message={`The last confirmed state is not available, so questions are paused instead of guessing. ${checkProcessedQuery.error instanceof Error ? checkProcessedQuery.error.message : "The status request failed."}`}
            actionLabel="Retry status check"
            onAction={() => checkProcessedQuery.refetch()}
          />
        )}

        {file && checkProcessedQuery.data && checkProcessedQuery.isError && (
          <FailureNotice
            testId="chat-status-stale"
            variant="stale"
            title="Showing the last confirmed state"
            message="This paper's status could not be refreshed. Nothing changed on the server, and questions stay available."
            actionLabel="Refresh status"
            onAction={() => checkProcessedQuery.refetch()}
          />
        )}

        {file && !checkProcessedQuery.isError && (!isProcessed || isIndexing) && (
          <div className="flex flex-col items-center py-16 text-center">
            <Loader2 className="size-6 animate-spin text-ink-faint" />
            <h4 className="mt-4 font-serif text-sm font-medium">
              {isIndexing ? "Reindexing Document…" : "Indexing Document…"}
            </h4>
            <p className="mt-1 max-w-[26ch] text-xs leading-relaxed text-ink-faint">Generating semantic vector representations for retrieval-augmented analysis.</p>
          </div>
        )}

        {file && isStaleIndex && !isIndexing && documentIndex && (
          <StaleIndexNotice index={documentIndex} />
        )}

        {file && messagesQuery.isError && (
          <FailureNotice
            testId="chat-conversation-error"
            title="Could not load this Conversation"
            message={`Past turns are unavailable right now — new questions still work. ${messagesQuery.error instanceof Error ? messagesQuery.error.message : "The request failed."}`}
            actionLabel="Retry loading turns"
            onAction={() => messagesQuery.refetch()}
          />
        )}

        {file && isProcessed && !isIndexing && !isStaleIndex && !messagesQuery.isError && messages.length === 0 && !thinking && (
          <div className="rounded-sm border border-rule bg-paper p-5 text-center shadow-sm">
            <h4 className="font-mono text-[0.68rem] font-semibold uppercase tracking-widest">Session Initialized</h4>
            <p className="mx-auto mt-2 max-w-[30ch] font-serif text-xs leading-relaxed text-ink-faint">
              Select a query template below or enter a custom prompt in the input workbench.
            </p>
            <div className="mt-4 space-y-1.5 text-left">
              {suggestedPrompts.map((q) => (
                <button
                  key={q}
                  type="button"
                  onClick={() => send(q)}
                  className="flex w-full items-center justify-between rounded-sm border border-rule bg-canvas px-3 py-2 text-xs text-ink-soft transition-colors hover:border-ink hover:text-ink"
                >
                  <span className="truncate">{q}</span>
                  <CornerDownLeft className="size-3 opacity-50" />
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m) => {
          const outcome = describe(m);
          // Narrowed once so the retry button never passes an undefined
          // question to a new turn.
          const retryQuestion = m.question;
          return m.sender === "user" ? (
            <div key={m.id} className="rise-in flex justify-end" data-testid="chat-message" data-sender="user">
              <p className="max-w-[85%] rounded-lg rounded-br-[2px] bg-ink px-3.5 py-2.5 text-[0.85rem] leading-snug text-paper">{m.text}</p>
            </div>
          ) : (
            <div key={m.id} className="rise-in" data-outcome={outcome.outcome} data-testid="chat-message" data-sender="bot">
              <p className="label-meta mb-2">Synthesis · {outcome.label}</p>
              {m.cancelled && !m.text && (
                <p className="font-serif text-xs italic text-ink-faint">
                  {m.stopReason
                    ? `This answer was stopped (${m.stopReason}) before it finished, so it was not saved.`
                    : "This answer was stopped before it finished, so it was not saved."}
                </p>
              )}
              {m.abstained && m.abstentionReason && (
                <p className="mb-1.5 font-mono text-[0.62rem] leading-relaxed tracking-wide text-ink-faint">
                  {ABSTENTION_HINTS[m.abstentionReason]}
                </p>
              )}
              {m.abstained && m.abstentionReason === "evidence_unusable" && (
                <button
                  type="button"
                  onClick={() => file && reindex.mutate(file.name)}
                  disabled={reindex.isPending || !file}
                  className="mb-1.5 inline-flex items-center gap-1.5 border border-rule px-2.5 py-1 font-mono text-[0.62rem] uppercase tracking-wide text-ink-soft transition-colors hover:border-ink hover:text-ink disabled:opacity-40"
                >
                  {reindex.isPending ? (
                    <Loader2 className="size-3 animate-spin" />
                  ) : (
                    <RefreshCw className="size-3" />
                  )}
                  {reindex.isPending ? "Queueing reindex…" : "Reindex this paper"}
                </button>
              )}
              {m.truncated && !m.failed && !m.cancelled && (
                <p className="mb-1.5 font-mono text-[0.62rem] tracking-wide text-ink-faint">
                  The model hit its answer limit, so this stops mid-thought.
                </p>
              )}
              {m.failed && m.failure && (
                <>
                  <p className="font-mono text-[0.62rem] font-semibold tracking-wide text-destructive">
                    {FAILURE_COPY[m.failure].heading}
                  </p>
                  <p className="mt-1 font-mono text-[0.62rem] leading-relaxed tracking-wide text-ink-soft">
                    {FAILURE_COPY[m.failure].hint}
                  </p>
                </>
              )}
              <div
                role={m.failed ? "alert" : undefined}
                className={cn(
                  "rounded-sm border px-3.5 py-3",
                  m.failed ? "border-destructive/30 bg-destructive/5 text-destructive" : "border-transparent bg-transparent px-0 py-0",
                )}
              >
                {m.text ? (
                  <>
                    <MarkdownRenderer text={m.text} />
                    {m.needsSettings && <OpenSettingsLink />}
                    {m.failed && m.failure && retryQuestion && (
                      <button
                        type="button"
                        onClick={() => send(retryQuestion)}
                        disabled={thinking || inputDisabled}
                        className="mt-2 inline-flex items-center gap-1.5 border border-destructive/50 bg-paper px-2.5 py-1 font-mono text-[0.62rem] text-destructive hover:border-destructive disabled:opacity-40"
                      >
                        <RotateCw className="size-3" /> {FAILURE_COPY[m.failure].retryLabel}
                      </button>
                    )}
                  </>
                ) : (
                  <span className="flex items-center gap-1.5 py-1">
                    <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-ink/20" />
                    <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-ink/20 [animation-delay:0.2s]" />
                    <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-ink/20 [animation-delay:0.4s]" />
                  </span>
                )}
              </div>
              {!m.failed && m.sources && m.sources.length > 0 && (
                <SourceList sources={m.sources} retrieval={m.retrieval} />
              )}
              {!m.failed && m.claims && m.claims.length > 0 && m.sources && (
                <ClaimList claims={m.claims} sources={m.sources} />
              )}
            </div>
          );
        })}

        {thinking && messages[messages.length - 1]?.sender !== "bot" && (
          <div className="rise-in">
            <p className="label-meta mb-2">Reading passages…</p>
            <div className="space-y-2">
              {[90, 76, 58].map((w, i) => (
                <span key={i} className="block h-2 animate-pulse rounded-full bg-ink/10" style={{ width: `${w}%`, animationDelay: `${i * 120}ms` }} />
              ))}
            </div>
          </div>
        )}
        <div ref={endRef} />
      </div>

      <div className="border-t border-rule px-5 py-4">
        <div className="mb-3 flex flex-wrap gap-1.5">
          {suggestedPrompts.map((p) => (
            <button
              key={p}
              type="button"
              onClick={() => send(p)}
              disabled={inputDisabled}
              className="rounded-full border border-rule px-2.5 py-1 text-[0.7rem] text-ink-soft transition-colors hover:border-ink hover:text-ink disabled:opacity-40"
            >
              {p}
            </button>
          ))}
        </div>

        <div className="rounded-sm border border-rule bg-card px-3 py-2.5 transition-colors focus-within:border-ink">
          <textarea
            ref={inputRef}
            rows={2}
            data-testid="chat-input"
            aria-label="Ask this paper a question"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send(value);
              }
            }}
            placeholder={
              !file
                ? "Select a paper…"
                : isStaleIndex
                  ? "Reindex this paper to ask questions…"
                  : isProcessed
                    ? "Ask this paper something…"
                    : "Indexing — questions paused…"
            }
            disabled={inputDisabled}
            className="w-full resize-none bg-transparent text-[0.85rem] leading-relaxed placeholder:text-ink-faint focus:outline-none disabled:opacity-60"
          />
          <div className="mt-1 flex items-center justify-between">
            <span className="label-meta flex items-center gap-1">
              <CornerDownLeft className="size-2.5" /> to send
            </span>
            <button
              type="button"
              onClick={() => send(value)}
              disabled={!value.trim() || thinking || inputDisabled}
              className="flex size-7 items-center justify-center rounded-sm bg-ink text-paper transition-opacity disabled:opacity-25"
              aria-label="Send"
              data-testid="chat-send"
            >
              <ArrowUp className="size-3.5" />
            </button>
          </div>
        </div>
        <p className="label-meta mt-2 text-center">PaperMind · Citations stay on the page</p>
      </div>
    </section>
  );
}
