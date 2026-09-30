import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Loader2, Scale, Search, Square, X } from "lucide-react";
import {
  BriefRequestError,
  briefStream,
  cancelBrief,
  briefUnavailableReason,
  checkBriefScope,
  isBriefBlocked,
  isBriefReadable,
  type IBriefBudget,
  type IBriefDocument,
  type IBriefEvidence,
} from "@/services/research";
import { getSettings } from "@/services/settings";
import { useFiles } from "@/hooks/useFiles";
import { MarkdownRenderer } from "./MarkdownRenderer";
import { FailureNotice } from "./FailureNotice";
import { displayTitle } from "@/types/db";
import type { File as DbFile } from "@/types/db";
import { cn } from "@/lib/utils";
import { useEscapeKey } from "@/hooks/useEscape";
import { useFocusTrap } from "@/hooks/useFocusTrap";
import usePdfStore from "@/store/pdf-state";
import useBriefUi from "@/store/brief-ui";

/** What a run ended as, in the one word the reader reads for it. */
const OUTCOME_LABEL: Record<string, string> = {
  complete: "complete",
  incomplete: "incomplete",
  failed: "failed",
  abstained: "abstained",
};

/** A budget for a run that was stopped before it could report one. */
const NO_BUDGET: IBriefBudget = {
  turns: 0,
  tool_calls: 0,
  repeated_calls: 0,
  tokens: 0,
  elapsed_seconds: 0,
  documents: 2,
  max_turns: 0,
  max_tool_calls: 0,
  max_repeated_calls: 0,
  max_tokens: 0,
  max_seconds: 0,
};

type Run =
  | { state: "idle" }
  | { state: "running"; documents: IBriefDocument[]; briefId: string | null }
  | {
      state: "done";
      status: "complete" | "incomplete";
      answer: string;
      stoppedBy: string | null;
      message: string | null;
      evidence: IBriefEvidence[];
      budget: IBriefBudget;
      documents: IBriefDocument[];
    }
  | {
      state: "abstained";
      message: string;
      documents: IBriefDocument[];
      stoppedBy: string | null;
    }
  | {
      state: "failed";
      message: string;
      evidence: IBriefEvidence[];
      documents: IBriefDocument[];
    };

/**
 * Start a Research Brief over two chosen Documents.
 *
 * The scope is chosen before anything runs and stays on screen the whole time,
 * because a brief that reads two papers is a claim about which two, and a reader
 * who cannot see the scope cannot check the result against it. The button stays
 * disabled until the pair is exactly two readable Documents and the configured
 * model can run one, and it says which of those is missing rather than simply
 * refusing.
 */
export function ResearchBriefDialog() {
  const isOpen = useBriefUi((state) => state.isOpen);
  return isOpen ? <ResearchBriefDialogContent /> : null;
}

function ResearchBriefDialogContent() {
  const { close } = useBriefUi();
  const filesQuery = useFiles();
  const settingsQuery = useQuery({ queryKey: ["settings"], queryFn: getSettings });
  const [selected, setSelected] = useState<string[]>([]);
  const [question, setQuestion] = useState("");
  const [run, setRun] = useState<Run>({ state: "idle" });
  const [failure, setFailure] = useState<string | null>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const controllerRef = useRef<AbortController | null>(null);
  const running = run.state === "running";

  useFocusTrap(true, dialogRef, closeRef);
  useEscapeKey(!running, close);

  useEffect(() => () => controllerRef.current?.abort(), []);

  // A Document that left the library cannot stay in the scope: the run would
  // name a paper the reader can no longer open.
  useEffect(() => {
    const names = new Set((filesQuery.data?.files ?? []).map((file) => file.name));
    setSelected((current) => current.filter((name) => names.has(name)));
  }, [filesQuery.data?.files]);

  const files = filesQuery.data?.files;
  const readable = useMemo(
    () => (files ?? []).filter((file) => isBriefReadable(file) && !isBriefBlocked(file)),
    [files],
  );
  // The server judges whether a pair can be read: a Document being reindexed
  // or carrying a stale index looks the same in the listing as one that is
  // fine, and only the server holds the index state that decides. The listing
  // filters the obvious cases so an unusable Document is never offered; this
  // confirms the pair before a brief is started on it.
  const scopeQuery = useQuery({
    queryKey: ["research-scope", selected.join("|")],
    queryFn: ({ signal }) => checkBriefScope(selected, signal),
    enabled: selected.length === 2,
    retry: false,
  });
  const model = useMemo(() => {
    const settings = settingsQuery.data;
    if (!settings?.provider || !settings.model) return null;
    return Object.values(settings.supported_models ?? {})
      .flat()
      .find((entry) => entry.provider === settings.provider && entry.id === settings.model);
  }, [settingsQuery.data]);
  const chosen = useMemo(
    () =>
      selected
        .map((name) => (files ?? []).find((file) => file.name === name))
        .filter((file): file is DbFile => Boolean(file)),
    [files, selected],
  );

  const listingReason = briefUnavailableReason(model ?? null, chosen, files);
  // The server's verdict is shown as its own refusal, not merged into the
  // listing's: the two answer different questions, and a reader who is told
  // only "something is wrong" learns nothing about what to do next.
  const serverReason =
    scopeQuery.isError && scopeQuery.error instanceof Error
      ? scopeQuery.error.message
      : null;
  const reason = listingReason ?? serverReason;
  const canStart =
    listingReason === null &&
    serverReason === null &&
    !scopeQuery.isFetching &&
    scopeQuery.isSuccess &&
    question.trim().length > 0 &&
    !running;

  const toggle = useCallback((file: DbFile) => {
    setSelected((current) =>
      current.includes(file.name)
        ? current.filter((name) => name !== file.name)
        : // A brief is a pair. A third selection would either silently drop one
          // or silently widen the scope, so the oldest selection goes instead.
          [...current, file.name].slice(-2),
    );
  }, []);

  const start = useCallback(async () => {
    const body = question.trim();
    if (!canStart || !body) return;
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setFailure(null);
    setRun({ state: "running", documents: [], briefId: null });
    try {
      await briefStream(
        body,
        [...selected],
        {
          onStart: (opened) =>
            setRun({ state: "running", documents: opened.documents, briefId: opened.briefId }),
          onDone: (done) =>
            setRun({
              state: "done",
              status: done.status,
              answer: done.answer,
              stoppedBy: done.stoppedBy,
              message: done.message,
              evidence: done.evidence,
              budget: done.budget,
              documents: done.documents,
            }),
          onAbstained: (result) =>
            setRun({
              state: "abstained",
              message: result.message,
              documents: result.documents,
              stoppedBy: result.stoppedBy,
            }),
          onProviderError: (providerFailure) =>
            setRun({
              state: "failed",
              message: providerFailure.message,
              evidence: providerFailure.evidence,
              documents: providerFailure.documents,
            }),
        },
        { signal: controller.signal },
      );
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        // The reader stopped the brief. The server answers that with the
        // evidence it had collected; this branch only runs when the connection
        // itself went away, and says so rather than inventing a result.
        setRun((current) =>
          current.state === "running"
            ? {
                state: "done",
                status: "incomplete",
                answer: "",
                stoppedBy: "cancelled",
                message: "You stopped this brief, so it was not finished.",
                evidence: [],
                budget: NO_BUDGET,
                documents: current.documents,
              }
            : current,
        );
        return;
      }
      setFailure(
        error instanceof BriefRequestError || error instanceof Error
          ? error.message
          : "The brief could not be read. Try again.",
      );
      setRun({ state: "idle" });
    } finally {
      if (controllerRef.current === controller) controllerRef.current = null;
    }
  }, [canStart, question, selected]);

  /**
   * Stop a brief without throwing away what it found.
   *
   * The server is asked to finish the run, which ends the stream with the
   * evidence collected so far and an explicit incomplete status. The connection
   * is only dropped if the server never answers, so a reader who pressed stop
   * is not left watching a spinner.
   */
  const stop = useCallback(async () => {
    const briefId = run.state === "running" ? run.briefId : null;
    if (briefId) {
      try {
        await cancelBrief(briefId);
        return;
      } catch {
        // The run may already have ended between the reader pressing stop and
        // this arriving; the abort below is the fallback either way.
      }
    }
    controllerRef.current?.abort();
  }, [run]);

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-ink/40 p-4 backdrop-blur-sm"
      data-testid="brief-backdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && !running) close();
      }}
    >
      <div
        ref={dialogRef}
        tabIndex={-1}
        role="dialog"
        aria-modal="true"
        aria-labelledby="brief-dialog-title"
        aria-busy={running}
        className="flex max-h-[90vh] w-full max-w-2xl flex-col border border-rule bg-paper shadow-sheet focus:outline-none"
        data-testid="brief-dialog"
      >
        <header className="flex items-center justify-between border-b border-rule px-5 py-3">
          <div className="flex items-center gap-2">
            <Scale size={16} className="text-ink-soft" />
            <h2
              id="brief-dialog-title"
              className="font-mono text-xs font-semibold uppercase tracking-widest text-ink"
            >
              Research brief
            </h2>
          </div>
          <button
            ref={closeRef}
            type="button"
            onClick={close}
            disabled={running}
            aria-label="Close research brief"
            className="grid size-11 place-items-center text-ink-faint hover:text-ink disabled:opacity-40"
          >
            <X size={16} />
          </button>
        </header>

        <div className="scroll-slim flex-1 overflow-y-auto px-5 py-4">
          {filesQuery.isPending ? (
            <p role="status" className="py-8 text-center font-mono text-xs text-ink-faint">
              <Loader2 size={14} className="mr-2 inline animate-spin" />
              Loading your library…
            </p>
          ) : filesQuery.isError ? (
            <FailureNotice
              testId="brief-library-error"
              title="Could not load your library"
              message={`The Documents a brief could read are unavailable, so none were selected. ${filesQuery.error instanceof Error ? filesQuery.error.message : "The library request failed."}`}
              actionLabel="Retry library load"
              onAction={() => filesQuery.refetch()}
            />
          ) : readable.length < 2 ? (
            <p
              className="py-8 text-center font-serif text-xs leading-relaxed text-ink-faint"
              data-testid="brief-not-enough"
            >
              A brief compares two indexed Documents. Index at least two before starting
              one.
            </p>
          ) : (
            <ScopePicker
              files={readable}
              selected={selected}
              onToggle={toggle}
              scope={scopeQuery.isSuccess ? scopeQuery.data.documents : null}
            />
          )}

          {failure && (
            <FailureNotice testId="brief-error" title="This brief could not start" message={failure} />
          )}

          <label className="mt-4 block">
            <span className="label-meta">Cross-document question</span>
            <textarea
              rows={3}
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              disabled={running}
              aria-label="Cross-document question"
              placeholder="Where do these two papers disagree?"
              data-testid="brief-question"
              className="w-full resize-y rounded-sm border border-rule bg-card px-3 py-2 text-xs leading-relaxed text-ink outline-none focus:border-ink disabled:opacity-60"
            />
          </label>

          {/* Nothing is said while App Settings are still loading: a reason
              read off a half-loaded catalog would name the model when the
              reader's problem is their Documents, and the other way round. */}
          {reason && !settingsQuery.isPending && (
            <p className="label-meta mt-2" data-testid="brief-reason">
              {reason}
            </p>
          )}

          <BriefResult run={run} />
        </div>

        <footer className="flex items-center justify-between gap-2 border-t border-rule px-5 py-3">
          {running ? (
            <button
              type="button"
              onClick={() => void stop()}
              data-testid="brief-stop"
              className="inline-flex items-center gap-1.5 border border-rule bg-paper px-3 py-1.5 font-mono text-[0.65rem] text-ink-soft hover:border-destructive hover:text-destructive"
            >
              <Square size={12} /> Stop
            </button>
          ) : (
            <p className="label-meta">Reads only the two Documents above.</p>
          )}
          <button
            type="button"
            onClick={start}
            disabled={!canStart}
            data-testid="brief-start"
            className="inline-flex h-11 min-h-[44px] items-center gap-1.5 bg-ink px-4 font-mono text-[0.65rem] text-paper hover:bg-ink/90 disabled:opacity-40"
          >
            {running ? <Loader2 size={12} className="animate-spin" /> : <Search size={12} />}
            {running ? "Reading…" : "Start brief"}
          </button>
        </footer>
      </div>
    </div>
  );
}

/**
 * The Documents a brief may read, chosen from what is actually readable.
 *
 * A Document that is indexing, being removed, or carrying a stale index is not
 * offered at all rather than offered and refused: it cannot be part of a scope,
 * and a checkbox that cannot be checked is a worse affordance than one that is
 * not there. The labels A and B are the ones the brief itself uses, shown on the
 * selection rather than only in the result.
 */
function ScopePicker({
  files,
  selected,
  onToggle,
  scope,
}: {
  files: DbFile[];
  selected: string[];
  onToggle: (file: DbFile) => void;
  /** The labels and titles the server assigned, once it has confirmed the pair. */
  scope: IBriefDocument[] | null;
}) {
  return (
    <fieldset>
      <legend className="label-meta">Scope · select exactly two</legend>
      {scope && (
        <p className="mt-1 font-mono text-[0.6rem] text-ink-faint" data-testid="brief-scope-labels">
          {[...scope].sort((a, b) => a.label.localeCompare(b.label)).map((item) => `[${item.label}] ${item.title}`).join("  ·  ")}
        </p>
      )}
      <ul className="mt-2 max-h-56 space-y-px overflow-y-auto" data-testid="brief-scope-list">
        {files.map((file) => {
          const active = selected.includes(file.name);
          return (
            <li key={file.id}>
              <label
                className={cn(
                  "flex min-h-[44px] cursor-pointer items-center gap-2 border-l-2 px-3 py-2 text-xs",
                  active ? "border-marker bg-paper text-ink" : "border-rule text-ink-soft",
                )}
              >
                <input
                  type="checkbox"
                  checked={active}
                  onChange={() => onToggle(file)}
                  data-testid={`brief-scope-${file.name}`}
                  aria-label={`Include ${displayTitle(file)} in the brief`}
                  className="size-3.5 accent-[var(--color-ink)]"
                />
                <span className="min-w-0 flex-1 truncate">{displayTitle(file)}</span>
                {active && (
                  <span className="font-mono text-[0.6rem] text-marker">
                    {selected.indexOf(file.name) === 0 ? "A" : "B"}
                  </span>
                )}
              </label>
            </li>
          );
        })}
      </ul>
    </fieldset>
  );
}

/** What one run produced, labelled by how it ended. */
function BriefResult({ run }: { run: Run }) {
  if (run.state === "idle") return null;
  if (run.state === "running") {
    return (
      <p
        role="status"
        className="mt-5 font-mono text-xs text-ink-faint"
        data-testid="brief-running"
      >
        <Loader2 size={12} className="mr-2 inline animate-spin" />
        Searching the two Documents…
      </p>
    );
  }
  if (run.state === "failed") {
    return (
      <div
        role="alert"
        className="mt-5 rounded-sm border border-destructive/30 bg-destructive/5 p-3"
        data-testid="brief-failed"
      >
        <p className="font-mono text-xs font-semibold text-destructive">
          {OUTCOME_LABEL.failed}
        </p>
        <p className="mt-1 font-mono text-[0.65rem] leading-relaxed text-ink-soft">
          {run.message}
        </p>
        <EvidenceList evidence={run.evidence} />
      </div>
    );
  }
  if (run.state === "abstained") {
    return (
      <div
        role="status"
        className="mt-5 rounded-sm border border-rule bg-canvas p-3"
        data-testid="brief-abstained"
      >
        <p className="label-meta">{OUTCOME_LABEL.abstained}</p>
        <p className="mt-1 font-serif text-xs leading-relaxed text-ink-soft">{run.message}</p>
        {run.stoppedBy && (
          <p className="mt-1 font-mono text-[0.6rem] text-ink-faint">
            Stopped by {humanize(run.stoppedBy)}.
          </p>
        )}
      </div>
    );
  }
  return (
    <div
      className="mt-5 border-t border-rule pt-4"
      data-testid="brief-result"
      data-status={run.status}
    >
      <div className="flex items-center justify-between gap-2">
        <p className="label-meta">
          Brief · {OUTCOME_LABEL[run.status]}
          {run.stoppedBy ? ` · ${humanize(run.stoppedBy)}` : ""}
        </p>
        <BudgetLine budget={run.budget} />
      </div>
      {run.message && (
        <p
          className="mt-2 font-mono text-[0.65rem] leading-relaxed text-ink-soft"
          data-testid="brief-incomplete-note"
        >
          {run.message}
        </p>
      )}
      {run.answer && <MarkdownRenderer text={run.answer} />}
      <EvidenceList evidence={run.evidence} />
    </div>
  );
}

/**
 * What the loop spent, and against which ceiling.
 *
 * Shown because a partial brief is only interpretable next to its limits: six
 * of six turns is a different situation from six turns of eight, and a reader
 * who cannot see the difference cannot tell which one they are looking at.
 */
function BudgetLine({ budget }: { budget: IBriefBudget }) {
  if (!budget.max_turns) return null;
  return (
    <span className="font-mono text-[0.6rem] text-ink-faint" data-testid="brief-budget">
      {budget.turns}/{budget.max_turns} turns · {budget.tool_calls}/{budget.max_tool_calls} reads
    </span>
  );
}

/** The Passages a brief collected, each openable at the Page it came from. */
function EvidenceList({ evidence }: { evidence: IBriefEvidence[] }) {
  const setCitationTarget = usePdfStore((state) => state.setCitationTarget);
  const [open, setOpen] = useState(true);
  if (evidence.length === 0) return null;
  return (
    <div className="mt-4 border-t border-rule pt-3">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        aria-label={`${open ? "Hide" : "Show"} ${evidence.length} passages found`}
        className="flex w-full items-center justify-between text-ink-faint hover:text-ink"
      >
        <span className="label-meta">{evidence.length} passages found</span>
      </button>
      {open && (
        <ol className="mt-2 space-y-2" data-testid="brief-evidence">
          {evidence.map((item) => (
            <li key={item.evidence_id} className="border-l-2 border-rule pl-3">
              <button
                type="button"
                disabled={item.page == null}
                onClick={() => item.page != null && setCitationTarget(item.page)}
                data-testid={`brief-evidence-${item.evidence_id}`}
                className={cn(
                  "block w-full text-left",
                  item.page != null ? "cursor-pointer hover:border-marker" : "cursor-default",
                )}
                title={item.page != null ? `Open page ${item.page}` : "This passage has no page"}
              >
                <span className="font-mono text-[0.6rem] text-ink-faint">
                  <span className="text-marker">[{item.evidence_id}]</span> {item.title}
                  {item.page != null ? ` · p. ${item.page}` : ""} · rank {item.rank}
                  {item.read ? "" : " · excerpt"}
                </span>
                <span className="mt-0.5 block font-serif text-xs italic leading-snug text-ink-soft">
                  “{item.content.slice(0, 200)}”
                </span>
              </button>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

/** Turn a stop reason into the word a reader reads. */
function humanize(reason: string): string {
  return reason.replace(/_/g, " ");
}