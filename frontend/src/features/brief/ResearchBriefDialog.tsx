import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Scale, X } from "lucide-react";
import {
  briefUnavailableReason,
  checkBriefScope,
  isBriefBlocked,
  isBriefReadable,
} from "@/services/research";
import { getSettings } from "@/services/settings";
import { getErrorMessage } from "@/lib/api-error";
import { useFiles } from "@/hooks/useFiles";
import { useBriefRun } from "@/hooks/useBriefRun";
import type { File as DbFile } from "@/types/db";
import { useEscapeKey } from "@/hooks/useEscape";
import { useFocusTrap } from "@/hooks/useFocusTrap";
import useBriefUi from "@/store/brief-ui";
import BriefResult from "./BriefResult";
import BriefFooter from "./BriefFooter";
import BriefRequestForm from "./BriefRequestForm";

/**
 * Start a Research Brief over two chosen Documents.
 *
 * The scope is chosen before anything runs and stays on screen the whole time,
 * because a brief that reads two papers is a claim about which two, and a
 * reader who cannot see the scope cannot check the result against it. The
 * button stays disabled until the pair is exactly two readable Documents and
 * the configured model can run one, and it says which of those is missing
 * rather than simply refusing.
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
  const dialogRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);

  const files = filesQuery.data?.files;

  // A Document that left the library cannot stay in the scope: the run would
  // name a paper the reader can no longer open.
  useEffect(() => {
    const names = new Set((files ?? []).map((file) => file.name));
    setSelected((current) => current.filter((name) => names.has(name)));
  }, [files]);

  const readable = useMemo(
    () => (files ?? []).filter((file) => isBriefReadable(file) && !isBriefBlocked(file)),
    [files],
  );

  // The server judges whether a pair can be read: a Document being reindexed or
  // carrying a stale index looks the same in the listing as one that is fine,
  // and only the server holds the index state that decides. The listing filters
  // the obvious cases so an unusable Document is never offered; this confirms
  // the pair before a brief is started on it.
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
  // listing's: the two answer different questions, and a reader told only
  // "something is wrong" learns nothing about what to do next.
  const serverReason =
    scopeQuery.isError ? getErrorMessage(scopeQuery.error) || null : null;
  const reason = listingReason ?? serverReason;
  const canStart =
    listingReason === null &&
    serverReason === null &&
    !scopeQuery.isFetching &&
    scopeQuery.isSuccess &&
    question.trim().length > 0;

  const brief = useBriefRun({ question, documents: selected, canStart });
  const running = brief.running;

  useFocusTrap(true, dialogRef, closeRef);
  useEscapeKey(!running, close);

  const toggle = useCallback((file: DbFile) => {
    setSelected((current) =>
      current.includes(file.name)
        ? current.filter((name) => name !== file.name)
        : // A brief is a pair. A third selection would either silently drop one
          // or silently widen the scope, so the oldest selection goes instead.
          [...current, file.name].slice(-2),
    );
  }, []);

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
          <BriefRequestForm
            library={
              filesQuery.isPending
                ? "loading"
                : filesQuery.isError
                  ? "failed"
                  : readable.length < 2
                    ? "tooFew"
                    : "ready"
            }
            libraryError={getErrorMessage(filesQuery.error) || null}
            onRetryLibrary={() => filesQuery.refetch()}
            files={readable}
            selected={selected}
            onToggle={toggle}
            scope={scopeQuery.isSuccess ? scopeQuery.data.documents : null}
            failure={brief.failure}
            question={question}
            onQuestion={setQuestion}
            disabled={running}
            reason={settingsQuery.isPending ? null : reason}
          />

          <BriefResult run={brief.run} />
        </div>

        <BriefFooter
          running={running}
          canStart={canStart && !running}
          onStart={() => void brief.start()}
          onStop={() => void brief.stop()}
        />
      </div>
    </div>
  );
}
