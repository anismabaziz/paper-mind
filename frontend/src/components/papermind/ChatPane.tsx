import { useEffect, useRef, useState, isValidElement, type ComponentPropsWithoutRef } from "react";
import { ArrowUp, ChevronDown, CornerDownLeft, Filter, Loader2, Settings, Copy, Check, FileText } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { checkIsProcessed, chatStream, getMessages, type ISource } from "@/services/files";
import usePdfStore from "@/store/pdf-state";
import useSettingsUi from "@/store/settings-ui";
import { cn } from "@/lib/utils";

type ChatMessage = {
  id: string;
  text: string;
  sender: "user" | "bot";
  sources?: ISource[];
  failed?: boolean;
  needsSettings?: boolean;
};

const SETTINGS_ERROR_PATTERNS = ["No chat provider configured", "Re-save your provider settings"];
const isSettingsError = (message: string) => SETTINGS_ERROR_PATTERNS.some((p) => message.includes(p));

const suggestedPrompts = [
  "What is the main topic?",
  "Summarize key findings",
  "Methodology used?",
];

function SourceList({ sources }: { sources: ISource[] }) {
  const [open, setOpen] = useState(true);
  const setCitationTarget = usePdfStore((s) => s.setCitationTarget);
  return (
    <div className="mt-4 border-t border-rule pt-3">
      <button type="button" onClick={() => setOpen((v) => !v)} className="flex w-full items-center justify-between text-ink-faint hover:text-ink">
        <span className="label-meta">Grounded in {sources.length} passages</span>
        <ChevronDown className={cn("size-3 transition-transform", open && "rotate-180")} />
      </button>
      {open && (
        <ol className="mt-3 space-y-3">
          {sources.map((s, idx) => {
            const conf = Math.round((s.score ?? 0) * 100);
            const hasPage = s.page != null;
            const citationButton = (
              <span className="font-mono text-[0.62rem] tracking-wide text-ink-faint">
                <span className="text-marker">[{idx + 1}]</span> {s.document} · chunk {s.chunk_index}
                {hasPage ? ` · p. ${s.page}` : ""}
              </span>
            );
            return (
              <li key={idx}>
                <button
                  type="button"
                  disabled={!hasPage}
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
                    <span className="font-mono text-[0.6rem] text-ink-faint">{conf}%</span>
                  </span>
                  <span className="mt-1 block font-serif text-[0.85rem] leading-snug text-ink-soft italic">“{s.content.slice(0, 220)}”</span>
                  <span className="mt-1.5 flex items-center gap-2">
                    <span className="h-px w-16 bg-rule">
                      <span className="block h-px bg-marker" style={{ width: `${conf}%` }} />
                    </span>
                    <span className="font-mono text-[0.58rem] text-ink-faint">{conf}% match</span>
                  </span>
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

function CodeBlock({ code, lang }: { code: string; lang: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="my-2 overflow-hidden rounded-sm border border-rule bg-canvas font-mono text-[0.72rem] leading-relaxed">
      <div className="flex items-center justify-between border-b border-rule bg-paper px-3 py-1.5 font-sans text-[0.62rem] uppercase tracking-wider text-ink-faint">
        <span>{lang || "code"}</span>
        <button
          onClick={() => {
            navigator.clipboard.writeText(code);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
          }}
          className="flex items-center gap-1 text-ink-faint hover:text-ink"
        >
          {copied ? <Check className="size-3 text-marker" /> : <Copy className="size-3" />}
          <span>{copied ? "Copied" : "Copy"}</span>
        </button>
      </div>
      <pre className="overflow-x-auto p-3 text-ink">
        <code>{code}</code>
      </pre>
    </div>
  );
}

function MarkdownPre({ children }: ComponentPropsWithoutRef<"pre">) {
  if (isValidElement<ComponentPropsWithoutRef<"code">>(children)) {
    const { className, children: code } = children.props;
    const lang = /language-(\S+)/.exec(className ?? "")?.[1] ?? "";
    return <CodeBlock code={String(code).replace(/\n$/, "")} lang={lang} />;
  }
  return <pre>{children}</pre>;
}

function MarkdownInlineCode({ children, ...props }: ComponentPropsWithoutRef<"code">) {
  return (
    <code className="rounded-sm border border-rule bg-canvas px-1.5 py-0.5 font-mono text-[0.68rem] text-ink mx-0.5" {...props}>
      {children}
    </code>
  );
}

function MarkdownRenderer({ text }: { text: string }) {
  return (
    <div className="font-serif text-[0.94rem] leading-[1.65] text-pretty [&_p]:my-2 [&_ul]:list-disc [&_ul]:pl-4 [&_ol]:list-decimal [&_ol]:pl-4 [&_li]:leading-relaxed [&_strong]:font-semibold [&_a]:text-marker [&_a]:underline [&_blockquote]:border-l-2 [&_blockquote]:border-rule [&_blockquote]:pl-3 [&_blockquote]:italic [&_blockquote]:text-ink-soft [&_h1]:text-base [&_h1]:font-semibold [&_h2]:text-sm [&_h2]:font-semibold">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={{ pre: MarkdownPre, code: MarkdownInlineCode }}>
        {text}
      </ReactMarkdown>
    </div>
  );
}

export function ChatPane() {
  const { file } = usePdfStore();
  const [value, setValue] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [thinking, setThinking] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const checkProcessedQuery = useQuery({
    queryKey: [file?.name, "is-processed"],
    queryFn: () => checkIsProcessed(file!),
    enabled: !!file,
    refetchInterval: (q) => (q.state.data?.is_processed ? false : 3000),
  });

  const messagesQuery = useQuery({
    queryKey: [file?.name, "messages"],
    queryFn: () => getMessages(file!.name),
    enabled: !!file && checkProcessedQuery.data?.is_processed === true,
  });

  useEffect(() => {
    if (file) {
      if (messagesQuery.data?.messages) {
        setMessages(messagesQuery.data.messages.map((m) => ({ id: m.id, text: m.text, sender: m.sender, sources: m.sources })));
      } else if (!messagesQuery.isFetching) {
        setMessages([]);
      }
    } else {
      setMessages([]);
    }
  }, [file, messagesQuery.data, messagesQuery.isFetching]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, thinking]);

  useEffect(() => {
    inputRef.current?.focus();
  }, [file]);

  async function send(text: string) {
    const body = text.trim();
    if (!body || !file || thinking) return;
    setValue("");
    setThinking(true);
    const botId = crypto.randomUUID();
    setMessages((m) => [...m, { id: crypto.randomUUID(), text: body, sender: "user" }, { id: botId, text: "", sender: "bot" }]);

    try {
      await chatStream(body, file.name, {
        onToken: (t) => setMessages((prev) => prev.map((msg) => (msg.id === botId ? { ...msg, text: msg.text + t } : msg))),
        onError: (message) =>
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === botId ? { ...msg, text: message, failed: true, needsSettings: isSettingsError(message) } : msg,
            ),
          ),
        onDone: (sources) => setMessages((prev) => prev.map((msg) => (msg.id === botId ? { ...msg, sources } : msg))),
      });
    } catch (e) {
      const message = e instanceof Error ? e.message : "";
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === botId
            ? {
                ...msg,
                text: message || "Connection lost. Please ensure the backend server is active.",
                failed: true,
                needsSettings: isSettingsError(message),
              }
            : msg,
        ),
      );
    } finally {
      setThinking(false);
      inputRef.current?.focus();
    }
  }

  const isProcessed = checkProcessedQuery.data?.is_processed;
  const inputDisabled = !file || !isProcessed;

  return (
    <section className="flex w-[26rem] shrink-0 flex-col border-l border-rule bg-background">
      <header className="flex h-14 items-center justify-between border-b border-rule px-5">
        <div>
          <p className="text-[0.82rem] font-medium">Reading companion</p>
          <p className="label-meta">Answers cite this paper only</p>
        </div>
        <button
          type="button"
          className="flex items-center gap-1.5 rounded-sm border border-rule px-2 py-1 text-[0.7rem] text-ink-soft hover:border-ink hover:text-ink"
        >
          <Filter className="size-3" /> Scope
        </button>
      </header>

      <div className="scroll-slim flex-1 space-y-7 overflow-y-auto px-5 py-6">
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

        {file && !isProcessed && (
          <div className="flex flex-col items-center py-16 text-center">
            <Loader2 className="size-6 animate-spin text-ink-faint" />
            <h4 className="mt-4 font-serif text-sm font-medium">Indexing Document…</h4>
            <p className="mt-1 max-w-[26ch] text-xs leading-relaxed text-ink-faint">Generating semantic vector representations for retrieval-augmented analysis.</p>
          </div>
        )}

        {file && isProcessed && messages.length === 0 && !thinking && (
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

        {messages.map((m) =>
          m.sender === "user" ? (
            <div key={m.id} className="rise-in flex justify-end">
              <p className="max-w-[85%] rounded-lg rounded-br-[2px] bg-ink px-3.5 py-2.5 text-[0.85rem] leading-snug text-paper">{m.text}</p>
            </div>
          ) : (
            <div key={m.id} className="rise-in">
              <p className="label-meta mb-2">Synthesis · {m.failed ? "failed" : "grounded"}</p>
              <div
                className={cn(
                  "rounded-sm border px-3.5 py-3",
                  m.failed ? "border-destructive/30 bg-destructive/5 text-destructive" : "border-transparent bg-transparent px-0 py-0",
                )}
              >
                {m.text ? (
                  <>
                    <MarkdownRenderer text={m.text} />
                    {m.needsSettings && <OpenSettingsLink />}
                  </>
                ) : (
                  <span className="flex items-center gap-1.5 py-1">
                    <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-ink/20" />
                    <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-ink/20 [animation-delay:0.2s]" />
                    <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-ink/20 [animation-delay:0.4s]" />
                  </span>
                )}
              </div>
              {m.sources && m.sources.length > 0 && <SourceList sources={m.sources} />}
            </div>
          ),
        )}

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
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send(value);
              }
            }}
            placeholder={file ? (isProcessed ? "Ask this paper something…" : "Indexing — questions paused…") : "Select a paper…"}
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
