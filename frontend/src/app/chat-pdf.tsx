import { MessageSquare, Send, ChevronRight, ChevronDown, Loader2, Cpu, User, Copy, Check, FileText } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import usePdfStore from "@/store/pdf-state";
import { useEffect, useState, useRef, type ComponentPropsWithoutRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useQuery } from "@tanstack/react-query";
import { checkIsProcessed, chatStream, getMessages, ISource } from "@/services/files";
import { cn } from "@/lib/utils";

interface ChatMessage {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  sources?: ISource[];
  failed?: boolean;
}

export default function ChatPDF() {
  const { file } = usePdfStore();
  const scrollRef = useRef<HTMLDivElement>(null);

  const questions = [
    "What is the main topic of this document?",
    "Summarize the key findings",
    "Methodology used in this paper?",
  ];

  const [inputValue, setInputValue] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);

  const checkProcessedQuery = useQuery({
    queryKey: [file?.name, "is-processed"],
    queryFn: () => checkIsProcessed(file!),
    enabled: !!file,
    refetchInterval: (query) => {
      return query.state.data?.is_processed ? false : 3000;
    }
  });

  const messagesQuery = useQuery({
    queryKey: [file?.name, "messages"],
    queryFn: () => getMessages(file!.name),
    enabled: !!file && checkProcessedQuery.data?.is_processed === true,
  });

  useEffect(() => {
    if (file) {
      setInputValue("");
      if (messagesQuery.data?.messages) {
        setMessages(messagesQuery.data.messages);
      } else {
        setMessages([]);
      }
    } else {
      setInputValue("");
      setMessages([]);
    }
  }, [file, messagesQuery.data]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = async () => {
    if (!inputValue.trim() || !file || isStreaming) return;
    const q = inputValue.trim();
    setInputValue("");
    setIsStreaming(true);

    const botId = crypto.randomUUID();
    setMessages((prev) => [
      ...prev,
      { id: crypto.randomUUID(), text: q, sender: 'user' },
      { id: botId, text: "", sender: 'bot' },
    ]);

    try {
      await chatStream(q, file.name, {
        onToken: (text) => {
          setMessages((prev) =>
            prev.map((msg) => msg.id === botId ? { ...msg, text: msg.text + text } : msg)
          );
        },
        onError: (message) => {
          setMessages((prev) =>
            prev.map((msg) => msg.id === botId ? { ...msg, text: message, failed: true } : msg)
          );
        },
        onDone: (sources) => {
          setMessages((prev) =>
            prev.map((msg) => msg.id === botId ? { ...msg, sources } : msg)
          );
        },
      });
    } catch {
      setMessages((prev) =>
        prev.map((msg) => msg.id === botId
          ? { ...msg, text: "Connection lost. Please ensure the backend server is active.", failed: true }
          : msg)
      );
    } finally {
      setIsStreaming(false);
    }
  };

  return (
    <div className="flex flex-col h-full glass rounded-xl overflow-hidden shadow-sm">
      <div className="h-[52px] px-4 border-b border-slate-200/80 bg-slate-50 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2">
          <MessageSquare size={16} className="text-slate-600" />
          <h3 className="font-semibold text-xs uppercase tracking-wider text-slate-700">Research Chat</h3>
        </div>
      </div>

      <div className="flex-grow overflow-y-auto p-4 space-y-5" ref={scrollRef}>
        {!file && (
          <div className="flex flex-col items-center justify-center h-full text-center p-6">
            <div className="w-12 h-12 bg-white border border-slate-200 rounded flex items-center justify-center mb-4 text-slate-400 shadow-sm">
              <MessageSquare size={18} />
            </div>
            <h4 className="text-sm font-semibold text-slate-700 mb-1.5">No Active Session</h4>
            <p className="text-xs text-slate-400 max-w-[220px]">
              Select a research paper from your library to start an interactive analysis session.
            </p>
          </div>
        )}

        {file && !checkProcessedQuery.data?.is_processed && (
          <div className="flex flex-col items-center justify-center h-full text-center p-6 space-y-3">
            <Loader2 className="animate-spin text-slate-500" size={24} />
            <div>
              <h4 className="text-xs font-semibold text-slate-700">Indexing Document...</h4>
              <p className="text-[11px] text-slate-400 mt-1 max-w-[220px]">Generating semantic vector representations for retrieval-augmented analysis.</p>
            </div>
          </div>
        )}

        {file && checkProcessedQuery.data?.is_processed && messages.length === 0 && (
          <div className="flex flex-col items-center justify-center min-h-full py-6">
            <div className="w-full max-w-[320px] p-5 rounded-lg border border-slate-250 bg-white shadow-sm text-center">
              <h4 className="text-xs font-semibold text-slate-705 uppercase tracking-wider mb-1">Session Initialized</h4>
              <p className="text-[11px] text-slate-400 leading-relaxed mb-5">Select a query template below or enter a custom prompt in the input workbench.</p>
               
              <div className="space-y-1.5">
                {questions.map((q, i) => (
                  <button 
                    key={i}
                    onClick={() => setInputValue(q)}
                    className="w-full text-left px-3 py-2 text-[11px] font-medium text-slate-600 bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded transition-colors flex items-center justify-between group"
                  >
                    <span className="truncate">{q}</span>
                    <ChevronRight size={12} className="opacity-50 group-hover:opacity-100 transition-opacity" />
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {messages.map((msg) => (
          <div key={msg.id} className={cn(
            "flex gap-2.5 animate-in fade-in duration-200",
            msg.sender === 'user' ? "flex-row-reverse" : "flex-row"
          )}>
            <div className={cn(
              "h-7 w-7 rounded border flex items-center justify-center shrink-0 text-xs font-semibold",
              msg.sender === 'user' ? "bg-primary text-white border-primary" : "bg-slate-50 text-slate-600 border-slate-200"
            )}>
              {msg.sender === 'user' ? <User size={13} strokeWidth={2} /> : <Cpu size={13} strokeWidth={2} />}
            </div>
            
            <div className={cn(
              "max-w-[85%] px-3.5 py-2.5 rounded text-xs leading-relaxed border shadow-none",
              msg.sender === 'user'
                ? "bg-primary text-white border-primary rounded-tr-none font-medium"
                : msg.failed
                  ? "bg-red-50 border-red-200 text-red-800 rounded-tl-none"
                  : "bg-white border-slate-200 text-slate-800 rounded-tl-none"
            )}>
              {msg.sender === 'bot' ? (
                <>
                  <div className="flex items-center gap-1 mb-1.5 text-[9px] font-semibold text-slate-450 uppercase tracking-wider">
                    System Response
                  </div>
                  {msg.text ? (
                    <MarkdownRenderer text={msg.text} />
                  ) : (
                    <Loader2 size={12} className="animate-spin text-slate-400" />
                  )}
                  {msg.sources && msg.sources.length > 0 && (
                    <SourcesPanel sources={msg.sources} />
                  )}
                </>
              ) : (
                <p className="whitespace-pre-wrap">{msg.text}</p>
              )}
            </div>
          </div>
        ))}

        {isStreaming && (
          <div className="flex gap-2.5 animate-pulse">
            <div className="h-7 w-7 rounded border bg-slate-50 border-slate-200 text-slate-400 flex items-center justify-center shrink-0">
              <Loader2 size={11} className="animate-spin" />
            </div>
            <div className="bg-white border border-slate-200 rounded rounded-tl-none px-3.5 py-2.5 max-w-[85%] flex items-center">
              <div className="flex gap-1.5">
                <div className="h-1.5 w-1.5 bg-slate-400 rounded-full animate-bounce" />
                <div className="h-1.5 w-1.5 bg-slate-400 rounded-full animate-bounce [animation-delay:0.2s]" />
                <div className="h-1.5 w-1.5 bg-slate-400 rounded-full animate-bounce [animation-delay:0.4s]" />
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="p-3 bg-slate-50 border-t border-slate-200/80">
        <div className="relative flex items-center">
          <Input
            placeholder={file ? "Query document contents..." : "Select a paper..."}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            disabled={!file || !checkProcessedQuery.data?.is_processed}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            className="h-10 pl-3.5 pr-11 bg-white border-slate-200 ring-primary/20 rounded text-xs placeholder:text-slate-450 transition-all focus-visible:ring-offset-0 focus-visible:ring-2"
          />
          <Button
            onClick={handleSend}
            disabled={!inputValue.trim() || isStreaming}
            className="absolute right-1 top-1 h-8 w-8 p-0 bg-primary hover:bg-primary/95 text-white rounded shadow-sm transition-colors"
          >
            <Send size={14} />
          </Button>
        </div>
        <p className="text-[9px] text-center text-slate-400 mt-2 font-medium tracking-wider uppercase">
          PaperMind Quantitative Ingestion Node
        </p>
      </div>
    </div>
  );
}

function SourcesPanel({ sources }: { sources: ISource[] }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mt-2 border-t border-slate-100 pt-1.5">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1 text-[9px] font-semibold text-slate-450 hover:text-slate-600 uppercase tracking-wider transition-colors cursor-pointer"
      >
        <FileText size={10} />
        Sources ({sources.length})
        <ChevronDown size={10} className={cn("transition-transform", open && "rotate-180")} />
      </button>
      {open && (
        <div className="mt-1.5 space-y-1.5">
          {sources.map((source, i) => (
            <div key={i} className="bg-slate-50 border border-slate-200 rounded p-2">
              <div className="flex items-center justify-between mb-1 text-[9px] font-medium text-slate-500">
                <span className="truncate">{source.document} · chunk {source.chunk_index}</span>
                <span className="shrink-0 ml-2 font-mono text-slate-400">
                  {source.score.toFixed(3)}
                </span>
              </div>
              <p className="text-[10px] leading-relaxed text-slate-600 line-clamp-3">
                {source.content}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function CodeBlock({ code, lang }: { code: string; lang: string }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  return (
    <div className="my-2 border border-slate-200 rounded-md overflow-hidden bg-slate-50 font-mono text-[11px] leading-relaxed shadow-sm">
      <div className="bg-slate-100/80 px-3 py-1.5 border-b border-slate-200 text-[10px] text-slate-500 uppercase font-sans font-medium flex justify-between items-center">
        <span>{lang || "code"}</span>
        <button
          onClick={handleCopy}
          className="text-slate-400 hover:text-slate-600 transition-colors flex items-center gap-1 cursor-pointer"
        >
          {copied ? <Check size={10} className="text-green-600" /> : <Copy size={10} />}
          <span>{copied ? "Copied" : "Copy"}</span>
        </button>
      </div>
      <pre className="p-3 overflow-x-auto text-slate-800">
        <code>{code}</code>
      </pre>
    </div>
  );
}

function MarkdownCode({ className, children, ...props }: ComponentPropsWithoutRef<"code">) {
  const isBlock = className?.includes("language-") || String(children).includes("\n");
  if (!isBlock) {
    return (
      <code
        className="px-1.5 py-0.5 rounded bg-slate-100 border border-slate-200/80 font-mono text-[10px] text-slate-800 font-medium mx-0.5"
        {...props}
      >
        {children}
      </code>
    );
  }
  const lang = /language-(\S+)/.exec(className ?? "")?.[1] ?? "";
  return <CodeBlock code={String(children).replace(/\n$/, "")} lang={lang} />;
}

function MarkdownRenderer({ text }: { text: string }) {
  return (
    <div className="space-y-2 [&_p]:my-1.5 [&_ul]:list-disc [&_ul]:pl-4 [&_ul]:space-y-1 [&_ul]:my-1.5 [&_ol]:list-decimal [&_ol]:pl-4 [&_ol]:space-y-1 [&_ol]:my-1.5 [&_li]:leading-relaxed [&_strong]:font-semibold [&_strong]:text-slate-900 [&_h1]:text-sm [&_h1]:font-semibold [&_h2]:text-xs [&_h2]:font-semibold [&_h3]:text-xs [&_h3]:font-semibold [&_a]:text-primary [&_a]:underline [&_blockquote]:border-l-2 [&_blockquote]:border-slate-200 [&_blockquote]:pl-2 [&_blockquote]:text-slate-600 [&_table]:text-[10px] [&_th]:border [&_th]:border-slate-200 [&_th]:px-1.5 [&_th]:py-1 [&_td]:border [&_td]:border-slate-200 [&_td]:px-1.5 [&_td]:py-1">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          pre: ({ children }) => <>{children}</>,
          code: MarkdownCode,
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}
