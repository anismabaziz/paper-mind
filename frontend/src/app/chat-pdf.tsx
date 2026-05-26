import { MessageSquare, Send, ChevronRight, Loader2, Cpu, User, Copy, Check } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import usePdfStore from "@/store/pdf-state";
import { useEffect, useState, useRef } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { checkIsProcessed, askQuestion, getMessages } from "@/services/files";
import { cn } from "@/lib/utils";

export default function ChatPDF() {
  const { file } = usePdfStore();
  const queryClient = useQueryClient();
  const scrollRef = useRef<HTMLDivElement>(null);
  
  const questions = [
    "What is the main topic of this document?",
    "Summarize the key findings",
    "Methodology used in this paper?",
  ];
  
  const [inputValue, setInputValue] = useState("");
  const [messages, setMessages] = useState<{id: string, text: string, sender: 'user' | 'bot'}[]>([]);

  const checkProcessedQuery = useQuery({
    queryKey: [file?.name, "is-processed"],
    queryFn: () => checkIsProcessed(file!),
    enabled: !!file,
    refetchInterval: (query) => {
      // @ts-ignore
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
        setMessages(messagesQuery.data.messages as any);
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

  const askMutation = useMutation({
    mutationFn: (query: string) => askQuestion(query, file!.name),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [file?.name, "messages"] });
    },
    onError: () => {
      setMessages((prev) => [
        ...prev,
        { id: crypto.randomUUID(), text: 'Connection lost. Please ensure the backend server is active.', sender: 'bot' }
      ]);
    }
  });

  const handleSend = () => {
    if (!inputValue.trim() || !file || askMutation.isPending) return;
    const q = inputValue.trim();
    setInputValue("");
    setMessages((prev) => [...prev, { id: crypto.randomUUID(), text: q, sender: 'user' }]);
    askMutation.mutate(q);
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
                : "bg-white border-slate-200 text-slate-800 rounded-tl-none"
            )}>
              {msg.sender === 'bot' ? (
                <>
                  <div className="flex items-center gap-1 mb-1.5 text-[9px] font-semibold text-slate-450 uppercase tracking-wider">
                    System Response
                  </div>
                  <MarkdownRenderer text={msg.text} />
                </>
              ) : (
                <p className="whitespace-pre-wrap">{msg.text}</p>
              )}
            </div>
          </div>
        ))}

        {askMutation.isPending && (
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
            disabled={!inputValue.trim() || askMutation.isPending}
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

function renderInlineCode(text: string) {
  const codeParts = text.split(/(`.*?`)/g);
  return codeParts.map((cPart, cIdx) => {
    if (cPart.startsWith("`") && cPart.endsWith("`")) {
      const codeText = cPart.slice(1, -1);
      return (
        <code
          key={cIdx}
          className="px-1.5 py-0.5 rounded bg-slate-100 border border-slate-200/80 font-mono text-[10px] text-slate-800 font-medium mx-0.5"
        >
          {codeText}
        </code>
      );
    }
    return cPart;
  });
}

function renderInlineMarkdown(text: string) {
  const boldParts = text.split(/(\*\*.*?\*\*)/g);
  return boldParts.map((bPart, bIdx) => {
    if (bPart.startsWith("**") && bPart.endsWith("**")) {
      const boldText = bPart.slice(2, -2);
      return (
        <strong key={bIdx} className="font-semibold text-slate-900">
          {renderInlineCode(boldText)}
        </strong>
      );
    }
    return <span key={bIdx}>{renderInlineCode(bPart)}</span>;
  });
}

function MarkdownRenderer({ text }: { text: string }) {
  const parts = text.split(/(```[\s\S]*?```)/g);

  return (
    <div className="space-y-2">
      {parts.map((part, index) => {
        if (part.startsWith("```")) {
          const lines = part.split("\n");
          const firstLine = lines[0];
          const lang = firstLine.replace("```", "").trim();
          const code = lines.slice(1, -1).join("\n");
          return <CodeBlock key={index} code={code} lang={lang} />;
        } else {
          const paragraphs = part.split("\n\n");
          return paragraphs.map((para, paraIdx) => {
            const trimmed = para.trim();
            if (!trimmed) return null;

            const lines = trimmed.split("\n");
            const isBulletList = lines.every(
              (line) => line.trim().startsWith("- ") || line.trim().startsWith("* ")
            );

            if (isBulletList) {
              return (
                <ul key={paraIdx} className="list-disc pl-4 space-y-1 my-1.5">
                  {lines.map((line, lineIdx) => {
                    const content = line.replace(/^[-*]\s+/, "");
                    return <li key={lineIdx}>{renderInlineMarkdown(content)}</li>;
                  })}
                </ul>
              );
            }

            return (
              <p key={paraIdx} className="my-1.5">
                {renderInlineMarkdown(para)}
              </p>
            );
          });
        }
      })}
    </div>
  );
}
