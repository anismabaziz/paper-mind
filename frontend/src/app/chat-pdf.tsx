import { MessageSquare, Send, ChevronRight, Loader2, Bot, User, Sparkles } from "lucide-react";
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
    <div className="flex flex-col h-full glass rounded-3xl overflow-hidden shadow-xl border-white/40">
      <div className="p-5 border-b border-white/20 bg-white/20 backdrop-blur-md flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="p-2 bg-emerald-500/10 rounded-lg text-emerald-600">
            <MessageSquare size={20} />
          </div>
          <h3 className="font-bold text-slate-800">Research Chat</h3>
        </div>
      </div>

      <div className="flex-grow overflow-y-auto p-4 space-y-6" ref={scrollRef}>
        {!file && (
          <div className="flex flex-col items-center justify-center h-full text-center p-8">
            <div className="w-16 h-16 bg-slate-100 rounded-2xl flex items-center justify-center mb-4 text-slate-300">
              <MessageSquare size={32} />
            </div>
            <p className="text-sm font-semibold text-slate-600">Select a paper to chat</p>
            <p className="text-xs text-slate-400 max-w-[200px] mt-1">Our AI will help you summarize and analyze its content instantly.</p>
          </div>
        )}

        {file && !checkProcessedQuery.data?.is_processed && (
           <div className="flex flex-col items-center justify-center h-full text-center p-8 space-y-4">
              <div className="relative">
                 <Loader2 className="animate-spin text-primary" size={48} />
                 <Sparkles className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-primary/40" size={20} />
              </div>
              <div>
                <p className="text-sm font-bold text-slate-800">Synthesizing Paper...</p>
                <p className="text-xs text-slate-400 mt-1 max-w-[220px]">Preparing vector embeddings for deep contextual analysis.</p>
              </div>
           </div>
        )}

        {file && checkProcessedQuery.data?.is_processed && messages.length === 0 && (
          <div className="flex flex-col items-center justify-center min-h-full py-10">
            <div className="glass p-6 rounded-3xl border-white/60 shadow-lg text-center max-w-[280px]">
               <Sparkles className="text-primary mx-auto mb-4" size={32} />
               <h4 className="font-bold text-slate-800 mb-2 text-lg">AI Ready</h4>
               <p className="text-xs text-slate-500 leading-relaxed mb-6">Ask anything about the researchers, methodology, or results of this paper.</p>
               
               <div className="space-y-2">
                 {questions.map((q, i) => (
                    <button 
                      key={i}
                      onClick={() => setInputValue(q)}
                      className="w-full text-left p-2.5 text-[11px] font-medium text-slate-600 bg-white/50 border border-white/20 rounded-xl hover:bg-primary/5 hover:border-primary/20 transition-all flex items-center justify-between group"
                    >
                      <span className="truncate">{q}</span>
                      <ChevronRight size={14} className="opacity-0 group-hover:opacity-100 transition-opacity" />
                    </button>
                 ))}
               </div>
            </div>
          </div>
        )}

        {messages.map((msg) => (
          <div key={msg.id} className={cn(
              "flex gap-3 animate-in fade-in slide-in-from-bottom-2 duration-300",
              msg.sender === 'user' ? "flex-row-reverse" : "flex-row"
          )}>
            <div className={cn(
                "h-8 w-8 rounded-xl flex items-center justify-center shadow-md shrink-0",
                msg.sender === 'user' ? "bg-primary text-white" : "bg-white text-emerald-600 border border-slate-100"
            )}>
              {msg.sender === 'user' ? <User size={14} strokeWidth={2.5} /> : <Bot size={14} strokeWidth={2.5} />}
            </div>
            
            <div className={cn(
                "max-w-[85%] p-4 rounded-2xl shadow-sm text-[13px] leading-relaxed",
                msg.sender === 'user' 
                  ? "bg-primary text-white font-medium rounded-tr-none" 
                  : "bg-white border border-slate-100 text-slate-700 rounded-tl-none"
            )}>
              {msg.sender === 'bot' && (
                <div className="flex items-center gap-1.5 mb-2 text-[10px] font-bold text-emerald-600/60 uppercase tracking-widest">
                  <Sparkles size={10} />
                  AI Analysis
                </div>
              )}
              <p className="whitespace-pre-wrap">{msg.text}</p>
            </div>
          </div>
        ))}

        {askMutation.isPending && (
          <div className="flex gap-3 animate-pulse">
            <div className="h-8 w-8 rounded-xl bg-white border border-slate-100 flex items-center justify-center shrink-0">
               <Loader2 size={12} className="animate-spin text-emerald-600" />
            </div>
            <div className="bg-white/60 border border-slate-100 rounded-2xl rounded-tl-none p-4 max-w-[85%]">
               <div className="flex gap-2">
                  <div className="h-2 w-2 bg-emerald-400 rounded-full animate-bounce" />
                  <div className="h-2 w-2 bg-emerald-400 rounded-full animate-bounce [animation-delay:0.2s]" />
                  <div className="h-2 w-2 bg-emerald-400 rounded-full animate-bounce [animation-delay:0.4s]" />
               </div>
            </div>
          </div>
        )}
      </div>

      <div className="p-4 bg-white/20 backdrop-blur-md border-t border-white/20">
        <div className="relative group">
          <Input
            placeholder={file ? "Type your command..." : "Select a paper..."}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            disabled={!file || !checkProcessedQuery.data?.is_processed}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            className="h-12 pl-4 pr-12 bg-white border-white ring-primary/20 rounded-2xl shadow-inner text-[13px] font-medium placeholder:text-slate-400 transition-all focus-visible:ring-offset-0 focus-visible:ring-4"
          />
          <Button 
            onClick={handleSend}
            disabled={!inputValue.trim() || askMutation.isPending}
            className="absolute right-1.5 top-1.5 h-9 w-9 p-0 bg-primary hover:bg-primary/90 text-white rounded-xl shadow-lg shadow-primary/20 transition-all hover:scale-105"
          >
            <Send size={16} />
          </Button>
        </div>
        <p className="text-[10px] text-center text-slate-400 mt-2 font-medium tracking-wide">
            Powered by PaperMind AI Integration
        </p>
      </div>
    </div>
  );
}
