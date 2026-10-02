import { ArrowUp, CornerDownLeft } from "lucide-react";
import { Textarea } from "@/components/ui/textarea";
import { suggestedPrompts } from "./chat-message";

type Props = {
  value: string;
  onChange: (value: string) => void;
  onSend: (question: string) => void;
  disabled: boolean;
  busy: boolean;
  placeholder: string;
  inputRef: React.RefObject<HTMLTextAreaElement | null>;
};

/**
 * Where a question is written, and the one control that sends it.
 *
 * Enter sends and Shift+Enter does not, which is what a reader expects from a
 * field they write multi-line questions into.
 */
export default function ChatComposer({
  value,
  onChange,
  onSend,
  disabled,
  busy,
  placeholder,
  inputRef,
}: Props) {
  return (
    <div className="border-t border-rule px-5 py-4">
      <div className="mb-3 flex flex-wrap gap-1.5">
        {suggestedPrompts.map((prompt) => (
          <button
            key={prompt}
            type="button"
            onClick={() => onSend(prompt)}
            disabled={disabled}
            className="rounded-full border border-rule px-2.5 py-1 text-[0.7rem] text-ink-soft transition-colors hover:border-ink hover:text-ink disabled:opacity-40"
          >
            {prompt}
          </button>
        ))}
      </div>

      <div className="rounded-sm border border-rule bg-card px-3 py-2.5 transition-colors focus-within:border-ink">
        <Textarea
          ref={inputRef}
          rows={2}
          data-testid="chat-input"
          aria-label="Ask this paper a question"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              onSend(value);
            }
          }}
          placeholder={placeholder}
          disabled={disabled}
          className="min-h-0 resize-none border-0 bg-transparent px-0 py-0 text-[0.85rem] leading-relaxed placeholder:text-ink-faint focus-visible:ring-0 disabled:bg-transparent disabled:opacity-60"
        />
        <div className="mt-1 flex items-center justify-between">
          <span className="label-meta flex items-center gap-1">
            <CornerDownLeft className="size-2.5" /> to send
          </span>
          <button
            type="button"
            onClick={() => onSend(value)}
            disabled={!value.trim() || busy || disabled}
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
  );
}