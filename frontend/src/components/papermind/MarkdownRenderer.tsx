import { useState, isValidElement, type ComponentPropsWithoutRef } from "react";
import { Copy, Check } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

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

export function MarkdownRenderer({ text }: { text: string }) {
  return (
    <div className="font-serif text-[0.94rem] leading-[1.65] text-pretty [&_p]:my-2 [&_ul]:list-disc [&_ul]:pl-4 [&_ol]:list-decimal [&_ol]:pl-4 [&_li]:leading-relaxed [&_strong]:font-semibold [&_a]:text-marker [&_a]:underline [&_blockquote]:border-l-2 [&_blockquote]:border-rule [&_blockquote]:pl-3 [&_blockquote]:italic [&_blockquote]:text-ink-soft [&_h1]:text-base [&_h1]:font-semibold [&_h2]:text-sm [&_h2]:font-semibold">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={{ pre: MarkdownPre, code: MarkdownInlineCode }}>
        {text}
      </ReactMarkdown>
    </div>
  );
}
