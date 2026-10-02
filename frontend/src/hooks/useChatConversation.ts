import { useCallback, useEffect, useRef, useState } from "react";
import { runChatTurn, type TurnUpdate } from "@/lib/chat-turn";
import type { IMessage } from "@/services/files";
import type { File as DbFile } from "@/types/db";
import type { ChatMessage } from "@/features/chat/chat-message";

/**
 * One Document's conversation, and the one stream that may be running in it.
 *
 * A stream belongs to the Document it was started for. Its identity is checked
 * on every chunk, because an abort and the next Document's stream can race: a
 * late token landing in the wrong conversation is worse than a lost one, and
 * the reader would have no way of telling which answer it belonged to.
 */
export function useChatConversation({
  file,
  storedMessages,
  /** Questions are refused while the paper is being indexed. */
  blocked,
}: {
  file: DbFile | null;
  /** The stored turns, or undefined while they load. */
  storedMessages: IMessage[] | undefined;
  blocked: boolean;
}) {
  const [value, setValue] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [thinking, setThinking] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const stickToBottomRef = useRef(true);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const streamControllerRef = useRef<AbortController | null>(null);
  const fileIdRef = useRef<string | null>(null);
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

  // Drop the previous conversation the moment the Document changes, instead of
  // flashing its history until the new query resolves.
  useEffect(() => {
    setMessages([]);
    stickToBottomRef.current = true;
  }, [watchedFileId]);

  useEffect(() => {
    if (!watchedFileId || !storedMessages) return;
    setMessages(fromStoredMessages(storedMessages));
  }, [watchedFileId, storedMessages]);

  useEffect(() => {
    if (stickToBottomRef.current) endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, thinking]);

  useEffect(() => {
    inputRef.current?.focus();
  }, [file]);

  const patchMessage = useCallback((id: string, update: TurnUpdate) => {
    setMessages((prev) =>
      prev.flatMap((msg) => {
        if (msg.id !== id) return [msg];
        const next = update(msg);
        return next ? [next] : [];
      }),
    );
  }, []);

  const send = useCallback(
    async (text: string) => {
      const body = text.trim();
      if (!body || !file || thinking || blocked) return;
      setValue("");
      setThinking(true);
      const botId = crypto.randomUUID();
      setMessages((current) => [
        ...current,
        { id: crypto.randomUUID(), text: body, sender: "user" },
        { id: botId, text: "", sender: "bot", question: body },
      ]);

      streamControllerRef.current?.abort();
      const controller = new AbortController();
      streamControllerRef.current = controller;
      const activeFileId = file.id;
      const isStale = () => fileIdRef.current !== activeFileId || controller.signal.aborted;

      try {
        await runChatTurn({
          body,
          filename: file.name,
          botId,
          signal: controller.signal,
          isStale,
          patch: patchMessage,
        });
      } finally {
        setThinking(false);
        inputRef.current?.focus();
      }
    },
    [blocked, file, patchMessage, thinking],
  );

  return {
    value,
    setValue,
    messages,
    thinking,
    send,
    endRef,
    scrollContainerRef,
    inputRef,
    /** Scrolling up is a decision to stop following the newest answer. */
    onScroll: () => {
      const el = scrollContainerRef.current;
      if (!el) return;
      stickToBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
    },
  };
}

/** Stored turns, as the pane renders them. */
function fromStoredMessages(turns: IMessage[]): ChatMessage[] {
  return turns.map((turn) => ({
    id: turn.id,
    text: turn.text,
    sender: turn.sender,
    sources: turn.sources,
    claims: turn.claims,
    // An answer is only as grounded as the claims that cite something: a stored
    // answer whose claims name no passage is not grounded.
    grounded:
      turn.sender === "bot" && turn.turn_status === "answered"
        ? (turn.claims ?? []).some((claim) => claim.sources.length > 0)
        : undefined,
    // An abstention stored in history is still an abstention, not an answer
    // that came back empty.
    abstained: turn.turn_status === "abstained",
    abstentionReason: turn.turn_abstention_reason ?? undefined,
  }));
}