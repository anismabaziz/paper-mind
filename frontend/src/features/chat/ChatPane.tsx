import { useFileStatus, useFileMessages, useReindex } from "@/hooks/useFiles";
import { getErrorMessage } from "@/lib/api-error";
import { useChatConversation } from "@/hooks/useChatConversation";
import usePdfStore from "@/store/pdf-state";
import { isIndexStale, isIngestionActive } from "@/types/db";
import ChatComposer from "./ChatComposer";
import ChatConversationStates from "./ChatConversationStates";
import ChatMessageItem from "./ChatMessageItem";
import StaleIndexNotice from "./StaleIndexNotice";
import { Skeleton } from "@/components/ui/skeleton";

/**
 * The reading companion: questions about the open Document, and the passages
 * each answer was read from.
 *
 * Questions are refused rather than queued while a paper indexes or while its
 * index no longer matches the app's settings. Both would fail at the search
 * step, and an answer to a question about passages the app never searched is
 * not worth having.
 */
export function ChatPane() {
  const { file } = usePdfStore();
  const checkProcessedQuery = useFileStatus(file);
  const ingestionJob = checkProcessedQuery.data?.ingestion;
  const documentIndex = checkProcessedQuery.data?.index;
  const isStaleIndex = isIndexStale(documentIndex);
  const isIndexing = isIngestionActive(ingestionJob?.state);
  const isProcessed = checkProcessedQuery.data?.is_processed;

  const messagesQuery = useFileMessages(
    file,
    checkProcessedQuery.data?.is_processed === true && !isIndexing,
  );
  const reindex = useReindex();

  const chat = useChatConversation({
    file,
    storedMessages: messagesQuery.data?.messages,
    blocked: isIndexing,
  });

  const inputDisabled = !file || !isProcessed || isIndexing || isStaleIndex;
  const showStaleIndex = Boolean(file && isStaleIndex && !isIndexing && documentIndex);

  return (
    <section
      className="flex w-full min-w-0 shrink-0 flex-col border-l border-rule bg-background lg:w-[26rem]"
      data-testid="chat-pane"
    >
      <header className="flex h-14 items-center border-b border-rule px-5">
        <div>
          <p className="text-[0.82rem] font-medium">Reading companion</p>
          <p className="label-meta">Answers cite this paper only</p>
        </div>
      </header>

      <div
        ref={chat.scrollContainerRef}
        data-testid="chat-messages"
        onScroll={chat.onScroll}
        className="scroll-slim flex-1 space-y-7 overflow-y-auto px-5 py-6"
      >
        <ChatConversationStates
          hasFile={Boolean(file)}
          statusError={
            file && checkProcessedQuery.isError && !checkProcessedQuery.data
              ? {
                  text: getErrorMessage(checkProcessedQuery.error) || null,
                  onRetry: () => void checkProcessedQuery.refetch(),
                }
              : null
          }
          statusStale={Boolean(file && checkProcessedQuery.data && checkProcessedQuery.isError)}
          onRetryStatus={() => void checkProcessedQuery.refetch()}
          ready={Boolean(isProcessed)}
          indexing={isIndexing}
          showStaleIndex={showStaleIndex}
          conversationError={
            file && messagesQuery.isError
              ? {
                  text: getErrorMessage(messagesQuery.error) || null,
                  onRetry: () => void messagesQuery.refetch(),
                }
              : null
          }          showPrompts={Boolean(
            file &&
              isProcessed &&
              !isIndexing &&
              !isStaleIndex &&
              !messagesQuery.isError &&
              chat.messages.length === 0 &&
              !chat.thinking,
          )}
          onPrompt={(prompt) => void chat.send(prompt)}
        />

        {showStaleIndex && documentIndex && <StaleIndexNotice index={documentIndex} />}

        {chat.messages.map((message) =>
          message.sender === "user" ? (
            <div
              key={message.id}
              className="rise-in flex justify-end"
              data-testid="chat-message"
              data-sender="user"
            >
              <p className="max-w-[85%] rounded-lg rounded-br-[2px] bg-ink px-3.5 py-2.5 text-[0.85rem] leading-snug text-paper">
                {message.text}
              </p>
            </div>
          ) : (
            <ChatMessageItem
              key={message.id}
              message={message}
              canRetry={!inputDisabled}
              busy={chat.thinking}
              onRetry={(question) => void chat.send(question)}
              reindexPending={reindex.isPending}
              onReindex={() => file && reindex.mutate(file.name)}
            />
          ),
        )}

        {chat.thinking && chat.messages[chat.messages.length - 1]?.sender !== "bot" && (
          <div className="rise-in" role="status" aria-label="Reading passages">
            <p className="label-meta mb-2">Reading passages…</p>
            <div className="space-y-2">
              {[90, 76, 58].map((w, i) => (
                <Skeleton
                  key={i}
                  className="h-2 rounded-full bg-ink/10"
                  style={{ width: `${w}%`, animationDelay: `${i * 120}ms` }}
                />
              ))}
            </div>
            <span className="sr-only">Reading passages…</span>
          </div>
        )}
        <div ref={chat.endRef} />
      </div>

      <ChatComposer
        value={chat.value}
        onChange={chat.setValue}
        onSend={(question) => void chat.send(question)}
        disabled={inputDisabled}
        busy={chat.thinking}
        placeholder={
          !file
            ? "Select a paper…"
            : isStaleIndex
              ? "Reindex this paper to ask questions…"
              : isProcessed
                ? "Ask this paper something…"
                : "Indexing — questions paused…"
        }
        inputRef={chat.inputRef}
      />
    </section>
  );
}