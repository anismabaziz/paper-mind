import client, { apiBaseUrl } from "./client";
import { File as FileType, DocumentIndex, IngestionJob } from "@/types/db";

interface IGetFiles {
  files: FileType[];
}
export async function getFiles() {
  return (await client.get<IGetFiles>("/files")).data;
}

interface IUploadFile {
  message: string;
  file: FileType;
  job: IngestionJob | null;
}
export async function uploadFile(file: File) {
  const formData = new FormData();
  formData.append("file", file);
  return (await client.post<IUploadFile>("/upload", formData)).data;
}

export interface IDeleteFile {
  message: string;
}
export async function deleteFile(file: FileType) {
  return (
    await client.delete<IDeleteFile>("/files/remove", {
      params: {
        path: file.name,
      },
    })
  ).data;
}
interface ICheckIsProcessed {
  is_processed: boolean;
  ingestion: IngestionJob | null;
  index: DocumentIndex;
}

export async function checkIsProcessed(file: FileType) {
  return (
    await client.post<ICheckIsProcessed>("/file/is-processed", {
      filename: file.name,
    })
  ).data;
}

export async function reindexFile(name: string) {
  const response = await client.post<{ job: IngestionJob }>(
    `/files/${encodeURIComponent(name)}/reindex`
  );
  return response.data.job;
}

export async function retryIngestionJob(name: string) {
  const response = await client.post<{ job: IngestionJob }>(
    `/ingestion-jobs/${encodeURIComponent(name)}/retry`
  );
  return response.data.job;
}

export async function cancelIngestionJob(name: string) {
  const response = await client.post<{ job: IngestionJob; cancelled: boolean }>(
    `/ingestion-jobs/${encodeURIComponent(name)}/cancel`
  );
  return response.data.job;
}

interface IMarkOpened {
  last_opened_at: string | null;
}

export async function markFileOpened(file: FileType) {
  return (
    await client.post<IMarkOpened>("/file/opened", {
      filename: file.name,
    })
  ).data;
}

export interface ISource {
  content: string;
  document: string;
  chunk_index: number;
  score: number;
  page: number | null;
}

export interface IRetrievalResult {
  method: "dense" | "sparse" | "hybrid";
  outcome: "success" | "empty";
}

/** Why the model could not answer. Each one reads differently to the user. */
export type ChatFailureCategory = "provider" | "timeout" | "empty_output";

export interface IChatDone {
  sources: ISource[];
  retrieval?: IRetrievalResult;
  /** The answer was cut short — a token cap, not a finished sentence. */
  truncated: boolean;
  finishReason: string | null;
}

export interface IStreamHandlers {
  onStart?: (info: { turnId: string }) => void;
  onToken: (text: string) => void;
  onProviderError?: (message: string, category: ChatFailureCategory) => void;
  onPersistenceError?: (message: string) => void;
  onCancelled?: (reason: string) => void;
  onDone: (result: IChatDone) => void;
}

/**
 * A stream that does not follow the event contract.
 *
 * Thrown rather than swallowed: a loader that never resolves is worse than a
 * readable error, because the user cannot tell a slow answer from a dead one.
 */
export class StreamProtocolError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "StreamProtocolError";
  }
}

const FAILURE_CATEGORIES: ChatFailureCategory[] = ["provider", "timeout", "empty_output"];

const ENDED_WITHOUT_ANSWER =
  "The answer stream ended before it finished. Please ask the question again.";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readString(data: Record<string, unknown>, key: string, event: string): string {
  const value = data[key];
  if (typeof value !== "string") {
    throw new StreamProtocolError(`The ${event} event has no ${key}.`);
  }
  return value;
}

/**
 * Dispatch one event to its handler.
 *
 * An event this protocol does not define, or one whose payload does not match
 * it, is a protocol error. Ignoring it would leave the loader spinning on a
 * stream the client no longer understands.
 */
function dispatch(
  event: { name: string; data: Record<string, unknown> },
  handlers: IStreamHandlers,
): boolean {
  const { name, data } = event;
  switch (name) {
    case "start":
      handlers.onStart?.({ turnId: readString(data, "turn_id", name) });
      return false;
    case "token":
      handlers.onToken(readString(data, "text", name));
      return false;
    case "provider_error": {
      const category = data.category;
      if (!FAILURE_CATEGORIES.includes(category as ChatFailureCategory)) {
        throw new StreamProtocolError("The provider_error event has no category.");
      }
      handlers.onProviderError?.(
        readString(data, "error", name),
        category as ChatFailureCategory,
      );
      return true;
    }
    case "persistence_error":
      handlers.onPersistenceError?.(readString(data, "error", name));
      return true;
    case "cancelled":
      handlers.onCancelled?.(readString(data, "reason", name));
      return true;
    case "done": {
      const { sources, retrieval, truncated, finish_reason: finishReason } = data;
      handlers.onDone({
        sources: Array.isArray(sources) ? (sources as ISource[]) : [],
        retrieval: retrieval as IRetrievalResult | undefined,
        truncated: truncated === true,
        finishReason: typeof finishReason === "string" ? finishReason : null,
      });
      return true;
    }
    default:
      throw new StreamProtocolError(`The stream sent an unknown ${name} event.`);
  }
}

export async function chatStream(
  query: string,
  filename: string,
  handlers: IStreamHandlers,
  options?: { signal?: AbortSignal }
) {
  const response = await fetch(`${apiBaseUrl}/response`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ query, filename }),
    signal: options?.signal,
  });

  if (!response.ok || !response.body) {
    const message = await response.json().catch(() => null);
    throw new Error(message?.error ?? `Request failed (${response.status})`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finished = false;

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) !== -1) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        if (isKeepAlive(block)) continue;
        const event = parseSSEBlock(block);
        if (!event) {
          throw new StreamProtocolError("The stream sent an unreadable event.");
        }
        // A terminal event ends the stream; anything after it is not an answer.
        if (dispatch(event, handlers)) {
          finished = true;
          break;
        }
      }
      if (finished) break;
    }
  } finally {
    reader.releaseLock?.();
  }

  if (!finished) throw new StreamProtocolError(ENDED_WITHOUT_ANSWER);
}

/** A block carrying only comment lines keeps the connection warm, not a message. */
function isKeepAlive(block: string): boolean {
  const lines = block.split("\n").filter((line) => line.trim() !== "");
  return lines.length > 0 && lines.every((line) => line.startsWith(":"));
}

export function parseSSEBlock(block: string): { name: string; data: Record<string, unknown> } | null {
  let name = "message";
  let data = "";
  for (const line of block.split("\n")) {
    if (line.startsWith("event: ")) name = line.slice(7);
    else if (line.startsWith("data: ")) data += line.slice(6);
  }
  if (!data) return null;
  try {
    const parsed: unknown = JSON.parse(data);
    return isRecord(parsed) ? { name, data: parsed } : null;
  } catch {
    return null;
  }
}

export interface IMessage {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  sources: ISource[];
  created_at: string;
  turn_id: string;
  turn_sequence: number;
  turn_status: 'pending' | 'answered' | 'failed' | 'cancelled' | 'unanswered';
}

interface IGetMessages {
  messages: IMessage[];
}

export async function getMessages(filename: string) {
  return (
    await client.get<IGetMessages>("/messages", {
      params: { filename }
    })
  ).data;
}

export interface FileMetaOutlineEntry {
  title: string;
  page: number;
  level: number;
}

export interface FileMeta {
  pageCount: number;
  outline: FileMetaOutlineEntry[];
}

export async function getFileMeta(name: string) {
  return (
    await client.get<FileMeta>(`/files/${encodeURIComponent(name)}/meta`)
  ).data;
}
