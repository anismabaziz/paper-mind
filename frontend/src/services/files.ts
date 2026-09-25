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

export interface IChatDone {
  sources: ISource[];
  retrieval?: IRetrievalResult;
}

interface IStreamHandlers {
  onToken: (text: string) => void;
  onError: (message: string) => void;
  onDone: (result: IChatDone) => void;
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

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const event = parseSSEBlock(block);
      if (!event) continue;

      if (event.name === "token") handlers.onToken(event.data.text as string);
      else if (event.name === "error") handlers.onError(event.data.error as string);
      else if (event.name === "done")
        handlers.onDone({
          sources: (event.data.sources as ISource[]) ?? [],
          retrieval: event.data.retrieval as IRetrievalResult | undefined,
        });
    }
  }
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
    return { name, data: JSON.parse(data) as Record<string, unknown> };
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
