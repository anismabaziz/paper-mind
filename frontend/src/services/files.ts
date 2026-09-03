import client from "./client";
import { File as FileType } from "@/types/db";

interface IGetFiles {
  files: FileType[];
}
export async function getFiles() {
  return (await client.get<IGetFiles>("/files")).data;
}

interface IUploadFile {
  message: string;
  file: FileType;
}
export async function uploadFile(file: File) {
  const formData = new FormData();
  formData.append("file", file);
  return (
    await client.post<IUploadFile>("/upload", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    })
  ).data;
}

interface IDeleteFile {
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
}

export async function checkIsProcessed(file: FileType) {
  return (
    await client.post<ICheckIsProcessed>("/file/is-processed", {
      filename: file.name,
    })
  ).data;
}

interface IProcessFile {
  message: string;
}
export async function processFile(file: FileType) {
  return (
    await client.post<IProcessFile>("/process-file", {
      filename: file.name,
    })
  ).data;
}

export interface ISource {
  content: string;
  document: string;
  chunk_index: number;
  score: number;
}

interface IStreamHandlers {
  onToken: (text: string) => void;
  onError: (message: string) => void;
  onDone: (sources: ISource[]) => void;
}

export async function chatStream(
  query: string,
  filename: string,
  handlers: IStreamHandlers,
  token?: string
) {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (token) headers.Authorization = `Bearer ${token}`;

  const response = await fetch(`${client.defaults.baseURL}/response`, {
    method: "POST",
    headers,
    body: JSON.stringify({ query, filename }),
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
        handlers.onDone((event.data.sources as ISource[]) ?? []);
    }
  }
}

function parseSSEBlock(block: string): { name: string; data: Record<string, unknown> } | null {
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

export type SSEEvent =
  | { event: "token"; text: string }
  | { event: "error"; error: string }
  | { event: "done"; done: boolean; sources: ISource[] };

export interface IMessage {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  sources: ISource[];
  created_at: string;
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
