import "@testing-library/jest-dom/vitest";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ChatPane } from "./ChatPane";
import { chatStream, reindexFile, StreamProtocolError, type IStreamHandlers } from "@/services/files";
import { checkIsProcessed, getMessages, type IChatDone, type ISource } from "@/services/files";
import usePdfStore from "@/store/pdf-state";
import type { DocumentIndex, File } from "@/types/db";

vi.mock("@/services/files", async (importOriginal) => {
  const original = await importOriginal<typeof import("@/services/files")>();
  return {
    ...original,
    chatStream: vi.fn(),
    checkIsProcessed: vi.fn(),
    getMessages: vi.fn(),
    reindexFile: vi.fn(),
  };
});

vi.mock("react-pdf", () => ({
  Document: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Page: () => null,
}));

const manifest: NonNullable<DocumentIndex["manifest"]> = {
  content_hash: "abc123",
  parser: "auto",
  parser_version: "pymupdf-docling-page-chunks-v1",
  chunk_size_tokens: 512,
  chunk_overlap_tokens: 50,
  embedding_model: "BAAI/bge-m3",
  embedding_revision: "",
  vector_dimension: 1024,
  sparse_method: "hashed-tf-qdrant-idf-v1",
  sparse_tokenizer_version: "lowercase-regex-stopwords-v1",
  reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2",
  reranker_revision: "",
  reranker_enabled: false,
  collection_name: "pdf-index",
  collection_schema_version: "named-dense-sparse-v1",
  index_generation: 2,
};

const file: File = {
  id: "file-1",
  is_processed: true,
  last_opened_at: null,
  deletion_state: "active",
  deletion_error: null,
  deletion_attempts: 0,
  ingestion: null,
  index: null,
  metadata: { content_type: "application/pdf", size: 2048 },
  name: "doc.pdf",
  title: "Attention Is All You Need",
  original_filename: "attention.pdf",
  url: "http://localhost/storage/doc.pdf",
};

function renderPane() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  render(
    <QueryClientProvider client={queryClient}>
      <ChatPane />
    </QueryClientProvider>
  );
  return queryClient;
}

beforeEach(() => {
  // jsdom has no layout, so the scroll-into-view the pane runs on mount is a stub.
  Element.prototype.scrollIntoView = vi.fn();
  usePdfStore.getState().setFile(file);
  vi.mocked(getMessages).mockResolvedValue({ messages: [] });
  vi.mocked(reindexFile).mockResolvedValue({
    id: "job-2",
    file_id: "file-1",
    filename: "doc.pdf",
    generation: 3,
    state: "queued",
    stage: "queued",
    progress: 0,
    attempt: 1,
    error_category: null,
    error_message: null,
    worker_id: null,
    created_at: "2026-09-25T10:00:00+00:00",
    updated_at: null,
    started_at: null,
    finished_at: null,
    heartbeat_at: null,
  });
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("ChatPane stale index", () => {
  it("shows the active values and the setting that needs a reindex", async () => {
    vi.mocked(checkIsProcessed).mockResolvedValue({
      is_processed: true,
      ingestion: null,
      index: {
        state: "stale",
        manifest,
        runtime_manifest: { ...manifest, embedding_revision: "9f1c2ab" },
        changes: ["embedding_revision"],
        change_details: [
          { field: "embedding_revision", label: "embedding revision", indexed: "", current: "9f1c2ab" },
        ],
      },
    });
    renderPane();

    expect(await screen.findByText(/Index needs reindexing/)).toBeInTheDocument();
    expect(screen.getByText("embedding revision")).toBeInTheDocument();
    expect(screen.getByText("unpinned → 9f1c2ab")).toBeInTheDocument();
    expect(screen.getByText(/Active index generation 2/)).toBeInTheDocument();
  });

  it("queues a reindex and pauses questions until it succeeds", async () => {
    vi.mocked(checkIsProcessed).mockResolvedValue({
      is_processed: true,
      ingestion: null,
      index: {
        state: "stale",
        manifest,
        runtime_manifest: { ...manifest, chunk_size_tokens: 1024 },
        changes: ["chunk_size_tokens"],
        change_details: [
          { field: "chunk_size_tokens", label: "chunk size", indexed: 512, current: 1024 },
        ],
      },
    });
    renderPane();

    const input = await screen.findByPlaceholderText(/Reindex this paper/);
    expect(input).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: /Reindex this paper/ }));

    await waitFor(() => expect(reindexFile).toHaveBeenCalledWith("doc.pdf"));
  });

  it("leaves questions available when the manifest still matches", async () => {
    vi.mocked(checkIsProcessed).mockResolvedValue({
      is_processed: true,
      ingestion: null,
      index: {
        state: "ready",
        manifest,
        runtime_manifest: manifest,
        changes: [],
        change_details: [],
      },
    });
    renderPane();

    expect(await screen.findByPlaceholderText(/Ask this paper something/)).toBeEnabled();
    expect(screen.queryByText(/Index needs reindexing/)).not.toBeInTheDocument();
  });
});

/** Renders the pane, asks a question, and runs the given stream script. */
async function ask(run: (handlers: IStreamHandlers) => void) {
  vi.mocked(chatStream).mockImplementation(async (_query, _file, handlers) => {
    run(handlers);
  });
  renderPane();
  const input = await screen.findByPlaceholderText(/Ask this paper something/);
  await act(async () => {
    fireEvent.change(input, { target: { value: "what?" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
  });
  return vi.mocked(chatStream).mock.calls[0];
}

function outcomeOf(label: string | RegExp): string | null | undefined {
  return screen.getByText(label).closest("[data-outcome]")?.getAttribute("data-outcome");
}

describe("ChatPane answer outcomes", () => {
  beforeEach(() => {
    vi.mocked(checkIsProcessed).mockResolvedValue({
      is_processed: true,
      ingestion: null,
      index: {
        state: "ready",
        manifest,
        runtime_manifest: manifest,
        changes: [],
        change_details: [],
      },
    });
  });

  it("sends the question for the open document", async () => {
    const [query, filename, handlers] = await ask((h) =>
      h.onDone?.({ sources: [], claims: [], grounded: false, promptVersion: "grounded-claims-v1", truncated: false, finishReason: "stop" }),
    );

    expect(query).toBe("what?");
    expect(filename).toBe("doc.pdf");
    expect(typeof handlers.onToken).toBe("function");
  });

  it("shows the answer as it streams in", async () => {
    await ask((h) => {
      h.onToken?.("The answer ");
      h.onToken?.("is 42.");
    });

    expect(screen.getByText("The answer is 42.")).toBeInTheDocument();
  });

  it.each([
    ["provider", "The model is unavailable."],
    ["timeout", "This answer took too long."],
    ["empty_output", "The model returned nothing."],
  ] as const)("shows a %s failure as its own outcome", async (category, message) => {
    await ask((h) => h.onProviderError?.(message, category));

    expect(screen.getByText(message)).toBeInTheDocument();
    expect(screen.getByText("Synthesis · failed")).toBeInTheDocument();
    expect(outcomeOf("Synthesis · failed")).toBe(category);
  });

  it("shows a save failure as its own outcome", async () => {
    await ask((h) => h.onPersistenceError?.("The answer could not be saved."));

    expect(outcomeOf("Synthesis · failed")).toBe("persistence");
  });

  it("says why a cancelled answer was not saved, not a loader", async () => {
    await ask((h) => h.onCancelled?.("document deleted"));

    expect(
      screen.getByText(/stopped \(document deleted\) before it finished/),
    ).toBeInTheDocument();
    expect(outcomeOf(/Synthesis · stopped/)).toBe("cancelled");
  });

  it("keeps the answer a cancellation interrupted", async () => {
    await ask((h) => {
      h.onToken?.("Half an answer");
      h.onCancelled?.("document deleted");
    });

    expect(screen.getByText(/Half an answer/)).toBeInTheDocument();
    expect(screen.getByText(/Stopped before this answer was saved/)).toBeInTheDocument();
  });

  it("flags an answer the model cut short", async () => {
    await ask((h) =>
      h.onDone?.({ sources: [], claims: [], grounded: false, promptVersion: "grounded-claims-v1", truncated: true, finishReason: "length" }),
    );

    expect(outcomeOf("Synthesis · truncated")).toBe("truncated");
    expect(screen.getByText(/hit its answer limit/)).toBeInTheDocument();
  });

  it("says the paper has nothing to answer from, without blaming a failure", async () => {
    await ask((h) =>
      h.onAbstained?.({
        message: "This paper has no passage that speaks to this question.",
        reason: "no_evidence",
      }),
    );

    expect(
      screen.getByText("This paper has no passage that speaks to this question."),
    ).toBeInTheDocument();
    expect(outcomeOf("Synthesis · abstained")).toBe("no_evidence");
    expect(screen.queryByText(/Reading passages/)).not.toBeInTheDocument();
  });

  it("offers a reindex when the matched passages could not be used", async () => {
    await ask((h) =>
      h.onAbstained?.({
        message: "The passages this question matched could not be read or cited.",
        reason: "evidence_unusable",
      }),
    );

    expect(
      screen.getByText("The passages this question matched could not be read or cited."),
    ).toBeInTheDocument();
    expect(outcomeOf("Synthesis · abstained")).toBe("evidence_unusable");

    fireEvent.click(screen.getByRole("button", { name: /Reindex this paper/ }));

    await waitFor(() => expect(reindexFile).toHaveBeenCalledWith("doc.pdf"));
  });

  it("offers no reindex when the paper simply has nothing to say", async () => {
    await ask((h) =>
      h.onAbstained?.({ message: "Nothing to answer from.", reason: "no_evidence" }),
    );

    expect(screen.queryByRole("button", { name: /Reindex/ })).not.toBeInTheDocument();
  });

  it("cites nothing for an abstention", async () => {
    await ask((h) =>
      h.onAbstained?.({ message: "Nothing to answer from.", reason: "no_evidence" }),
    );

    expect(screen.queryByText(/Grounded in/)).not.toBeInTheDocument();
  });

  it("replays a stored abstention from history as an abstention", async () => {
    vi.mocked(getMessages).mockResolvedValue({
      messages: [
        {
          id: "t1-user",
          text: "what?",
          sender: "user",
          sources: [],
          created_at: "2026-09-26T10:00:00+00:00",
          turn_id: "t1",
          turn_sequence: 1,
          turn_status: "abstained",
          turn_abstention_reason: "no_evidence",
        },
        {
          id: "t1-bot",
          text: "This paper has no passage that speaks to this question.",
          sender: "bot",
          sources: [],
          created_at: "2026-09-26T10:00:01+00:00",
          turn_id: "t1",
          turn_sequence: 1,
          turn_status: "abstained",
          turn_abstention_reason: "no_evidence",
        },
      ],
    });
    renderPane();

    expect(await screen.findByText("Synthesis · abstained")).toBeInTheDocument();
    expect(outcomeOf("Synthesis · abstained")).toBe("no_evidence");
  });

  it("stops the loader when the stream ends with a protocol error", async () => {
    vi.mocked(chatStream).mockRejectedValue(
      new StreamProtocolError("The answer stream ended before it finished."),
    );
    renderPane();
    const input = await screen.findByPlaceholderText(/Ask this paper something/);
    await act(async () => {
      fireEvent.change(input, { target: { value: "what?" } });
      fireEvent.click(screen.getByRole("button", { name: "Send" }));
    });

    expect(
      screen.getByText("The answer stream ended before it finished."),
    ).toBeInTheDocument();
    expect(screen.getByText("The answer was interrupted")).toBeInTheDocument();
    expect(outcomeOf("Synthesis · failed")).toBe("interrupted");
    expect(screen.getByRole("button", { name: "Try again" })).toBeInTheDocument();
    expect(screen.queryByText(/Reading passages/)).not.toBeInTheDocument();
  });

  it("names a retrieval outage as unavailable search with its own retry", async () => {
    vi.mocked(chatStream).mockRejectedValue(new Error("Vector store is unavailable"));
    renderPane();
    const input = await screen.findByPlaceholderText(/Ask this paper something/);
    await act(async () => {
      fireEvent.change(input, { target: { value: "what?" } });
      fireEvent.click(screen.getByRole("button", { name: "Send" }));
    });

    expect(screen.getByText("Search is unavailable")).toBeInTheDocument();
    expect(outcomeOf("Synthesis · failed")).toBe("retrieval_unavailable");
    expect(screen.getByRole("button", { name: "Retry search" })).toBeInTheDocument();
  });

  it("retries a failed question as a new turn", async () => {
    vi.mocked(chatStream).mockRejectedValueOnce(new Error("Vector store is unavailable"));
    vi.mocked(chatStream).mockImplementationOnce(async (_query, _file, handlers) => {
      handlers.onDone?.({ sources: [], claims: [], grounded: false, promptVersion: "grounded-claims-v1", truncated: false, finishReason: "stop" });
    });
    renderPane();
    const input = await screen.findByPlaceholderText(/Ask this paper something/);
    await act(async () => {
      fireEvent.change(input, { target: { value: "what?" } });
      fireEvent.click(screen.getByRole("button", { name: "Send" }));
    });

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Retry search" }));
    });

    await waitFor(() => expect(vi.mocked(chatStream)).toHaveBeenCalledTimes(2));
    expect(vi.mocked(chatStream).mock.calls[1][0]).toBe("what?");
  });

  it("pauses instead of indexing when the status check fails with no confirmed state", async () => {
    vi.mocked(checkIsProcessed).mockRejectedValue(new Error("status down"));
    renderPane();

    expect(await screen.findByTestId("chat-status-error")).toBeInTheDocument();
    expect(screen.getByText(/Could not check this paper/)).toBeInTheDocument();
    expect(screen.queryByText(/Indexing Document/)).not.toBeInTheDocument();
  });

  it("keeps the last confirmed state when a status refresh fails", async () => {
    vi.mocked(checkIsProcessed).mockResolvedValue({
      is_processed: true,
      ingestion: null,
      index: { state: "ready", manifest, runtime_manifest: manifest, changes: [], change_details: [] },
    });
    const queryClient = renderPane();
    expect(await screen.findByPlaceholderText(/Ask this paper something/)).toBeInTheDocument();
    queryClient.setQueryData(["files", "doc.pdf", "is-processed"], undefined);
    vi.mocked(checkIsProcessed).mockRejectedValueOnce(new Error("refresh down"));
    await act(async () => {
      await queryClient.invalidateQueries({ queryKey: ["files", "doc.pdf", "is-processed"] });
    });

    expect(await screen.findByTestId("chat-status-stale")).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/Ask this paper something/)).toBeEnabled();
  });

  it("offers a Conversation retry instead of an empty session when the Conversation fails", async () => {
    vi.mocked(getMessages).mockRejectedValue(new Error("conversation down"));
    renderPane();

    expect(await screen.findByTestId("chat-conversation-error")).toBeInTheDocument();
    expect(screen.queryByText("Session Initialized")).not.toBeInTheDocument();
  });

  it("names missing App Settings as its own outcome, not an interrupted stream", async () => {
    vi.mocked(chatStream).mockRejectedValue(new Error("No chat provider configured."));
    renderPane();
    const input = await screen.findByPlaceholderText(/Ask this paper something/);
    await act(async () => {
      fireEvent.change(input, { target: { value: "what?" } });
      fireEvent.click(screen.getByRole("button", { name: "Send" }));
    });

    expect(screen.getByText("No chat provider is configured")).toBeInTheDocument();
    expect(outcomeOf("Synthesis · failed")).toBe("settings");
    expect(screen.getByRole("button", { name: "Open Settings" })).toBeInTheDocument();
    expect(screen.queryByText("The answer was interrupted")).not.toBeInTheDocument();
  });

  it("withholds citations on a failed answer", async () => {
    await ask((h) => h.onProviderError?.("The model is unavailable.", "provider"));

    expect(screen.getByText("The model could not answer")).toBeInTheDocument();
    expect(screen.queryByText(/passages/)).not.toBeInTheDocument();
    expect(screen.queryByTestId("source-jump-S1")).not.toBeInTheDocument();
  });

  it("does not carry a late token into the next document", async () => {
    // A stream that resolves late: its token arrives after the pane moved on.
    let late: ((text: string) => void) | undefined;
    vi.mocked(chatStream).mockImplementation(async (_query, _file, handlers) => {
      late = handlers.onToken;
    });
    renderPane();
    const input = await screen.findByPlaceholderText(/Ask this paper something/);
    await act(async () => {
      fireEvent.change(input, { target: { value: "what?" } });
      fireEvent.click(screen.getByRole("button", { name: "Send" }));
    });

    usePdfStore.getState().setFile({ ...file, id: "file-2", name: "other.pdf" });
    await waitFor(() => expect(screen.getByText("Session Initialized")).toBeInTheDocument());

    act(() => late?.("A token from the old document."));

    expect(screen.queryByText("A token from the old document.")).not.toBeInTheDocument();
  });
});

const passage = (overrides: Partial<ISource> = {}): ISource => ({
  content: "Gradient accumulation lets a small batch behave like a large one.",
  document: "doc.pdf",
  chunk_index: 0,
  score: 0.9,
  page: 4,
  source_id: "S1",
  rank: 1,
  ...overrides,
});

function completes(overrides: Partial<IChatDone> = {}): IChatDone {
  return {
    sources: [],
    claims: [],
    grounded: false,
    promptVersion: "grounded-claims-v1",
    truncated: false,
    finishReason: "stop",
    ...overrides,
  };
}

describe("ChatPane claim citations", () => {
  beforeEach(() => {
    vi.mocked(checkIsProcessed).mockResolvedValue({
      is_processed: true,
      ingestion: null,
      index: { state: "ready", manifest, runtime_manifest: manifest, changes: [], change_details: [] },
    });
  });

  it("lists each claim with the passages that support it", async () => {
    await ask((h) => {
      h.onToken?.("It accumulates gradients over a small batch.");
      h.onDone?.(
        completes({
          sources: [passage()],
          claims: [{ claim: "It is 42.", sources: ["S1"] }],
          grounded: true,
        }),
      );
    });

    expect(await screen.findByText("It is 42.")).toBeInTheDocument();
    expect(outcomeOf(/Synthesis/)).toBeUndefined();
  });

  it("opens the supporting page when a claim's citation is clicked", async () => {
    await ask((h) =>
      h.onDone?.(
        completes({
          sources: [passage({ page: 12 })],
          claims: [{ claim: "It is 42.", sources: ["S1"] }],
          grounded: true,
        }),
      ),
    );

    const citation = await screen.findByRole("button", { name: /Open page 12 for citation S1/ });
    fireEvent.click(citation);

    expect(usePdfStore.getState().citationTarget?.page).toBe(12);
  });

  it("offers no page to open when the supporting passage has none", async () => {
    await ask((h) =>
      h.onDone?.(
        completes({
          sources: [passage({ page: null })],
          claims: [{ claim: "It is 42.", sources: ["S1"] }],
          grounded: true,
        }),
      ),
    );

    const citation = await screen.findByRole("button", { name: /Citation S1, no page to open/ });

    expect(citation).toBeDisabled();
  });

  it("does not call an answer grounded when no claim names a passage", async () => {
    await ask((h) => {
      h.onToken?.("Something the paper never said.");
      h.onDone?.(
        completes({
          sources: [passage()],
          claims: [{ claim: "Something the paper never said.", sources: [] }],
          grounded: false,
        }),
      );
    });

    expect(await screen.findByText("Synthesis · ungrounded")).toBeInTheDocument();
  });

  it("shows the retrieval rank and method, not a confidence percentage", async () => {
    await ask((h) =>
      h.onDone?.(
        completes({
          sources: [passage({ score: 0.87 })],
          claims: [{ claim: "It is 42.", sources: ["S1"] }],
          grounded: true,
          retrieval: { method: "hybrid", outcome: "success" },
        }),
      ),
    );

    expect(await screen.findByText(/hybrid retrieval/)).toBeInTheDocument();
    expect(screen.getByText(/Rank 1/)).toBeInTheDocument();
    expect(screen.queryByText(/87%/)).not.toBeInTheDocument();
    expect(screen.queryByText(/match/)).not.toBeInTheDocument();
  });

  it("calls an answer nothing while it is still arriving", async () => {
    await ask((h) => h.onToken?.("Streaming in."));

    expect(outcomeOf("Synthesis · answering")).toBe("answering");
  });

  it("calls a completed answer grounded only once a claim has cited something", async () => {
    await ask((h) =>
      h.onDone?.(
        completes({
          sources: [passage()],
          claims: [{ claim: "Streaming in.", sources: ["S1"] }],
          grounded: true,
        }),
      ),
    );

    expect(outcomeOf("Synthesis · grounded")).toBeUndefined();
  });

  it("marks a claim the paper did not back as unsupported", async () => {
    await ask((h) =>
      h.onDone?.(
        completes({
          sources: [passage()],
          claims: [{ claim: "The paper never said this.", sources: [] }],
          grounded: false,
        }),
      ),
    );

    expect(await screen.findByText("The paper never said this.")).toBeInTheDocument();
    expect(screen.getByText("not supported")).toBeInTheDocument();
    expect(outcomeOf("Synthesis · ungrounded")).toBe("ungrounded");
  });

  it("shows citations that could not be resolved as their own outcome", async () => {
    await ask((h) => h.onProviderError?.("The answer cited a passage that was not supplied.", "citations"));

    expect(outcomeOf("Synthesis · failed")).toBe("citations");
  });

  it("replays a stored answer with its claims from history", async () => {
    vi.mocked(getMessages).mockResolvedValue({
      messages: [
        {
          id: "t1-user",
          text: "what?",
          sender: "user",
          sources: [],
          created_at: "2026-09-26T10:00:00+00:00",
          turn_id: "t1",
          turn_sequence: 1,
          turn_status: "answered",
        },
        {
          id: "t1-bot",
          text: "It accumulates gradients.",
          sender: "bot",
          sources: [passage({ page: 9 })],
          claims: [{ claim: "It accumulates gradients.", sources: ["S1"] }],
          created_at: "2026-09-26T10:00:01+00:00",
          turn_id: "t1",
          turn_sequence: 1,
          turn_status: "answered",
        },
      ],
    });
    renderPane();

    fireEvent.click(await screen.findByRole("button", { name: /Open page 9 for citation S1/ }));

    expect(usePdfStore.getState().citationTarget?.page).toBe(9);
  });
});
