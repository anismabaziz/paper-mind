import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ResearchBriefDialog } from "./ResearchBriefDialog";
import useBriefUi from "@/store/brief-ui";
import usePdfStore from "@/store/pdf-state";
import type { DocumentIndex, File } from "@/types/db";
import type {
  IBriefAbstained,
  IBriefBudget,
  IBriefDone,
  IBriefDocument,
  IBriefEvidence,
  IBriefProviderError,
} from "@/services/research";
import {
  BriefRequestError,
  briefStream,
  cancelBrief,
  checkBriefScope,
} from "@/services/research";
import { getSettings } from "@/services/settings";

vi.mock("@/services/research", async (importOriginal) => {
  const original = await importOriginal<typeof import("@/services/research")>();
  return {
    ...original,
    briefStream: vi.fn(),
    checkBriefScope: vi.fn(),
    cancelBrief: vi.fn(),
  };
});
vi.mock("@/services/settings", () => ({ getSettings: vi.fn() }));
vi.mock("@/hooks/useFiles", () => ({ useFiles: () => ({ data: { files: library }, isPending: false, isError: false, error: null, refetch: vi.fn() }) }));

const READY: DocumentIndex = {
  state: "ready",
  manifest: null,
  runtime_manifest: {} as never,
  changes: [],
  change_details: [],
};

function paper(overrides: Partial<File> = {}): File {
  return {
    id: "doc-1",
    is_processed: true,
    index_generation: 1,
    index: READY,
    last_opened_at: null,
    deletion_state: "active",
    deletion_error: null,
    deletion_attempts: 0,
    ingestion: null,
    metadata: { content_type: "application/pdf", size: 2048 },
    name: "a.pdf",
    title: "Paper A",
    original_filename: null,
    url: "/storage/a.pdf",
    ...overrides,
  };
}

let library: File[] = [];

function renderDialog() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  useBriefUi.getState().open();
  return render(
    <QueryClientProvider client={queryClient}>
      <ResearchBriefDialog />
    </QueryClientProvider>,
  );
}

/**
 * Choose both papers and ask a question, then wait for the model to load.
 *
 * Whether a brief may start depends on the configured model's capabilities, and
 * those arrive with the settings query, so a test that picked its scope before
 * they arrived would be testing a disabled button.
 */
async function chooseBothPapersAndAsk() {
  fireEvent.click(screen.getByTestId("brief-scope-a.pdf"));
  fireEvent.click(screen.getByTestId("brief-scope-b.pdf"));
  fireEvent.change(screen.getByTestId("brief-question"), {
    target: { value: "where do they disagree?" },
  });
  await waitFor(() => {
    expect(vi.mocked(checkBriefScope)).toHaveBeenCalled();
    expect(screen.getByTestId<HTMLButtonElement>("brief-start")).not.toBeDisabled();
  });
}

function start() {
  fireEvent.click(screen.getByTestId("brief-start"));
}

const SETTINGS = {
  provider: "groq",
  model: "openai/gpt-oss-120b",
  masked_key: "••••abcd",
  supported_models: {
    groq: [
      {
        provider: "groq",
        id: "openai/gpt-oss-120b",
        context_window_tokens: 131_072,
        max_output_tokens: 65_536,
        structured_output: true,
        tool_use: true,
        input_cost_per_million_usd: 0.15,
        output_cost_per_million_usd: 0.6,
        pricing_tier: "standard" as const,
        data_location: "cloud" as const,
        timeout_seconds: 15,
      },
    ],
  },
};

const SCOPE: IBriefDocument[] = [
  { label: "A", title: "Paper A", document_id: "doc-1", index_generation: 1 },
  { label: "B", title: "Paper B", document_id: "doc-2", index_generation: 1 },
];

const BUDGET: IBriefBudget = {
  turns: 2,
  tool_calls: 1,
  repeated_calls: 0,
  tokens: 900,
  elapsed_seconds: 1.4,
  documents: 2,
  max_turns: 6,
  max_tool_calls: 8,
  max_repeated_calls: 2,
  max_tokens: 120_000,
  max_seconds: 120,
};

const EVIDENCE: IBriefEvidence[] = [
  {
    evidence_id: "E1",
    label: "A",
    document_id: "doc-1",
    document: "Paper A",
    chunk_index: 0,
    page: 4,
    rank: 1,
    position: 1,
    method: "hybrid",
    content: "Data is kept for thirty months.",
    read: true,
    title: "Paper A",
  },
];

const COMPLETE: IBriefDone = {
  status: "complete",
  answer: "A keeps data longer than B.",
  stoppedBy: null,
  message: null,
  evidence: EVIDENCE,
  documents: SCOPE,
  budget: BUDGET,
};

beforeEach(() => {
  library = [paper(), paper({ id: "doc-2", name: "b.pdf", title: "Paper B" })];
  vi.mocked(getSettings).mockResolvedValue(SETTINGS);
  vi.mocked(checkBriefScope).mockResolvedValue({ documents: SCOPE, document_count: 2 });
  vi.mocked(cancelBrief).mockResolvedValue({ brief_id: "brief-1", cancelled: true });
  useBriefUi.setState({ isOpen: false });
  usePdfStore.setState({ citationTarget: null });
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("choosing the scope", () => {
  it("offers only the Documents a brief can read", () => {
    library = [
      paper(),
      paper({ id: "doc-2", name: "b.pdf", title: "Paper B" }),
      paper({ id: "doc-3", name: "c.pdf", title: "Paper C", ingestion: { state: "running" } as never }),
    ];
    renderDialog();

    const list = screen.getByTestId("brief-scope-list");
    expect(within(list).getByText("Paper A")).toBeInTheDocument();
    expect(within(list).getByText("Paper B")).toBeInTheDocument();
    expect(within(list).queryByText("Paper C")).not.toBeInTheDocument();
  });

  it("labels the selection with the labels the brief itself uses", () => {
    renderDialog();

    fireEvent.click(screen.getByTestId("brief-scope-a.pdf"));
    fireEvent.click(screen.getByTestId("brief-scope-b.pdf"));

    const list = screen.getByTestId("brief-scope-list");
    expect(within(list).getByText("A")).toBeInTheDocument();
    expect(within(list).getByText("B")).toBeInTheDocument();
  });

  it("keeps the scope to exactly two Documents", () => {
    library = [
      paper(),
      paper({ id: "doc-2", name: "b.pdf", title: "Paper B" }),
      paper({ id: "doc-3", name: "c.pdf", title: "Paper C" }),
    ];
    renderDialog();

    fireEvent.click(screen.getByTestId("brief-scope-a.pdf"));
    fireEvent.click(screen.getByTestId("brief-scope-b.pdf"));
    fireEvent.click(screen.getByTestId("brief-scope-c.pdf"));

    expect(screen.getByTestId<HTMLInputElement>("brief-scope-a.pdf").checked).toBe(false);
    expect(screen.getByTestId<HTMLInputElement>("brief-scope-b.pdf").checked).toBe(true);
    expect(screen.getByTestId<HTMLInputElement>("brief-scope-c.pdf").checked).toBe(true);
  });

  it("does not start until the pair is chosen and a question asked", async () => {
    renderDialog();
    await waitFor(() =>
      expect(screen.getByTestId("brief-reason")).toHaveTextContent(/Select two/),
      { timeout: 2000 },
    );
    expect(screen.getByTestId<HTMLButtonElement>("brief-start")).toBeDisabled();

    fireEvent.click(screen.getByTestId("brief-scope-a.pdf"));
    await waitFor(() =>
      expect(screen.getByTestId("brief-reason")).toHaveTextContent(/one more/),
    );
    fireEvent.click(screen.getByTestId("brief-scope-b.pdf"));
    await waitFor(() =>
      expect(screen.getByTestId("brief-scope-labels")).toHaveTextContent("[A]"),
    );
    expect(screen.getByTestId<HTMLButtonElement>("brief-start")).toBeDisabled();

    fireEvent.change(screen.getByTestId("brief-question"), { target: { value: "why?" } });
    await waitFor(() =>
      expect(screen.getByTestId<HTMLButtonElement>("brief-start")).not.toBeDisabled(),
    );
  });

  it("says which Document is still missing", async () => {
    renderDialog();

    await waitFor(() =>
      expect(screen.getByTestId("brief-reason")).toHaveTextContent(/Select two/),
    );
  });
});

describe("the model a brief runs on", () => {
  it("explains that a model without tool use cannot run one", async () => {
    vi.mocked(getSettings).mockResolvedValue({
      ...SETTINGS,
      supported_models: {
        groq: [{ ...SETTINGS.supported_models.groq[0], tool_use: false }],
      },
    });
    renderDialog();

    await waitFor(() => expect(screen.getByTestId("brief-reason")).toHaveTextContent(/tool use/));
  });

  it("explains that a model without structured output cannot run one", async () => {
    vi.mocked(getSettings).mockResolvedValue({
      ...SETTINGS,
      supported_models: {
        groq: [{ ...SETTINGS.supported_models.groq[0], structured_output: false }],
      },
    });
    renderDialog();

    await waitFor(() =>
      expect(screen.getByTestId("brief-reason")).toHaveTextContent(/structured output/),
    );
  });

  it("does not call the server for a model that cannot run a brief", async () => {
    vi.mocked(getSettings).mockResolvedValue({
      ...SETTINGS,
      supported_models: {
        groq: [{ ...SETTINGS.supported_models.groq[0], tool_use: false }],
      },
    });
    renderDialog();

    fireEvent.click(screen.getByTestId("brief-scope-a.pdf"));
    fireEvent.click(screen.getByTestId("brief-scope-b.pdf"));
    fireEvent.change(screen.getByTestId("brief-question"), { target: { value: "why?" } });

    await waitFor(() => expect(screen.getByTestId<HTMLButtonElement>("brief-start")).toBeDisabled());
    expect(briefStream).not.toHaveBeenCalled();
  });
});

describe("the server's view of the scope", () => {
  it("shows the labels the server assigned to the pair", async () => {
    renderDialog();

    await chooseBothPapersAndAsk();

    expect(screen.getByTestId("brief-scope-labels")).toHaveTextContent("[A] Paper A");
    expect(screen.getByTestId("brief-scope-labels")).toHaveTextContent("[B] Paper B");
  });

  it("reports the server's refusal rather than starting anyway", async () => {
    vi.mocked(checkBriefScope).mockRejectedValue(
      new BriefRequestError("Document B's index no longer matches the current settings."),
    );
    renderDialog();

    fireEvent.click(screen.getByTestId("brief-scope-a.pdf"));
    fireEvent.click(screen.getByTestId("brief-scope-b.pdf"));
    fireEvent.change(screen.getByTestId("brief-question"), { target: { value: "why?" } });

    await waitFor(() =>
      expect(screen.getByTestId("brief-reason")).toHaveTextContent(/no longer matches/),
    );
    expect(screen.getByTestId<HTMLButtonElement>("brief-start")).toBeDisabled();
  });
});

describe("running a brief", () => {
  it("sends the chosen pair and question, then shows what came back", async () => {
    vi.mocked(briefStream).mockImplementation(async (_q, _d, handlers) => {
      handlers.onStart?.({
        briefId: "brief-1",
        documents: SCOPE,
        promptVersion: "brief-evidence-v1",
        limits: BUDGET,
      });
      handlers.onDone(COMPLETE);
    });
    renderDialog();

    await chooseBothPapersAndAsk();
    start();

    await waitFor(() => expect(screen.getByTestId("brief-result")).toBeInTheDocument());
    expect(vi.mocked(briefStream).mock.calls[0][0]).toBe("where do they disagree?");
    expect(vi.mocked(briefStream).mock.calls[0][1]).toEqual(["a.pdf", "b.pdf"]);
    expect(screen.getByTestId("brief-result")).toHaveAttribute("data-status", "complete");
    expect(screen.getByText("A keeps data longer than B.")).toBeInTheDocument();
    expect(screen.getByTestId("brief-budget")).toHaveTextContent("2/6 turns · 1/8 reads");
  });

  it("labels a brief a limit stopped as incomplete", async () => {
    vi.mocked(briefStream).mockImplementation(async (_q, _d, handlers) => {
      handlers.onDone({
        status: "incomplete",
        answer: "",
        stoppedBy: "turns",
        message: "This brief was stopped before it finished (turns).",
        evidence: EVIDENCE,
        documents: SCOPE,
        budget: BUDGET,
      });
    });
    renderDialog();

    await chooseBothPapersAndAsk();
    start();

    await waitFor(() => expect(screen.getByTestId("brief-result")).toBeInTheDocument());
    expect(screen.getByTestId("brief-result")).toHaveAttribute("data-status", "incomplete");
    expect(screen.getByTestId("brief-incomplete-note")).toHaveTextContent(/stopped/);
    expect(screen.getByTestId("brief-result")).toHaveTextContent("turns");
  });

  it("shows what the brief read, openable at the Page it came from", async () => {
    vi.mocked(briefStream).mockImplementation(async (_q, _d, handlers) => {
      handlers.onDone(COMPLETE);
    });
    renderDialog();

    await chooseBothPapersAndAsk();
    start();

    await waitFor(() => expect(screen.getByTestId("brief-evidence")).toBeInTheDocument());
    fireEvent.click(screen.getByTestId("brief-evidence-E1"));

    expect(usePdfStore.getState().citationTarget).toMatchObject({ page: 4 });
  });

  it("shows an abstention as an abstention rather than an empty brief", async () => {
    vi.mocked(briefStream).mockImplementation(async (_q, _d, handlers) => {
      const abstained: IBriefAbstained = {
        message: "Neither Document contains evidence.",
        reason: "no_evidence",
        documents: SCOPE,
        stoppedBy: null,
      };
      handlers.onAbstained?.(abstained);
    });
    renderDialog();

    await chooseBothPapersAndAsk();
    start();

    await waitFor(() => expect(screen.getByTestId("brief-abstained")).toBeInTheDocument());
    expect(screen.queryByTestId("brief-result")).not.toBeInTheDocument();
  });

  it("keeps the evidence a provider failure left behind", async () => {
    vi.mocked(briefStream).mockImplementation(async (_q, _d, handlers) => {
      const failure: IBriefProviderError = {
        message: "The language model could not continue this brief.",
        category: "timeout",
        status: "incomplete",
        evidence: EVIDENCE,
        documents: SCOPE,
        budget: BUDGET,
      };
      handlers.onProviderError?.(failure);
    });
    renderDialog();

    await chooseBothPapersAndAsk();
    start();

    await waitFor(() => expect(screen.getByTestId("brief-failed")).toBeInTheDocument());
    expect(within(screen.getByTestId("brief-failed")).getByText(/E1/)).toBeInTheDocument();
  });

  it("reports a refused brief in the server's own words", async () => {
    vi.mocked(briefStream).mockRejectedValue(
      new BriefRequestError("A research brief compares exactly 2 Documents."),
    );
    renderDialog();

    await chooseBothPapersAndAsk();
    start();

    await waitFor(() => expect(screen.getByTestId("brief-error")).toBeInTheDocument());
    expect(screen.getByTestId("brief-error")).toHaveTextContent(/exactly 2 Documents/);
  });

  it("asks the server to stop, so the passages already read survive", async () => {
    // The stream is scripted to stay open until the cancel arrives, which is
    // what pressing stop looks like: the run is still going, and the server
    // decides where to end it.
    vi.mocked(briefStream).mockImplementation((_q, _d, handlers) => {
      handlers.onStart?.({
        briefId: "brief-1",
        documents: SCOPE,
        promptVersion: "brief-evidence-v1",
        limits: BUDGET,
      });
      return new Promise<void>((resolve) => {
        vi.mocked(cancelBrief).mockImplementation(async () => {
          handlers.onDone?.({
            status: "incomplete",
            answer: "",
            stoppedBy: "cancelled",
            message: "This brief was stopped before it finished (cancelled).",
            evidence: EVIDENCE,
            documents: SCOPE,
            budget: BUDGET,
          });
          resolve();
          return { brief_id: "brief-1", cancelled: true };
        });
      });
    });
    renderDialog();

    await chooseBothPapersAndAsk();
    start();
    await waitFor(() => expect(screen.getByTestId("brief-running")).toBeInTheDocument());

    fireEvent.click(screen.getByTestId("brief-stop"));

    await waitFor(() => expect(screen.getByTestId("brief-result")).toBeInTheDocument());
    expect(cancelBrief).toHaveBeenCalledWith("brief-1");
    expect(screen.getByTestId("brief-result")).toHaveAttribute("data-status", "incomplete");
    expect(screen.getByTestId("brief-result")).toHaveTextContent("cancelled");
    // The passages the brief had already read are still on screen, which is
    // the whole reason stopping is not the same as leaving.
    expect(within(screen.getByTestId("brief-evidence")).getByText(/E1/)).toBeInTheDocument();
  });

  it("drops the connection when the server never answers the stop", async () => {
    vi.mocked(briefStream).mockImplementation((_q, _d, handlers, options) => {
      handlers.onStart?.({
        briefId: "brief-1",
        documents: SCOPE,
        promptVersion: "brief-evidence-v1",
        limits: BUDGET,
      });
      return new Promise((_resolve, reject) => {
        options?.signal?.addEventListener(
          "abort",
          () => reject(new DOMException("stopped", "AbortError")),
          { once: true },
        );
      });
    });
    vi.mocked(cancelBrief).mockRejectedValue(new Error("already gone"));
    renderDialog();

    await chooseBothPapersAndAsk();
    start();
    await waitFor(() => expect(screen.getByTestId("brief-running")).toBeInTheDocument());

    fireEvent.click(screen.getByTestId("brief-stop"));

    await waitFor(() => expect(screen.getByTestId("brief-result")).toBeInTheDocument());
    expect(screen.getByTestId("brief-result")).toHaveAttribute("data-status", "incomplete");
    expect(screen.getByTestId("brief-incomplete-note")).toHaveTextContent(/You stopped/);
  });
});