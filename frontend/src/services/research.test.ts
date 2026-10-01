import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import {
  BriefRequestError,
  briefStream,
  checkBriefScope,
  briefUnavailableReason,
  cancelBrief,
  isBriefReadable,
} from "./research";
import { StreamProtocolError, parseSSEBlock } from "./files";
import type { File as DbFile } from "@/types/db";

vi.mock("./client", () => {
  const get = vi.fn();
  const post = vi.fn();
  const defaultClient = { get, post };
  return { default: defaultClient, apiBaseUrl: "http://api.test" };
});

function file(overrides: Partial<DbFile> = {}): DbFile {
  return {
    id: "doc-1",
    is_processed: true,
    index_generation: 1,
    index: {
      state: "ready",
      manifest: null,
      runtime_manifest: {} as never,
      changes: [],
      change_details: [],
    },
    last_opened_at: null,
    deletion_state: "active",
    deletion_error: null,
    deletion_attempts: 0,
    ingestion: null,
    metadata: { content_type: "application/pdf", size: 1024 },
    name: "a.pdf",
    title: "Paper A",
    original_filename: null,
    url: "/storage/a.pdf",
    ...overrides,
  };
}

function sse(body: string): Response {
  const bytes = new TextEncoder().encode(body);
  return new Response(
    new ReadableStream({
      start(controller) {
        controller.enqueue(bytes);
        controller.close();
      },
    }),
    { status: 200, headers: { "Content-Type": "text/event-stream" } },
  );
}

function event(name: string, payload: unknown): string {
  return `event: ${name}\ndata: ${JSON.stringify(payload)}\n\n`;
}

const SCOPE = [
  { label: "A", title: "Paper A", document_id: "doc-1", index_generation: 1 },
  { label: "B", title: "Paper B", document_id: "doc-2", index_generation: 1 },
];

const EVIDENCE = [
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
    content: "the passage",
    read: true,
    title: "Paper A",
  },
];

const BUDGET = {
  turns: 2,
  tool_calls: 1,
  repeated_calls: 0,
  tokens: 900,
  elapsed_seconds: 1.5,
  documents: 2,
  max_turns: 6,
  max_tool_calls: 8,
  max_repeated_calls: 2,
  max_tokens: 120000,
  max_seconds: 120,
};

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn());
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("briefStream", () => {
  it("reads a completed brief with its scope, evidence, and budget", async () => {
    vi.mocked(fetch).mockResolvedValue(
      sse(
        event("start", {
          brief_id: "brief-1",
          documents: SCOPE,
          prompt_version: "brief-evidence-v1",
          limits: BUDGET,
        }) +
          event("done", {
            done: true,
            status: "complete",
            answer: "They disagree.",
            evidence: EVIDENCE,
            documents: SCOPE,
            budget: BUDGET,
          }),
      ),
    );
    const onStart = vi.fn();
    const onDone = vi.fn();

    await briefStream("why?", ["a.pdf", "b.pdf"], { onStart, onDone });

    expect(onStart).toHaveBeenCalledWith({
      briefId: "brief-1",
      documents: SCOPE,
      promptVersion: "brief-evidence-v1",
      limits: BUDGET,
    });
    expect(onDone).toHaveBeenCalledWith({
      status: "complete",
      answer: "They disagree.",
      stoppedBy: null,
      message: null,
      evidence: EVIDENCE,
      documents: SCOPE,
      budget: BUDGET,
      brief: expect.anything(),
      claims: [],
      gaps: [],
      abstained: false,
      promptVersion: null,
      model: null,
    });
  });

  it("keeps a brief stopped by a limit marked incomplete", async () => {
    vi.mocked(fetch).mockResolvedValue(
      sse(
        event("done", {
          done: true,
          status: "incomplete",
          complete: false,
          stopped_by: "turns",
          message: "This brief was stopped before it finished (turns).",
          evidence: EVIDENCE,
          documents: SCOPE,
          budget: BUDGET,
        }),
      ),
    );
    const onDone = vi.fn();

    await briefStream("why?", ["a.pdf", "b.pdf"], { onDone });

    expect(onDone.mock.calls[0][0]).toMatchObject({
      status: "incomplete",
      stoppedBy: "turns",
    });
  });

  it("reads an abstention as its own outcome rather than as an empty brief", async () => {
    vi.mocked(fetch).mockResolvedValue(
      sse(
        event("abstained", {
          abstained: true,
          message: "Neither selected Document contains evidence.",
          reason: "no_evidence",
          documents: SCOPE,
        }),
      ),
    );
    const onDone = vi.fn();
    const onAbstained = vi.fn();

    await briefStream("why?", ["a.pdf", "b.pdf"], { onDone, onAbstained });

    expect(onDone).not.toHaveBeenCalled();
    expect(onAbstained).toHaveBeenCalledWith(
      expect.objectContaining({ reason: "no_evidence", stoppedBy: null }),
    );
  });

  it("keeps the evidence a provider failure left behind", async () => {
    vi.mocked(fetch).mockResolvedValue(
      sse(
        event("provider_error", {
          error: "The language model could not continue this brief.",
          category: "timeout",
          status: "incomplete",
          evidence: EVIDENCE,
          documents: SCOPE,
          budget: BUDGET,
        }),
      ),
    );
    const onProviderError = vi.fn();

    await briefStream("why?", ["a.pdf", "b.pdf"], { onDone: vi.fn(), onProviderError });

    expect(onProviderError.mock.calls[0][0]).toMatchObject({
      category: "timeout",
      evidence: EVIDENCE,
    });
  });

  it("reports a refusal as the server worded it", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response(JSON.stringify({ error: "A research brief compares exactly 2 Documents." }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }),
    );

    await expect(briefStream("why?", ["a.pdf"], { onDone: vi.fn() })).rejects.toThrow(
      "A research brief compares exactly 2 Documents.",
    );
  });

  it("names the capabilities a model is missing when one blocks the brief", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response(
        JSON.stringify({
          error: "A research brief needs a model that can call tools.",
          category: "research_brief_unsupported_model",
          missing_capabilities: ["tool_use"],
        }),
        { status: 409, headers: { "Content-Type": "application/json" } },
      ),
    );

    const failure = await briefStream(
      "why?",
      ["a.pdf", "b.pdf"],
      { onDone: vi.fn() },
    ).catch((error: unknown) => error);

    expect(failure).toBeInstanceOf(BriefRequestError);
    expect((failure as BriefRequestError).missingCapabilities).toEqual(["tool_use"]);
  });

  it("sends the question and the pair it was given", async () => {
    vi.mocked(fetch).mockResolvedValue(sse(event("done", { status: "complete", answer: "x" })));

    await briefStream("why do they disagree?", ["a.pdf", "b.pdf"], { onDone: vi.fn() });

    const [, init] = vi.mocked(fetch).mock.calls[0];
    expect(JSON.parse(String(init?.body))).toEqual({
      question: "why do they disagree?",
      documents: ["a.pdf", "b.pdf"],
    });
  });

  it("reads a structured brief with claims, gaps, and model metadata", async () => {
    vi.mocked(fetch).mockResolvedValue(
      sse(
        event("done", {
          done: true,
          status: "complete",
          answer: "They disagree.",
          brief: {
            summary: "They disagree.",
            claims: [
              { order: 1, claim: "A keeps data.", supports: ["E1"], conflicts: [], status: "supported" },
              { order: 2, claim: "B deletes data.", supports: [], conflicts: ["E1"], status: "contested" },
            ],
            gaps: ["Whether the policy changed."],
            abstained: false,
          },
          claims: [
            { order: 1, claim: "A keeps data.", supports: ["E1"], conflicts: [], status: "supported" },
          ],
          gaps: ["Whether the policy changed."],
          evidence: EVIDENCE,
          documents: SCOPE,
          budget: BUDGET,
          prompt_version: "brief-structured-v1",
          model: { provider: "groq", model: "openai/gpt-oss-120b" },
        }),
      ),
    );
    const onDone = vi.fn();

    await briefStream("why?", ["a.pdf", "b.pdf"], { onDone });

    expect(onDone.mock.calls[0][0]).toMatchObject({
      status: "complete",
      claims: expect.arrayContaining([expect.objectContaining({ order: 1 })]),
      gaps: ["Whether the policy changed."],
      promptVersion: "brief-structured-v1",
      model: { provider: "groq", model: "openai/gpt-oss-120b" },
    });
    expect(onDone.mock.calls[0][0].brief?.claims).toHaveLength(2);
  });

  it("reads an old brief without structure as a summary with no claims", async () => {
    vi.mocked(fetch).mockResolvedValue(
      sse(
        event("done", {
          done: true,
          status: "complete",
          answer: "They agree.",
          evidence: EVIDENCE,
          documents: SCOPE,
          budget: BUDGET,
        }),
      ),
    );
    const onDone = vi.fn();

    await briefStream("why?", ["a.pdf", "b.pdf"], { onDone });

    expect(onDone.mock.calls[0][0]).toMatchObject({
      answer: "They agree.",
      claims: [],
      gaps: [],
      brief: expect.objectContaining({ summary: "They agree." }),
    });
  });

  it("throws rather than hanging when the stream ends without a result", async () => {
    vi.mocked(fetch).mockResolvedValue(sse(event("start", { brief_id: "b", documents: [] })));

    await expect(
      briefStream("why?", ["a.pdf", "b.pdf"], { onDone: vi.fn() }),
    ).rejects.toBeInstanceOf(StreamProtocolError);
  });

  it("rejects an event this protocol does not define", async () => {
    vi.mocked(fetch).mockResolvedValue(sse(event("citations", { claims: [] })));

    await expect(
      briefStream("why?", ["a.pdf", "b.pdf"], { onDone: vi.fn() }),
    ).rejects.toBeInstanceOf(StreamProtocolError);
  });
});

describe("the shared event framing", () => {
  it("reads a keep-alive block as no event at all", () => {
    expect(parseSSEBlock(": keep-alive\n")).toBeNull();
  });

  it("rejects a block whose data is not a JSON object", () => {
    expect(parseSSEBlock("event: done\ndata: [1,2]\n")).toBeNull();
  });
});

describe("briefUnavailableReason", () => {
  const capable = { tool_use: true, structured_output: true };

  it("allows a brief over two ready Documents", () => {
    expect(briefUnavailableReason(capable, [file(), file({ id: "doc-2", name: "b.pdf" })], [])).toBeNull();
  });

  it("asks for a model before anything else", () => {
    expect(briefUnavailableReason(null, [], [])).toMatch(/App Settings/);
  });

  it("names the capability a model is missing", () => {
    expect(
      briefUnavailableReason({ tool_use: false, structured_output: true }, [], []),
    ).toMatch(/tool use/);
  });

  it("asks for exactly two Documents", () => {
    expect(briefUnavailableReason(capable, [], [])).toMatch(/Select two/);
    expect(briefUnavailableReason(capable, [file()], [])).toMatch(/one more/);
  });

  it("names the Document that is being reindexed", () => {
    const busy = file({
      title: "Paper B",
      ingestion: { state: "running" } as never,
    });
    expect(briefUnavailableReason(capable, [file(), busy], [])).toMatch(
      "Paper B is still indexing",
    );
  });

  it("names the Document whose index is stale", () => {
    const stale = file({ title: "Paper B", index: { ...file().index!, state: "stale" } });
    expect(briefUnavailableReason(capable, [file(), stale], [])).toMatch(/stale index/);
  });

  it("names the Document that was never indexed", () => {
    const fresh = file({ title: "Paper B", is_processed: false, index: null });
    expect(briefUnavailableReason(capable, [file(), fresh], [])).toMatch("not indexed yet");
  });

  it("names the Document that is being removed", () => {
    const gone = file({ title: "Paper B", deletion_state: "deleting" });
    expect(briefUnavailableReason(capable, [file(), gone], [])).toMatch("being removed");
  });
});

describe("isBriefReadable", () => {
  it("is true only for a Document with a ready index", () => {
    expect(isBriefReadable(file())).toBe(true);
    expect(isBriefReadable(file({ is_processed: false }))).toBe(false);
    expect(isBriefReadable(file({ index: null }))).toBe(false);
  });
});

describe("cancelBrief", () => {
  it("asks the server to stop the run rather than dropping the connection", async () => {
    const { default: http } = await import("./client");
    vi.mocked(http.post).mockResolvedValue({
      data: { brief_id: "brief-1", cancelled: true },
    });

    const stopped = await cancelBrief("brief-1");

    expect(stopped).toEqual({ brief_id: "brief-1", cancelled: true });
    const [path, payload] = vi.mocked(http.post).mock.calls[0];
    expect(path).toBe("/research/cancel");
    expect(payload).toEqual({ brief_id: "brief-1" });
  });
});

describe("checkBriefScope", () => {
  it("asks the server which pair it will accept", async () => {
    const { default: http } = await import("./client");
    vi.mocked(http.get).mockResolvedValue({ data: { documents: SCOPE, document_count: 2 } });

    const scope = await checkBriefScope(["a.pdf", "b.pdf"]);

    expect(scope).toEqual({ documents: SCOPE, document_count: 2 });
    const [path, options] = vi.mocked(http.get).mock.calls[0];
    expect(path).toBe("/research/scope");
    expect(options?.params).toEqual(["documents", "a.pdf", "documents", "b.pdf"]);
  });
});

describe("saved briefs", () => {
  it("persists a finished brief and loads it back", async () => {
    const { saveBrief, loadBrief, clearSavedBrief } = await import("./research");
    clearSavedBrief();
    expect(loadBrief()).toBeNull();

    const saved = {
      question: "where do they disagree?",
      documents: ["a.pdf", "b.pdf"],
      run: {
        status: "complete" as const,
        answer: "They disagree.",
        stoppedBy: null,
        message: null,
        evidence: EVIDENCE,
        budget: BUDGET,
        scope: SCOPE,
        brief: {
          summary: "They disagree.",
          claims: [
            { order: 1, claim: "A keeps data.", supports: ["E1"], conflicts: [], status: "supported" as const },
          ],
          gaps: [],
          abstained: false,
        },
        claims: [
          { order: 1, claim: "A keeps data.", supports: ["E1"], conflicts: [], status: "supported" as const },
        ],
        gaps: [],
        abstained: false,
        promptVersion: "brief-structured-v1",
        model: { provider: "groq", model: "openai/gpt-oss-120b" },
      },
      savedAt: new Date().toISOString(),
    };
    saveBrief(saved);

    expect(loadBrief()).toMatchObject({
      question: "where do they disagree?",
      run: expect.objectContaining({
        status: "complete",
        promptVersion: "brief-structured-v1",
      }),
    });
    clearSavedBrief();
    expect(loadBrief()).toBeNull();
  });
});