import { describe, expect, it, vi, afterEach } from "vitest";
import { chatStream, StreamProtocolError } from "./files";

function sseStream(blocks: string[]): Response {
  const encoder = new TextEncoder();
  const body = blocks.map((block) => encoder.encode(`${block}\n\n`));
  let index = 0;
  return {
    ok: true,
    status: 200,
    body: {
      getReader: () => ({
        read: async () =>
          index < body.length
            ? { done: false, value: body[index++] }
            : { done: true, value: undefined },
      }),
    },
  } as unknown as Response;
}

function serve(blocks: string[]) {
  const fetchMock = vi.fn(async () => sseStream(blocks));
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function collect() {
  const seen: string[] = [];
  const handlers = {
    onStart: vi.fn(() => seen.push("start")),
    onToken: vi.fn((text: string) => seen.push(`token:${text}`)),
    onProviderError: vi.fn((message: string, category: string) =>
      seen.push(`provider_error:${category}:${message}`),
    ),
    onPersistenceError: vi.fn((message: string) =>
      seen.push(`persistence_error:${message}`),
    ),
    onCancelled: vi.fn((reason: string) => seen.push(`cancelled:${reason}`)),
    onDone: vi.fn(() => seen.push("done")),
  };
  return { seen, handlers };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("chatStream", () => {
  it("reports the turn the answer is for", async () => {
    serve([
      'event: start\ndata: {"turn_id":"abc"}',
      'event: done\ndata: {"done":true,"sources":[],"truncated":false}',
    ]);
    const { handlers } = collect();

    await chatStream("q", "doc.pdf", handlers);

    expect(handlers.onStart).toHaveBeenCalledWith({ turnId: "abc" });
  });

  it("delivers every token in order", async () => {
    serve([
      'event: start\ndata: {"turn_id":"abc"}',
      'event: token\ndata: {"text":"Hello "}',
      'event: token\ndata: {"text":"there."}',
      'event: done\ndata: {"done":true,"sources":[],"truncated":false,"finish_reason":"stop"}',
    ]);
    const { seen, handlers } = collect();

    await chatStream("q", "doc.pdf", handlers);

    expect(seen).toEqual(["start", "token:Hello ", "token:there.", "done"]);
  });

  it("carries the sources and truncation flag on a stored answer", async () => {
    const sources = [
      { content: "chunk", document: "doc.pdf", chunk_index: 0, score: 0.9, page: 3 },
    ];
    serve([
      `event: done\ndata: ${JSON.stringify({
        done: true,
        sources,
        truncated: true,
        finish_reason: "length",
      })}`,
    ]);
    const onDone = vi.fn();

    await chatStream("q", "doc.pdf", { onToken: vi.fn(), onDone });

    expect(onDone).toHaveBeenCalledWith({
      sources,
      retrieval: undefined,
      truncated: true,
      finishReason: "length",
    });
  });

  it.each([
    ["provider", "The model is unavailable."],
    ["timeout", "This answer took too long."],
    ["empty_output", "The model returned nothing."],
  ])("reports a %s failure as a provider error", async (category, message) => {
    serve([`event: provider_error\ndata: ${JSON.stringify({ error: message, category })}`]);
    const { seen, handlers } = collect();

    await chatStream("q", "doc.pdf", handlers);

    expect(seen).toEqual([`provider_error:${category}:${message}`]);
  });

  it("reports a save failure as its own outcome", async () => {
    serve(['event: persistence_error\ndata: {"error":"Not saved."}']);
    const { seen, handlers } = collect();

    await chatStream("q", "doc.pdf", handlers);

    expect(seen).toEqual(["persistence_error:Not saved."]);
  });

  it("reports a cancellation and why", async () => {
    serve(['event: cancelled\ndata: {"reason":"document deleted"}']);
    const { seen, handlers } = collect();

    await chatStream("q", "doc.pdf", handlers);

    expect(seen).toEqual(["cancelled:document deleted"]);
  });

  it("rejects a stream that ends without a terminal event", async () => {
    serve([
      'event: start\ndata: {"turn_id":"abc"}',
      'event: token\ndata: {"text":"Half an ans',
    ]);
    const { handlers } = collect();

    await expect(chatStream("q", "doc.pdf", handlers)).rejects.toThrow(
      StreamProtocolError,
    );
  });

  it("rejects a stream that produced no events at all", async () => {
    serve([]);
    const { handlers } = collect();

    await expect(chatStream("q", "doc.pdf", handlers)).rejects.toThrow(
      StreamProtocolError,
    );
  });

  it("rejects an event the protocol does not define", async () => {
    serve(['event: mystery\ndata: {"text":"?"}']);
    const { handlers } = collect();

    await expect(chatStream("q", "doc.pdf", handlers)).rejects.toThrow(
      StreamProtocolError,
    );
  });

  it("rejects a token event whose text is not a string", async () => {
    serve(['event: token\ndata: {"text":42}']);
    const { handlers } = collect();

    await expect(chatStream("q", "doc.pdf", handlers)).rejects.toThrow(
      StreamProtocolError,
    );
  });

  it("rejects a start event with no turn", async () => {
    serve(["event: start\ndata: {}"]);
    const { handlers } = collect();

    await expect(chatStream("q", "doc.pdf", handlers)).rejects.toThrow(
      StreamProtocolError,
    );
  });

  it("rejects a provider error with an unknown category", async () => {
    serve(['event: provider_error\ndata: {"error":"x","category":"meltdown"}']);
    const { handlers } = collect();

    await expect(chatStream("q", "doc.pdf", handlers)).rejects.toThrow(
      StreamProtocolError,
    );
  });

  it("rejects a block whose data is not readable JSON", async () => {
    serve(['event: token\ndata: not-json']);
    const { handlers } = collect();

    await expect(chatStream("q", "doc.pdf", handlers)).rejects.toThrow(
      StreamProtocolError,
    );
  });

  it("rejects a data payload that is not an object", async () => {
    serve(['event: token\ndata: ["text"]']);
    const { handlers } = collect();

    await expect(chatStream("q", "doc.pdf", handlers)).rejects.toThrow(
      StreamProtocolError,
    );
  });

  it("ignores a heartbeat comment", async () => {
    serve([
      ": keep-alive",
      'event: token\ndata: {"text":"ok"}',
      'event: done\ndata: {"done":true,"sources":[],"truncated":false}',
    ]);
    const { seen, handlers } = collect();

    await chatStream("q", "doc.pdf", handlers);

    expect(seen).toEqual(["token:ok", "done"]);
  });

  it("stops reading once the stream is terminal", async () => {
    const trailing = 'event: token\ndata: {"text":"after"}';
    serve([
      'event: done\ndata: {"done":true,"sources":[],"truncated":false}',
      trailing,
    ]);
    const onToken = vi.fn();

    await chatStream("q", "doc.pdf", { onToken, onDone: vi.fn() });

    expect(onToken).not.toHaveBeenCalled();
  });

  it("reassembles events split across network chunks", async () => {
    const encoder = new TextEncoder();
    const whole = encoder.encode(
      'event: token\ndata: {"text":"split"}\n\nevent: done\ndata: {"done":true,"sources":[],"truncated":false}\n\n',
    );
    const body = [whole.slice(0, 20), whole.slice(20)];
    let index = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        status: 200,
        body: {
          getReader: () => ({
            read: async () =>
              index < body.length
                ? { done: false, value: body[index++] }
                : { done: true, value: undefined },
          }),
        },
      })),
    );
    const onToken = vi.fn();
    const onDone = vi.fn();

    await chatStream("q", "doc.pdf", { onToken, onDone });

    expect(onToken).toHaveBeenCalledWith("split");
    expect(onDone).toHaveBeenCalled();
  });

  it("surfaces a failed response as the message the server sent", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: false,
        status: 400,
        json: async () => ({ error: "No chat provider configured." }),
      })),
    );
    const { handlers } = collect();

    await expect(chatStream("q", "doc.pdf", handlers)).rejects.toThrow(
      "No chat provider configured.",
    );
  });
});
