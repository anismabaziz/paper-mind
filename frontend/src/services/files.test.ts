import { describe, expect, it } from "vitest";
import { parseSSEBlock } from "./files";

describe("parseSSEBlock", () => {
  it("parses a token event", () => {
    const event = parseSSEBlock('event: token\ndata: {"text":"hello"}');

    expect(event).toEqual({ name: "token", data: { text: "hello" } });
  });

  it("parses an error event", () => {
    const event = parseSSEBlock('event: error\ndata: {"error":"model is down"}');

    expect(event).toEqual({ name: "error", data: { error: "model is down" } });
  });

  it("parses a done event carrying sources", () => {
    const sources = [
      { content: "chunk", document: "doc.pdf", chunk_index: 0, score: 0.9, page: 3 },
    ];
    const event = parseSSEBlock(
      `event: done\ndata: ${JSON.stringify({ done: true, sources })}`
    );

    expect(event).toEqual({ name: "done", data: { done: true, sources } });
  });

  it("defaults to a message event when no event line is present", () => {
    const event = parseSSEBlock('data: {"done":true}');

    expect(event).toEqual({ name: "message", data: { done: true } });
  });

  it("returns null when the block carries no data", () => {
    expect(parseSSEBlock("event: token")).toBeNull();
    expect(parseSSEBlock("")).toBeNull();
  });

  it("returns null for invalid JSON", () => {
    expect(parseSSEBlock("event: token\ndata: not-json")).toBeNull();
  });

  it("joins multi-line data payloads", () => {
    const event = parseSSEBlock('event: token\ndata: {"text":"he\ndata: llo"}');

    expect(event).toEqual({ name: "token", data: { text: "hello" } });
  });
});
