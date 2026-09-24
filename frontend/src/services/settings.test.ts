import { afterEach, describe, expect, it, vi } from "vitest";
import client from "./client";
import { verifySettings } from "./settings";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("verifySettings", () => {
  it("sends the candidate to the verification endpoint", async () => {
    const post = vi.spyOn(client, "post").mockResolvedValue({
      data: { ok: true, error: null },
    });
    const candidate = {
      provider: "groq",
      model: "openai/gpt-oss-120b",
      api_key: "sk-test",
    };

    await verifySettings(candidate);

    expect(post).toHaveBeenCalledWith("/settings/verify", candidate);
  });

  it("forwards an abort signal to the verification request", async () => {
    const post = vi.spyOn(client, "post").mockResolvedValue({
      data: { ok: true, error: null },
    });
    const controller = new AbortController();
    const candidate = {
      provider: "google",
      model: "gemini-2.5-flash",
      api_key: "sk-test",
    };

    await verifySettings(candidate, controller.signal);

    expect(post).toHaveBeenCalledWith("/settings/verify", candidate, {
      signal: controller.signal,
    });
  });
});
