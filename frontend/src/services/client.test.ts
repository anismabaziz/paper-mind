import { afterEach, describe, expect, it, vi } from "vitest";
import { resolveApiBaseUrl } from "./client";

describe("resolveApiBaseUrl", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });
  it("trims whitespace and trailing slashes", () => {
    expect(resolveApiBaseUrl("http://127.0.0.1:3000///  ")).toBe(
      "http://127.0.0.1:3000"
    );
  });

  it("accepts https URLs", () => {
    expect(resolveApiBaseUrl("https://api.example.com")).toBe(
      "https://api.example.com"
    );
  });

  it("throws when the URL is missing", () => {
    expect(() => resolveApiBaseUrl("")).toThrow(/VITE_API_URL is not set/);
    vi.stubEnv("VITE_API_URL", "");
    expect(() => resolveApiBaseUrl(undefined)).toThrow(/VITE_API_URL is not set/);
  });

  it("throws when the URL is malformed", () => {
    expect(() => resolveApiBaseUrl("not-a-url")).toThrow(/VITE_API_URL is invalid/);
  });

  it("throws for unsupported protocols", () => {
    expect(() => resolveApiBaseUrl("ftp://127.0.0.1:3000")).toThrow(
      /Unsupported protocol/
    );
  });
});
