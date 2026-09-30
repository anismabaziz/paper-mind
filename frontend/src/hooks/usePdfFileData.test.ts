import "@testing-library/jest-dom/vitest";
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { usePdfFileData } from "./usePdfFileData";

afterEach(() => cleanup());

function pdfBytes(): Uint8Array {
  return new Uint8Array([37, 80, 68, 70, 45, 49, 46, 52]);
}

describe("usePdfFileData buffer ownership", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn(async () => ({
      ok: true,
      arrayBuffer: async () => pdfBytes().buffer,
    }) as Response));
  });

  it("fetches one source buffer per Document", async () => {
    const { result } = renderHook(() =>
      usePdfFileData({ id: "doc-a", url: "http://localhost/storage/a.pdf" }),
    );
    await waitFor(() => expect(result.current.data).not.toBeNull());
    expect(Array.from(result.current.data!)).toEqual(Array.from(pdfBytes()));
  });

  it("drops the previous Document's bytes when switching", async () => {
    const { result, rerender } = renderHook(
      ({ file }) => usePdfFileData(file),
      { initialProps: { file: { id: "doc-a", url: "http://localhost/storage/a.pdf" } } },
    );
    await waitFor(() => expect(result.current.data).not.toBeNull());
    const first = result.current.data;

    rerender({ file: { id: "doc-b", url: "http://localhost/storage/b.pdf" } });
    // The old buffer is released synchronously before the new fetch lands.
    expect(result.current.data === null || result.current.data !== first).toBe(true);
    await waitFor(() => expect(result.current.data).not.toBeNull());
  });

  it("reload refetches fresh bytes after a detach", async () => {
    const fetchMock = vi.mocked(fetch);
    const { result } = renderHook(() =>
      usePdfFileData({ id: "doc-a", url: "http://localhost/storage/a.pdf" }),
    );
    await waitFor(() => expect(result.current.data).not.toBeNull());
    const calls = fetchMock.mock.calls.length;
    act(() => result.current.reload());
    await waitFor(() => expect(fetchMock.mock.calls.length).toBeGreaterThan(calls));
  });

  it("reports a download failure instead of an endless loader", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({ ok: false, status: 500 }) as Response),
    );
    const { result } = renderHook(() =>
      usePdfFileData({ id: "doc-a", url: "http://localhost/storage/a.pdf" }),
    );
    await waitFor(() => expect(result.current.error).not.toBeNull());
    expect(result.current.data).toBeNull();
    expect(result.current.error).toMatch(/Failed to load PDF \(500\)/);
    expect(result.current.isLoading).toBe(false);
  });

  it("clears the failure so a retry fetches again", async () => {
    const fetchMock = vi.fn(async () => ({ ok: false, status: 500 }) as Response);
    vi.stubGlobal("fetch", fetchMock);
    const { result } = renderHook(() =>
      usePdfFileData({ id: "doc-a", url: "http://localhost/storage/a.pdf" }),
    );
    await waitFor(() => expect(result.current.error).not.toBeNull());
    const calls = fetchMock.mock.calls.length;
    act(() => result.current.reload());
    expect(result.current.error).toBeNull();
    await waitFor(() => expect(fetchMock.mock.calls.length).toBeGreaterThan(calls));
  });
});
