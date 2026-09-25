import { describe, expect, it } from "vitest";
import { isDetached } from "./bytes";

describe("isDetached", () => {
  it("reports live buffers as attached", () => {
    expect(isDetached(new Uint8Array([1, 2, 3]))).toBe(false);
  });

  it("reports transferred buffers as detached", () => {
    const data = new Uint8Array([1, 2, 3]);
    structuredClone(data.buffer, { transfer: [data.buffer] });

    expect(isDetached(data)).toBe(true);
  });
});
