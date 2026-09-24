import { describe, expect, it } from "vitest";
import { isDetached, sharedView } from "./bytes";

describe("sharedView", () => {
  it("shares the same buffer instead of copying", () => {
    const data = new Uint8Array([1, 2, 3]);

    const view = sharedView(data);

    expect(view.buffer).toBe(data.buffer);
    expect(Array.from(view)).toEqual([1, 2, 3]);
  });

  it("preserves byte offsets into a larger buffer", () => {
    const backing = new Uint8Array([0, 9, 8, 0]);
    const data = new Uint8Array(backing.buffer, 1, 2);

    const view = sharedView(data);

    expect(view.byteOffset).toBe(1);
    expect(Array.from(view)).toEqual([9, 8]);
  });

  it("writes through to the original bytes", () => {
    const data = new Uint8Array([1, 2, 3]);

    sharedView(data)[0] = 7;

    expect(data[0]).toBe(7);
  });
});

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
