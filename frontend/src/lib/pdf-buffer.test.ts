import { describe, expect, it } from "vitest";
import { clonePdfData, retainedBytes } from "./pdf-buffer";
import { isDetached } from "./bytes";

describe("pdf-buffer ownership", () => {
  it("hands each Document its own copy so the worker cannot detach a shared view", () => {
    const source = new Uint8Array([1, 2, 3, 4]);
    const first = clonePdfData(source);
    const second = clonePdfData(source);
    expect(first).not.toBeNull();
    expect(second).not.toBeNull();
    expect(first!.data).not.toBe(source);
    expect(first!.data).toEqual(source);
    // Mutating a clone never touches the source other Documents read from.
    first!.data[0] = 9;
    expect(source[0]).toBe(1);
    expect(isDetached(source)).toBe(false);
  });

  it("returns null when there is nothing to render", () => {
    expect(clonePdfData(null)).toBeNull();
    expect(clonePdfData(undefined)).toBeNull();
  });

  it("refuses to clone a buffer pdf.js already detached", () => {
    const dead = new Uint8Array([1, 2, 3]);
    structuredClone(dead.buffer, { transfer: [dead.buffer] });
    expect(isDetached(dead)).toBe(true);
    expect(() => clonePdfData(dead)).toThrow(/detached/);
  });

  it("counts retained bytes so switching Documents can prove the old buffer is gone", () => {
    const a = new Uint8Array(10);
    const b = new Uint8Array(20);
    expect(retainedBytes([a, b])).toBe(30);
    expect(retainedBytes([a, null, undefined])).toBe(10);
    expect(retainedBytes([])).toBe(0);
  });
});
