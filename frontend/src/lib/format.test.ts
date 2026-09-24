import { describe, expect, it } from "vitest";
import { formatFileSize } from "./format";

describe("formatFileSize", () => {
  it("formats bytes below one kilobyte", () => {
    expect(formatFileSize(0)).toBe("0 B");
    expect(formatFileSize(512)).toBe("512 B");
    expect(formatFileSize(1023)).toBe("1023 B");
  });

  it("formats kilobytes without decimals", () => {
    expect(formatFileSize(1024)).toBe("1 KB");
    expect(formatFileSize(1536)).toBe("2 KB");
  });

  it("formats megabytes with one decimal", () => {
    expect(formatFileSize(1048576)).toBe("1.0 MB");
    expect(formatFileSize(52_428_800)).toBe("50.0 MB");
  });
});
