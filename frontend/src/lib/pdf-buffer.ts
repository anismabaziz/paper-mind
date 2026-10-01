import { isDetached } from "./bytes";

// pdf.js transfers the buffer it is given to its worker, detaching it. Every
// Renderer therefore owns exactly one clone: the fetched source is never
// handed to a Document directly, and a detached view is never cloned because it
// would render blank.
export function clonePdfData(data: Uint8Array | null | undefined): { data: Uint8Array } | null {
  if (data == null) return null;
  if (isDetached(data)) {
    throw new Error("Cannot clone a detached PDF buffer — refetch fresh bytes first.");
  }
  return { data: data.slice() };
}

// Bytes still held for one Document: the fetched source plus one clone per
// mounted Document. Switching Documents must drop the previous set so only
// the current Document counts.
export function retainedBytes(buffers: Array<Uint8Array | null | undefined>): number {
  let total = 0;
  for (const buffer of buffers) {
    if (buffer != null) total += buffer.byteLength;
  }
  return total;
}
