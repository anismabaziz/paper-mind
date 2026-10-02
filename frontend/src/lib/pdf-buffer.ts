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


