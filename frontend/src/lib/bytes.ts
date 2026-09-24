// View over the same ArrayBuffer without copying bytes. Use this when two
// consumers (e.g. the reading sheet and the thumbnail strip) share one
// fetched PDF buffer so a large file is held once in JS memory instead of
// two to three times via .slice().
export function sharedView(data: Uint8Array): Uint8Array {
  return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
}

// pdf.js transfers the buffer to its worker on load, detaching it. A view
// over a detached buffer has zero length and can no longer be rendered from;
// callers use this to decide when to refetch fresh bytes.
export function isDetached(data: Uint8Array): boolean {
  return data.buffer.byteLength === 0;
}
