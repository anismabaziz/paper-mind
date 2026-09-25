// pdf.js transfers the buffer to its worker on load, detaching it. A view
// over a detached buffer has zero length and can no longer be rendered from;
// callers use this to decide when to refetch fresh bytes. Each Document must
// get its own copy (Uint8Array.slice) because the transfer detaches whatever
// buffer it was given.
export function isDetached(data: Uint8Array): boolean {
  return data.buffer.byteLength === 0;
}
