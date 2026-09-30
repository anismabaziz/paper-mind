// Minimal multi-page PDF builder for browser-limit tests. No dependencies:
// one Helvetica page per sheet with a "Page N" label, enough for pdf.js to
// report the right page count and render thumbnails.
export function buildPdf(pageCount: number): Uint8Array {
  if (!Number.isInteger(pageCount) || pageCount < 1) {
    throw new Error(`pageCount must be a positive integer, got ${pageCount}`);
  }
  const enc = new TextEncoder();
  const chunks: Uint8Array[] = [];
  const offsets: number[] = [];
  let position = 0;
  const push = (text: string) => {
    const bytes = enc.encode(text);
    chunks.push(bytes);
    position += bytes.length;
  };
  const pushBytes = (bytes: Uint8Array) => {
    chunks.push(bytes);
    position += bytes.length;
  };

  const fontObj = 3;
  const pageObjStart = 4;
  const totalObjects = 3 + pageCount * 2;

  push("%PDF-1.4\n");
  // Catalog
  offsets[1] = position;
  push("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
  // Pages
  offsets[2] = position;
  const kids = Array.from({ length: pageCount }, (_, i) => `${pageObjStart + i * 2} 0 R`).join(" ");
  push(`2 0 obj\n<< /Type /Pages /Kids [${kids}] /Count ${pageCount} >>\nendobj\n`);
  // Font
  offsets[fontObj] = position;
  push(`${fontObj} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n`);

  for (let i = 1; i <= pageCount; i++) {
    const pageObj = pageObjStart + (i - 1) * 2;
    const contentObj = pageObj + 1;
    offsets[pageObj] = position;
    push(
      `${pageObj} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] ` +
        `/Resources << /Font << /F1 ${fontObj} 0 R >> >> /Contents ${contentObj} 0 R >>\nendobj\n`,
    );
    const text = `BT /F1 24 Tf 100 700 Td (Page ${i} of ${pageCount}) Tj ET`;
    const stream = enc.encode(text);
    offsets[contentObj] = position;
    push(`${contentObj} 0 obj\n<< /Length ${stream.length} >>\nstream\n`);
    pushBytes(stream);
    push("\nendstream\nendobj\n");
  }

  const xrefStart = position;
  push(`xref\n0 ${totalObjects + 1}\n`);
  push("0000000000 65535 f \n");
  for (let n = 1; n <= totalObjects; n++) {
    push(`${String(offsets[n]).padStart(10, "0")} 00000 n \n`);
  }
  push(`trailer\n<< /Size ${totalObjects + 1} /Root 1 0 R >>\nstartxref\n${xrefStart}\n%%EOF`);

  const total = chunks.reduce((sum, c) => sum + c.length, 0);
  const out = new Uint8Array(total);
  let at = 0;
  for (const c of chunks) {
    out.set(c, at);
    at += c.length;
  }
  return out;
}
