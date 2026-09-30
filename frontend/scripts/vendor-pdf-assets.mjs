import { cpSync, mkdirSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..");
const src = join(root, "node_modules", "pdfjs-dist");
const destCmaps = join(root, "public", "cmaps");
const destFonts = join(root, "public", "standard_fonts");

for (const [from, to] of [
  [join(src, "cmaps"), destCmaps],
  [join(src, "standard_fonts"), destFonts],
]) {
  if (!existsSync(from)) {
    throw new Error(`pdfjs-dist assets missing at ${from}. Run npm install first.`);
  }
  mkdirSync(to, { recursive: true });
  cpSync(from, to, { recursive: true });
  console.log(`vendored ${from} -> ${to}`);
}
