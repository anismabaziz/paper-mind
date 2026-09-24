import { create } from "zustand";
import { File as FileType } from "@/types/db";
type PdfState = {
  file: FileType | null;
  setFile: (file: FileType | null) => void;
  citationTarget: { page: number; key: number } | null;
  setCitationTarget: (page: number | null) => void;
};

// Monotonic counter so rapid citation jumps never share a key.
// Date.now() could collide within the same millisecond and drop a jump.
let citationSeq = 0;

const usePdfStore = create<PdfState>((set) => ({
  file: null,
  setFile: (file: FileType | null) => set(() => ({ file })),
  citationTarget: null,
  setCitationTarget: (page) =>
    set(() => ({
      citationTarget: page == null ? null : { page, key: ++citationSeq },
    })),
}));

export default usePdfStore;
