import { create } from "zustand";
import { File as FileType } from "@/types/db";
type PdfState = {
  file: FileType | null;
  setFile: (file: FileType | null) => void;
  citationTarget: { page: number; key: number } | null;
  setCitationTarget: (page: number | null) => void;
};

const usePdfStore = create<PdfState>((set) => ({
  file: null,
  setFile: (file: FileType | null) => set(() => ({ file })),
  citationTarget: null,
  setCitationTarget: (page) =>
    set(() => ({
      citationTarget: page == null ? null : { page, key: Date.now() },
    })),
}));

export default usePdfStore;
