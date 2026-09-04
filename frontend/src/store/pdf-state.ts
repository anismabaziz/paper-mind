import { create } from "zustand";
import { File as FileType } from "@/types/db";
type PdfState = {
  file: FileType | null;
  setFile: (file: FileType | null) => void;
};

const usePdfStore = create<PdfState>((set) => ({
  file: null,
  setFile: (file: FileType | null) => set(() => ({ file })),
}));

export default usePdfStore;
