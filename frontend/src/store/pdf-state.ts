import { create } from "zustand";
import { File as FileType } from "@/types/db";
type PdfState = {
  file: FileType | null;
  processed: boolean;
  setFile: (file: FileType | null) => void;
};

const usePdfStore = create<PdfState>((set) => ({
  file: null,
  processed: false,
  setFile: (file: FileType | null) => set(() => ({ file })),
}));

export default usePdfStore;
