import { create } from "zustand";
import { File as FileType } from "@/types/db";
type PdfState = {
  file: FileType | null;
  processed: boolean;
  setFile: (file: FileType) => void;
};

const usePdfStore = create<PdfState>((set) => ({
  file: null,
  processed: false,
  setFile: (file: FileType) => set(() => ({ file })),
}));

export default usePdfStore;
