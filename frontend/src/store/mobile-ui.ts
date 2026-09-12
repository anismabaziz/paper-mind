import { create } from "zustand";

type MobileUiState = {
  libraryOpen: boolean;
  chatOpen: boolean;
  setLibraryOpen: (open: boolean) => void;
  setChatOpen: (open: boolean) => void;
};

const useMobileUi = create<MobileUiState>((set) => ({
  libraryOpen: false,
  chatOpen: false,
  setLibraryOpen: (open) => set({ libraryOpen: open }),
  setChatOpen: (open) => set({ chatOpen: open }),
}));

export default useMobileUi;
