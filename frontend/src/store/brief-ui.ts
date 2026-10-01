import { create } from "zustand";

interface IBriefUiState {
  isOpen: boolean;
  open: () => void;
  close: () => void;
}

/**
 * Whether the Research Brief dialog is open.
 *
 * A brief reads two Documents at once, so it is not attached to whichever one
 * happens to be open in the reader. It is its own surface, opened deliberately
 * from the library rather than reached by opening a paper.
 */
const useBriefUi = create<IBriefUiState>((set) => ({
  isOpen: false,
  open: () => set({ isOpen: true }),
  close: () => set({ isOpen: false }),
}));

export default useBriefUi;