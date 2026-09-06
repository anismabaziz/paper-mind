import { create } from "zustand";

interface ISettingsUiState {
  isOpen: boolean;
  open: () => void;
  close: () => void;
}

const useSettingsUi = create<ISettingsUiState>((set) => ({
  isOpen: false,
  open: () => set({ isOpen: true }),
  close: () => set({ isOpen: false }),
}));

export default useSettingsUi;
