import { useCallback, useEffect, useRef } from "react";
import { LibraryRail } from "@/components/papermind/LibraryRail";
import { ReaderPane } from "@/components/papermind/ReaderPane";
import { ChatPane } from "@/components/papermind/ChatPane";
import SettingsDialog from "./app/settings-dialog";
import { useEscapeKey } from "@/hooks/useEscape";
import useMobileUi from "@/store/mobile-ui";

export default function App() {
  const { libraryOpen, chatOpen, setLibraryOpen, setChatOpen } = useMobileUi();
  const libraryPanelRef = useRef<HTMLDivElement>(null);
  const chatPanelRef = useRef<HTMLDivElement>(null);
  const restoreFocusRef = useRef<HTMLElement | null>(null);

  const closeDrawers = useCallback(() => {
    setLibraryOpen(false);
    setChatOpen(false);
  }, [setLibraryOpen, setChatOpen]);
  const drawersOpen = libraryOpen || chatOpen;
  useEscapeKey(drawersOpen, closeDrawers);

  // Move focus into whichever drawer opens; restore it when both close.
  useEffect(() => {
    if (!drawersOpen) return;
    if (!restoreFocusRef.current) {
      restoreFocusRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    }
    (libraryOpen ? libraryPanelRef : chatPanelRef).current?.focus();
    return () => {
      if (!useMobileUi.getState().libraryOpen && !useMobileUi.getState().chatOpen) {
        restoreFocusRef.current?.focus();
        restoreFocusRef.current = null;
      }
    };
  }, [libraryOpen, chatOpen, drawersOpen]);

  return (
    <div className="flex h-screen max-w-full min-h-[680px] w-full min-w-0 overflow-hidden overflow-x-hidden bg-background text-foreground flex-col lg:flex-row">
      <div className="hidden lg:flex lg:w-64 lg:shrink-0">
        <LibraryRail />
      </div>
      <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
        <ReaderPane />
      </div>
      <div className="hidden lg:flex lg:w-[26rem] lg:shrink-0">
        <ChatPane />
      </div>

      {libraryOpen && (
        <div className="fixed inset-0 z-40 flex lg:hidden">
          <div
            ref={libraryPanelRef}
            tabIndex={-1}
            role="dialog"
            aria-modal="true"
            aria-label="Library"
            className="w-64 shrink-0 bg-sidebar focus:outline-none"
          >
            <LibraryRail />
          </div>
          <button type="button" aria-label="Close library" className="flex-1 bg-black/40" onClick={() => setLibraryOpen(false)} />
        </div>
      )}

      {chatOpen && (
        <div className="fixed inset-0 z-40 flex justify-end lg:hidden">
          <button type="button" aria-label="Close chat" className="flex-1 bg-black/40" onClick={() => setChatOpen(false)} />
          <div
            ref={chatPanelRef}
            tabIndex={-1}
            role="dialog"
            aria-modal="true"
            aria-label="Reading companion"
            className="w-[85vw] max-w-[26rem] shrink-0 bg-background focus:outline-none"
          >
            <ChatPane />
          </div>
        </div>
      )}

      <SettingsDialog />
    </div>
  );
}
