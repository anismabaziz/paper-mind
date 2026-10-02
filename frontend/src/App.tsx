import { useCallback } from "react";
import { useRef } from "react";
import { LibraryRail } from "@/features/library/LibraryRail";
import { ReaderPane } from "@/features/reader/ReaderPane";
import { ChatPane } from "@/features/chat/ChatPane";
import SettingsDialog from "@/features/settings/SettingsDialog";
import { ResearchBriefDialog } from "@/features/brief/ResearchBriefDialog";
import { useEscapeKey } from "@/hooks/useEscape";
import { useFocusTrap } from "@/hooks/useFocusTrap";
import useMobileUi from "@/store/mobile-ui";

export default function App() {
  const { libraryOpen, chatOpen, setLibraryOpen, setChatOpen } = useMobileUi();
  const libraryOverlayRef = useRef<HTMLDivElement>(null);
  const libraryPanelRef = useRef<HTMLDivElement>(null);
  const chatOverlayRef = useRef<HTMLDivElement>(null);
  const chatPanelRef = useRef<HTMLDivElement>(null);

  // Each drawer traps focus while open and restores it to the toolbar
  // control that opened it when closed. The trap covers the whole overlay
  // (panel plus its backdrop control) so Tab cannot leak behind the dialog.
  useFocusTrap(libraryOpen, libraryOverlayRef, libraryPanelRef);
  useFocusTrap(chatOpen, chatOverlayRef, chatPanelRef);

  const closeDrawers = useCallback(() => {
    setLibraryOpen(false);
    setChatOpen(false);
  }, [setLibraryOpen, setChatOpen]);
  const drawersOpen = libraryOpen || chatOpen;
  useEscapeKey(drawersOpen, closeDrawers);

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
        <div ref={libraryOverlayRef} className="fixed inset-0 z-40 flex lg:hidden">
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
        <div ref={chatOverlayRef} className="fixed inset-0 z-40 flex justify-end lg:hidden">
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
      <ResearchBriefDialog />
    </div>
  );
}
