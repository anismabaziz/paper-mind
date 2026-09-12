import { LibraryRail } from "@/components/papermind/LibraryRail";
import { ReaderPane } from "@/components/papermind/ReaderPane";
import { ChatPane } from "@/components/papermind/ChatPane";
import SettingsDialog from "./app/settings-dialog";
import useMobileUi from "@/store/mobile-ui";

export default function App() {
  const { libraryOpen, chatOpen, setLibraryOpen, setChatOpen } = useMobileUi();

  return (
    <div className="flex h-screen max-w-full min-h-[680px] w-full min-w-0 overflow-hidden overflow-x-hidden bg-background text-foreground flex-col lg:flex-row">
      <div className="hidden lg:flex lg:w-64 lg:shrink-0">
        <LibraryRail />
      </div>
      <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <ReaderPane />
      </div>
      <div className="hidden lg:flex lg:w-[26rem] lg:shrink-0">
        <ChatPane />
      </div>

      {libraryOpen && (
        <div className="fixed inset-0 z-40 flex lg:hidden">
          <div className="w-64 shrink-0 bg-sidebar">
            <LibraryRail />
          </div>
          <button type="button" aria-label="Close library" className="flex-1 bg-black/40" onClick={() => setLibraryOpen(false)} />
        </div>
      )}

      {chatOpen && (
        <div className="fixed inset-0 z-40 flex justify-end lg:hidden">
          <button type="button" aria-label="Close chat" className="flex-1 bg-black/40" onClick={() => setChatOpen(false)} />
          <div className="w-[85vw] max-w-[26rem] shrink-0 bg-background">
            <ChatPane />
          </div>
        </div>
      )}

      <SettingsDialog />
    </div>
  );
}
