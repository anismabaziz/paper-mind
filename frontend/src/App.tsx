import { LibraryRail } from "@/components/papermind/LibraryRail";
import { ReaderPane } from "@/components/papermind/ReaderPane";
import { ChatPane } from "@/components/papermind/ChatPane";
import SettingsDialog from "./app/settings-dialog";

export default function App() {
  return (
    <div className="flex h-screen max-w-full min-h-[680px] w-full min-w-0 overflow-hidden overflow-x-hidden bg-background text-foreground">
      <LibraryRail />
      <ReaderPane />
      <ChatPane />
      <SettingsDialog />
    </div>
  );
}
