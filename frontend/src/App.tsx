import { LibraryRail } from "@/components/papermind/LibraryRail";
import { ReaderPane } from "@/components/papermind/ReaderPane";
import { ChatPane } from "@/components/papermind/ChatPane";
import SettingsDialog from "./app/settings-dialog";

export default function App() {
  return (
    <div className="flex h-screen min-h-[680px] w-full overflow-hidden bg-background text-foreground">
      <LibraryRail />
      <ReaderPane />
      <ChatPane />
      <SettingsDialog />
    </div>
  );
}
