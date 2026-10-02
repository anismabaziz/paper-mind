import { useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useFiles, useUploadFile, useTouchFileOpened } from "@/hooks/useFiles";
import { useLibraryItemActions } from "@/hooks/useLibraryItemActions";
import { useLibraryListing } from "@/hooks/useLibraryListing";
import usePdfStore from "@/store/pdf-state";
import useSettingsUi from "@/store/settings-ui";
import useBriefUi from "@/store/brief-ui";
import useMobileUi from "@/store/mobile-ui";
import { isBriefBlocked, isBriefReadable } from "@/services/research";
import { getErrorMessage } from "@/lib/api-error";
import type { LibraryFailure } from "./LibraryFailures";
import LibraryFailures, { DeleteFailure } from "./LibraryFailures";
import LibraryList from "./LibraryList";
import RailHeader from "./RailHeader";
import RailSearch from "./RailSearch";
import RailTabs from "./RailTabs";
import RailFooter from "./RailFooter";

/**
 * The library rail: what is in it, what each Document's state allows, and the
 * few actions the rail itself starts.
 *
 * Rows decide their own state and actions; what is left here is which Documents
 * are listed, which one is open, and which of the rail's requests is running.
 */
export function LibraryRail() {
  const queryClient = useQueryClient();
  const { file: selectedFile, setFile } = usePdfStore();
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const openSettings = useSettingsUi((s) => s.open);
  const openBrief = useBriefUi((s) => s.open);
  const setLibraryOpen = useMobileUi((s) => s.setLibraryOpen);
  const filesQuery = useFiles();
  const touchOpened = useTouchFileOpened();

  const files = filesQuery.data?.files ?? [];
  const listing = useLibraryListing({
    files,
    selectedFile,
    onOpen: (file) => {
      setFile(file);
      touchOpened.mutate(file);
      setLibraryOpen(false);
    },
    onFallback: setFile,
    onDeselect: () => setFile(null),
  });

  const uploadMutation = useUploadFile({
    onSuccess: (data) => {
      // A fresh upload counts as opened so it enters Recent readings. The
      // upload response already carries the queued ingestion job, so the
      // listing polls it directly with no client-side processing call.
      touchOpened.mutate(data.file);
      queryClient.invalidateQueries({ queryKey: ["files"] });
    },
    onError: (err) => {
      setUploadError(getErrorMessage(err, "Upload failed. Try again."));
    },
  });

  const actions = useLibraryItemActions({
    selectedFileId: selectedFile?.id ?? null,
    onSelect: listing.onOpen,
    onDeselect: () => setFile(null),
  });

  const uploadFailures: LibraryFailure[] = uploadError
    ? [
        {
          key: "upload",
          heading: "Upload failed",
          detail: uploadError,
          reset: () => setUploadError(null),
        },
      ]
    : [];

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    if (e.target.files?.[0]) uploadMutation.mutate(e.target.files[0]);
    if (e.target) e.target.value = "";
  }

  // The same rule the brief dialog applies, so the control is offered exactly
  // when a brief could actually be started rather than opening onto a refusal.
  const briefable = files.filter((file) => isBriefReadable(file) && !isBriefBlocked(file));

  return (
    <aside className="flex w-64 max-w-full min-w-0 shrink-0 flex-col overflow-x-hidden border-r border-rule bg-sidebar">
      <RailHeader
        uploadPending={uploadMutation.isPending}
        onPick={() => fileInputRef.current?.click()}
        onFileChange={handleFileChange}
        fileInputRef={fileInputRef}
      />

      <RailSearch query={listing.query} onQuery={listing.setQuery} />

      <nav className="scroll-slim flex-1 overflow-x-hidden overflow-y-auto min-w-0 max-w-full px-3 py-4">
        <RailTabs
          tab={listing.tab}
          onTab={listing.setTab}
          totalCount={files.length}
          recentCount={listing.recents.length}
          listTitle={listing.listTitle}
        />

        <LibraryFailures failures={[...uploadFailures, ...actions.failures]} />
        {actions.deleteFailure && <DeleteFailure {...actions.deleteFailure} />}

        <LibraryList
          rows={actions.rowsFor(listing.filtered)}
          tab={listing.tab}
          state={
            filesQuery.isPending
              ? "pending"
              : filesQuery.isError
                ? "error"
                : listing.filtered.length === 0
                  ? "empty"
                  : "ready"
          }
          errorText={getErrorMessage(filesQuery.error)}
          onRetryLoad={() => filesQuery.refetch()}
          onUpload={() => fileInputRef.current?.click()}
          onShowLibrary={() => listing.setTab("library")}
        />
      </nav>

      <RailFooter
        briefCount={briefable.length}
        onOpenBrief={openBrief}
        onOpenSettings={openSettings}
      />
    </aside>
  );
}
