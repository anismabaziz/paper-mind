import { lazy, Suspense, useEffect, useState } from "react";
import usePdfStore from "@/store/pdf-state";
import useMobileUi from "@/store/mobile-ui";
import { usePdfFileData } from "@/hooks/usePdfFileData";
import { useReaderNavigation } from "@/hooks/useReaderNavigation";
import { useReaderDocumentState } from "@/hooks/useReaderDocumentState";
import { isDetached } from "@/lib/bytes";
import { ThumbnailPlaceholder } from "./PageStripPlaceholder";
import PageStrip from "./PageStrip";
import OutlineStrip from "./OutlineStrip";
import ReaderToolbar from "./ReaderToolbar";
import ReaderBanners from "./ReaderBanners";
import ReaderEmptyState from "./ReaderEmptyState";
import ReaderTitleSheet from "./ReaderTitleSheet";
import ReaderDocumentSheet from "./ReaderDocumentSheet";

const ReaderDocument = lazy(() => import("./ReaderDocument"));

/**
 * The reading pane: the open Document, its pages, and everything that decides
 * what can be done with it.
 *
 * The PDF is fetched and rendered in one place each — the bytes here, the
 * renderer in its own module — because pdf.js detaches whatever buffer it is
 * handed, and two renderers sharing one would blank each other.
 */
export function ReaderPane() {
  const { file, citationTarget, setFile } = usePdfStore();
  const [zoom, setZoom] = useState(100);
  const [showStrip, setShowStrip] = useState(true);
  const { setLibraryOpen, setChatOpen } = useMobileUi();

  // Single owner of the fetched bytes. Each lazy Renderer (sheet, strip) clones
  // its own copy; unmounting a Renderer releases its clone and switching
  // Documents drops the source, so previous buffers are never retained.
  const { data: fileData, error: fileDataError, reload: reloadFileData } = usePdfFileData(file);

  // Reopening the strip remounts its Document. If the worker already detached
  // the source, refetch fresh bytes so thumbnails reload instead of rendering
  // from a dead view.
  useEffect(() => {
    if (showStrip && fileData && isDetached(fileData)) reloadFileData();
  }, [showStrip, fileData, reloadFileData]);

  const doc = useReaderDocumentState(file, () => setFile(null));
  const nav = useReaderNavigation({
    fileId: file?.id ?? null,
    zoom,
    pagesReady: Boolean(fileData),
    citationTarget,
  });

  const zoomTo = (next: number) => setZoom(Math.min(140, Math.max(80, next)));
  const goToPage = (next: number) => {
    const target = Math.max(1, next);
    nav.setPage(target);
    nav.scrollToPage(target);
  };

  return (
    <main className="flex min-h-0 min-w-0 flex-1 flex-col bg-canvas" data-testid="reader">
      {file && (
        <ReaderBanners
          retryFailed={doc.retry.failed}
          retryDetail={doc.retry.detail}
          cancelFailed={doc.cancel.failed}
          cancelDetail={doc.cancel.detail}
          statusStale={doc.statusStale}
          statusMissing={doc.statusMissing}
          statusErrorText={doc.statusErrorText}
          deleteFailed={doc.remove.failed}
          deleteDetail={doc.remove.detail}
          onRetryStatus={doc.refetchStatus}
          onRetryDelete={doc.remove.run}
          onDismissDelete={doc.remove.dismiss}
        />
      )}

      <ReaderToolbar
        file={file}
        statusLine={doc.statusLine}
        zoom={zoom}
        onZoom={zoomTo}
        page={nav.page}
        numPages={nav.numPages}
        onPage={goToPage}
        showStrip={showStrip}
        onToggleStrip={() => setShowStrip((v) => !v)}
        onDelete={doc.remove.run}
        onOpenLibrary={() => setLibraryOpen(true)}
        onOpenChat={() => setChatOpen(true)}
      />

      {file && showStrip && (
        <OutlineStrip
          outline={doc.outline}
          state={doc.outlineState}
          errorText={doc.outlineErrorText}
          activePage={nav.page}
          onJump={goToPage}
          onRetry={doc.refetchOutline}
          stripRef={nav.outlineStripRef}
        >
          <Suspense
            fallback={
              <div className="px-4 py-3">
                <ThumbnailPlaceholder count={doc.outlinePageCount ?? nav.numPages ?? 4} />
              </div>
            }
          >
            <PageStrip
              key={`${file.id}-strip`}
              fileId={file.id}
              source={fileData}
              pageCount={doc.outlinePageCount ?? nav.numPages ?? 0}
              activePage={nav.page}
              onSelect={goToPage}
              onBufferDetached={reloadFileData}
            />
          </Suspense>
        </OutlineStrip>
      )}

      <div className="relative flex min-h-0 min-w-0 flex-1 overflow-hidden">
        <div
          ref={nav.scrollRef}
          onScroll={nav.handleScroll}
          className="min-h-0 min-w-0 flex-1 overflow-x-hidden overflow-y-auto overscroll-contain px-3 sm:px-6 py-8 [scrollbar-width:none] [-ms-overflow-style:none] [&::-webkit-scrollbar]:hidden"
        >
          {!file ? (
            <ReaderEmptyState />
          ) : (
            <div
              className="mx-auto origin-top transition-[width] duration-200"
              style={{ width: `${Math.min(880, 7.6 * zoom)}px`, maxWidth: "100%" }}
            >
              <ReaderTitleSheet
                file={file}
                page={nav.page}
                isProcessed={doc.isProcessed}
                isJobActive={doc.isJobActive}
                isJobRetryable={doc.isJobRetryable}
                isJobFailed={doc.isJobFailed}
                ingestionJob={doc.ingestionJob}
                retryPending={doc.retry.pending}
                onRetry={doc.retry.run}
              />
              <ReaderDocumentSheet
                page={nav.page}
                downloadError={fileDataError}
                onRetryDownload={() => reloadFileData()}
                onChooseAnother={() => setLibraryOpen(true)}
                isProcessed={doc.isProcessed}
                isJobActive={doc.isJobActive}
                isJobRetryable={doc.isJobRetryable}
                isJobFailed={doc.isJobFailed}
                canCancel={doc.canCancel}
                ingestionJob={doc.ingestionJob}
                retryPending={doc.retry.pending}
                cancelPending={doc.cancel.pending}
                onRetryIndexing={doc.retry.run}
                onCancelIndexing={doc.cancel.run}
              >
                <ReaderDocument
                  file={file}
                  zoom={zoom}
                  onLoadSuccess={nav.setNumPages}
                  data={fileData}
                  activePage={nav.page}
                  flashedPage={nav.flashedPage}
                  pendingPage={nav.pendingPage}
                  onRenderError={reloadFileData}
                />
              </ReaderDocumentSheet>
            </div>
          )}
        </div>

        <div className="pointer-events-none absolute top-0 right-0 bottom-0 w-1 bg-transparent">
          <div className="h-full w-px bg-rule" />
          <div
            className="absolute top-0 left-0 w-px bg-marker transition-[height] duration-150"
            style={{ height: `${nav.progress}%` }}
          />
        </div>
      </div>
    </main>
  );
}