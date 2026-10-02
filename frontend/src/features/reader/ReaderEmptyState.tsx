import { UploadIcon } from "lucide-react";

/**
 * What the reader shows with no Document open.
 *
 * It says what to do rather than only that there is nothing here, because the
 * reader usually reaches this pane by accident on a phone.
 */
export default function ReaderEmptyState() {
  return (
    <div className="mx-auto flex min-h-[520px] max-w-[560px] flex-col items-center justify-center">
      <div className="paper-grain w-full bg-paper px-10 py-16 text-center shadow-sheet">
        <div className="mx-auto grid size-10 place-items-center border border-rule bg-canvas text-ink-faint">
          <UploadIcon className="size-5" />
        </div>
        <h2 className="mt-6 font-serif text-xl font-medium">No document selected</h2>
        <p className="mx-auto mt-2 max-w-[32ch] font-serif text-sm leading-relaxed text-ink-soft">
          Select a publication from your library to review its contents in this pane — the sheet
          keeps your margins clean.
        </p>
        <p className="label-meta mt-6">Ingest a PDF from the rail to begin</p>
      </div>
    </div>
  );
}