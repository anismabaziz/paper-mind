import { Loader2, Upload } from "lucide-react";

type Props = {
  uploadPending: boolean;
  onPick: () => void;
  onFileChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  fileInputRef: React.RefObject<HTMLInputElement | null>;
};

/**
 * The rail's header: what this workspace is, and the one button that adds to
 * it. The file input lives here rather than at the rail's root so the control
 * that opens it and the control it feeds stay together.
 */
export default function RailHeader({ uploadPending, onPick, onFileChange, fileInputRef }: Props) {
  return (
    <header className="flex h-16 items-center justify-between border-b border-rule px-5">
      <div className="flex items-center gap-3">
        <span className="grid size-8 place-items-center bg-ink font-mono text-[0.7rem] font-bold text-paper">PM</span>
        <div>
          <p className="font-mono text-sm font-bold">PaperMind</p>
          <p className="text-[0.65rem] text-ink-faint">Research workspace</p>
        </div>
      </div>
      <button
        type="button"
        onClick={onPick}
        disabled={uploadPending}
        className="grid size-8 place-items-center border border-rule text-ink-soft hover:border-marker hover:text-marker disabled:opacity-40"
        aria-label="Upload paper"
        title="Upload paper"
        data-testid="upload-paper"
      >
        {uploadPending ? <Loader2 className="size-3.5 animate-spin" /> : <Upload className="size-3.5" />}
      </button>
      <input
        ref={fileInputRef}
        type="file"
        hidden
        accept=".pdf"
        onChange={onFileChange}
        data-testid="file-input"
      />
    </header>
  );
}