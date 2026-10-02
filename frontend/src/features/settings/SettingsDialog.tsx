import { useMemo, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { Loader2, Settings, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { getSettings, type IModelCatalog } from "@/services/settings";
import FormFeedback from "@/features/settings/FormFeedback";
import { useEscapeKey } from "@/hooks/useEscape";
import { useSettingsForm } from "@/hooks/useSettingsForm";
import useSettingsUi from "@/store/settings-ui";
import { useFocusTrap } from "@/hooks/useFocusTrap";
import { errorMessage } from "@/lib/settings-format";
import SettingsForm from "./SettingsForm";

export default function SettingsDialog() {
  const isOpen = useSettingsUi((state) => state.isOpen);
  return isOpen ? <SettingsDialogContent /> : null;
}

/**
 * The dialog chrome and the three states it can be in: still loading, failed
 * with nothing to fall back on, and the form itself.
 *
 * A failed load that still carries a catalog is not a dead end — the reader can
 * type a new provider and model over it — so the error is shown above the form
 * rather than instead of it.
 */
function SettingsDialogContent() {
  const { isOpen, close } = useSettingsUi();
  const settingsQuery = useQuery({
    queryKey: ["settings"],
    queryFn: getSettings,
    retry: false,
  });

  const supportedModels = useSupportedModels(settingsQuery.data, settingsQuery.error);
  const catalog = useMemo(
    () => Object.values(supportedModels).flat(),
    [supportedModels],
  );

  const form = useSettingsForm(settingsQuery.data, catalog, close);
  const closeRef = useRef<HTMLButtonElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);

  // Trap focus while open and restore it to the invoking control on close.
  useFocusTrap(isOpen, dialogRef, closeRef);
  // Escape aborts a read-only verification but never interrupts a save.
  useEscapeKey(isOpen && !form.saving, form.close);

  const loadErrorText = settingsQuery.isError
    ? errorMessage(settingsQuery.error, "Could not load settings.")
    : null;
  const providers = Object.keys(supportedModels);
  const modelsForProvider = catalog
    .filter((entry) => entry.provider === form.provider)
    .map((entry) => entry.id);

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-ink/40 p-4 backdrop-blur-sm"
      data-testid="settings-backdrop"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) form.close();
      }}
    >
      <div
        ref={dialogRef}
        tabIndex={-1}
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-dialog-title"
        aria-busy={form.busy}
        className="w-full max-w-md border border-rule bg-paper shadow-sheet p-6 relative focus:outline-none"
        data-testid="settings-dialog"
      >
        <button
          ref={closeRef}
          onClick={form.close}
          disabled={form.saving}
          aria-label="Close settings"
          className="absolute right-4 top-4 grid size-11 place-items-center text-ink-faint hover:text-ink cursor-pointer disabled:opacity-40"
        >
          <X size={16} />
        </button>
        <div className="flex items-center gap-2 mb-5">
          <Settings size={16} className="text-ink-soft" />
          <h2
            id="settings-dialog-title"
            className="font-mono text-xs font-semibold uppercase tracking-widest text-ink"
          >
            Settings
          </h2>
        </div>
        <div className="space-y-3">
          {settingsQuery.isLoading ? (
            <div
              className="flex justify-center py-6"
              role="status"
              aria-label="Loading settings"
            >
              <Loader2
                size={18}
                className="animate-spin text-ink-faint motion-reduce:animate-none"
                aria-hidden="true"
              />
              <span className="sr-only">Loading settings…</span>
            </div>
          ) : settingsQuery.isError && providers.length === 0 ? (
            <LoadFailure message={loadErrorText} onRetry={() => settingsQuery.refetch()} />
          ) : (
            <SettingsForm
              providers={providers}
              provider={form.provider}
              onProvider={form.selectProvider}
              models={modelsForProvider}
              model={form.model}
              onModel={form.selectModel}
              selectedModel={catalog.find(
                (entry) => entry.id === form.model && entry.provider === form.provider,
              )}
              apiKey={form.apiKey}
              onApiKey={form.setApiKey}
              savedMaskedKey={settingsQuery.data?.masked_key ?? null}
              hasLocalChatModel={catalog.some((entry) => entry.data_location === "local")}
              feedback={form.feedback}
              busy={form.busy}
              saving={form.saving}
              testing={form.testing}
              loadErrorText={loadErrorText}
              onRetryLoad={() => settingsQuery.refetch()}
              onSave={() => void form.save()}
              onTest={() => void form.test()}
            />
          )}
        </div>
      </div>
    </div>
  );
}

/** No settings and no catalog: the form has nothing to offer, only a retry. */
function LoadFailure({ message, onRetry }: { message: string | null; onRetry: () => void }) {
  return (
    <div className="space-y-3">
      <FormFeedback
        kind="error"
        text={message ?? "Could not load settings."}
      />
      <p className="text-xs leading-5 text-ink-soft">
        Saved settings were left unchanged. Retrying reloads them — nothing you typed is
        restored from the server.
      </p>
      <Button
        type="button"
        onClick={onRetry}
        variant="outline"
        className="h-11 min-h-[44px] w-full text-xs border-rule bg-paper hover:bg-canvas text-ink"
        data-testid="settings-retry"
      >
        Retry loading settings
      </Button>
    </div>
  );
}

/**
 * The catalog the form is built from, taken from the loaded settings or from a
 * failed load that still sent one. A failed load usually carries the catalog
 * precisely so the reader is not locked out of the settings they can still see.
 */
function useSupportedModels(
  settings: { supported_models: IModelCatalog } | undefined,
  error: unknown,
): IModelCatalog {
  return useMemo(() => {
    const loaded = settings?.supported_models;
    if (loaded) return loaded;
    const fromError = (error as { response?: { data?: { supported_models?: IModelCatalog } } })
      ?.response?.data?.supported_models;
    return fromError ?? {};
  }, [error, settings]);
}
