import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Loader2, Settings, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  getSettings,
  saveSettings,
  verifySettings,
  type IModelCatalog,
} from "@/services/settings";
import { cn } from "@/lib/utils";
import { useEscapeKey } from "@/hooks/useEscape";
import useSettingsUi from "@/store/settings-ui";

type Feedback = { kind: "success" | "error"; text: string } | null;

export default function SettingsDialog() {
  const isOpen = useSettingsUi((state) => state.isOpen);
  return isOpen ? <SettingsDialogContent /> : null;
}

function SettingsDialogContent() {
  const { isOpen, close } = useSettingsUi();
  const queryClient = useQueryClient();
  const settingsQuery = useQuery({
    queryKey: ["settings"],
    queryFn: getSettings,
    retry: false,
  });

  const [provider, setProvider] = useState("");
  const [model, setModel] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [feedback, setFeedback] = useState<Feedback>(null);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const closeRef = useRef<HTMLButtonElement>(null);
  const restoreFocusRef = useRef<HTMLElement | null>(null);
  const requestControllerRef = useRef<AbortController | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      requestControllerRef.current?.abort();
      requestControllerRef.current = null;
    };
  }, []);

  const closeSettings = useCallback(() => {
    requestControllerRef.current?.abort();
    requestControllerRef.current = null;
    setApiKey("");
    setFeedback(null);
    close();
  }, [close]);

  const settings = settingsQuery.data;
  const loadErrorData = (settingsQuery.error as {
    response?: {
      data?: {
        supported_models?: IModelCatalog;
      };
    };
  } | null)?.response?.data ?? null;
  const storedSupportedModels =
    settings?.supported_models ?? loadErrorData?.supported_models;
  const effectiveSupportedModels = useMemo(
    () => storedSupportedModels ?? {},
    [storedSupportedModels],
  );
  const effectiveModelCatalog = useMemo(
    () => Object.values(effectiveSupportedModels).flat(),
    [effectiveSupportedModels],
  );
  const loadErrorText = settingsQuery.isError
    ? errorMessage(settingsQuery.error, "Could not load settings.")
    : null;

  // Move focus into the dialog on open, restore it on close.
  useEffect(() => {
    if (!isOpen) return;
    restoreFocusRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    closeRef.current?.focus();
    return () => {
      restoreFocusRef.current?.focus();
    };
  }, [isOpen]);
  useEscapeKey(isOpen, closeSettings);

  useEffect(() => {
    if (settings) {
      setProvider(settings.provider ?? "");
      setModel(settings.model ?? "");
    }
  }, [settings]);

  useEffect(() => {
    if (!model) return;
    const supported = effectiveModelCatalog.some(
      (entry) => entry.provider === provider && entry.id === model,
    );
    if (!supported) setModel("");
  }, [effectiveModelCatalog, model, provider]);

  const allProviders = Object.keys(effectiveSupportedModels);
  const catalogModels = effectiveModelCatalog.filter(
    (entry) => entry.provider === provider,
  );
  const models = catalogModels.map((entry) => entry.id);
  const selectedModel = catalogModels.find((entry) => entry.id === model);
  const savedMaskedKey = settings?.masked_key ?? null;
  const canRecoverFromLoadError = allProviders.length > 0;
  const hasLocalChatModel = effectiveModelCatalog.some(
    (entry) => entry.data_location === "local",
  );

  const beginRequest = () => {
    requestControllerRef.current?.abort();
    const controller = new AbortController();
    requestControllerRef.current = controller;
    return controller;
  };

  const finishRequest = (controller: AbortController) => {
    if (requestControllerRef.current === controller) {
      requestControllerRef.current = null;
    }
  };

  const handleSave = async () => {
    if (!provider || !model || !apiKey || saving || testing) return;
    const controller = beginRequest();
    setSaving(true);
    setFeedback(null);
    try {
      await saveSettings(
        { provider, model, api_key: apiKey },
        controller.signal,
      );
      if (!mountedRef.current || controller.signal.aborted) return;
      setApiKey("");
      await queryClient.invalidateQueries({ queryKey: ["settings"] });
      if (!mountedRef.current || controller.signal.aborted) return;
      setFeedback({ kind: "success", text: "Settings saved." });
    } catch (e) {
      if (mountedRef.current && !controller.signal.aborted) {
        setFeedback({ kind: "error", text: errorMessage(e, "Could not save settings.") });
      }
    } finally {
      finishRequest(controller);
      if (mountedRef.current) setSaving(false);
    }
  };

  const handleTest = async () => {
    if (testing || saving) return;
    if (!provider || !model || !apiKey) {
      setFeedback({
        kind: "error",
        text: "Provider, model, and API key are required before testing.",
      });
      return;
    }
    const controller = beginRequest();
    setTesting(true);
    setFeedback(null);
    try {
      const result = await verifySettings(
        { provider, model, api_key: apiKey },
        controller.signal,
      );
      if (!mountedRef.current || controller.signal.aborted) return;
      setFeedback(
        result.ok
          ? {
              kind: "success",
              text: "Connection works. Save settings to make it active.",
            }
          : { kind: "error", text: result.error ?? "Connection failed." }
      );
    } catch (e) {
      if (mountedRef.current && !controller.signal.aborted) {
        setFeedback({ kind: "error", text: errorMessage(e, "Could not run the connection test.") });
      }
    } finally {
      finishRequest(controller);
      if (mountedRef.current) setTesting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-ink/40 p-4 backdrop-blur-sm">
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-dialog-title"
        className="w-full max-w-md border border-rule bg-paper shadow-sheet p-6 relative"
      >
        <button
          ref={closeRef}
          onClick={closeSettings}
          aria-label="Close settings"
          className="absolute right-4 top-4 text-ink-faint hover:text-ink cursor-pointer"
        >
          <X size={16} />
        </button>
        <div className="flex items-center gap-2 mb-5">
          <Settings size={16} className="text-ink-soft" />
          <h2 id="settings-dialog-title" className="font-mono text-xs font-semibold uppercase tracking-widest text-ink">
            Settings
          </h2>
        </div>
        <div className="space-y-3">
          {settingsQuery.isLoading ? (
            <div className="flex justify-center py-6">
              <Loader2 size={18} className="animate-spin text-ink-faint" />
            </div>
          ) : settingsQuery.isError && !canRecoverFromLoadError ? (
            <FormFeedback kind="error" text={loadErrorText ?? "Could not load settings."} />
          ) : (
            <>
              {loadErrorText && <FormFeedback kind="error" text={loadErrorText} />}
              <label className="block">
                <span className="label-meta">Provider</span>
                <select
                  value={provider}
                  onChange={(e) => {
                    setProvider(e.target.value);
                    setModel("");
                  }}
                  className={selectClass}
                >
                  <option value="">Select a provider…</option>
                  {allProviders.map((p) => (
                    <option key={p} value={p}>
                      {p}
                    </option>
                  ))}
                </select>
              </label>

              <label className="block">
                <span className="label-meta">Model</span>
                <select
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  disabled={!provider}
                  className={selectClass}
                >
                  <option value="">Select a model…</option>
                  {models.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </label>

              {selectedModel && (
                <div
                  className="border-y border-rule/70 py-3"
                  aria-label="Model capabilities"
                >
                  <div className="mb-2 font-mono text-xs text-ink">
                    {selectedModel.id}
                  </div>
                  <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
                    <div>
                      <div className="text-ink-faint">Context</div>
                      <div className="mt-0.5 text-ink">
                        {formatTokenCount(selectedModel.context_window_tokens)} tokens
                      </div>
                    </div>
                    <div>
                      <div className="text-ink-faint">Output</div>
                      <div className="mt-0.5 text-ink">
                        {formatTokenCount(selectedModel.max_output_tokens)} tokens
                      </div>
                    </div>
                    <div>
                      <div className="text-ink-faint">Structured output</div>
                      <div className="mt-0.5 text-ink">
                        {selectedModel.structured_output ? "Supported" : "Not supported"}
                      </div>
                    </div>
                    <div>
                      <div className="text-ink-faint">Tool use</div>
                      <div className="mt-0.5 text-ink">
                        {selectedModel.tool_use ? "Supported" : "Not supported"}
                      </div>
                    </div>
                    <div>
                      <div className="text-ink-faint">Verification timeout</div>
                      <div className="mt-0.5 text-ink">
                        {selectedModel.timeout_seconds} seconds
                      </div>
                    </div>
                  </div>
                  <p className="mt-3 text-xs leading-5 text-ink-soft">
                    {formatUsd(selectedModel.input_cost_per_million_usd)} input / {" "}
                    {formatUsd(selectedModel.output_cost_per_million_usd)} output per 1M
                    tokens · {selectedModel.pricing_tier} pricing ·{" "}
                    {selectedModel.data_location === "cloud"
                      ? "cloud processing"
                      : "local processing"}
                  </p>
                </div>
              )}

              <label className="block">
                <span className="label-meta">API key</span>
                <Input
                  type="password"
                  placeholder={savedMaskedKey
                    ? `Saved key ${savedMaskedKey} — paste a new key to replace`
                    : "Paste your API key"}
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  className="h-10 bg-paper border-rule text-xs focus-visible:border-ink focus-visible:ring-0"
                  autoComplete="off"
                />
              </label>

              <p className="rounded-sm border border-rule/70 bg-canvas px-3 py-2 text-xs leading-5 text-ink-soft">
                {selectedModel?.data_location === "local"
                  ? "Questions and retrieved passages stay on this machine."
                  : `Questions and retrieved passages leave this machine when you use ${
                      provider || "the selected cloud provider"
                    }.`} {" "}
                {hasLocalChatModel
                  ? "A local-only chat option is available in the catalog."
                  : "This build has no local-only chat option."} Embeddings and reranking
                stay on this machine.
              </p>

              {feedback && <FormFeedback kind={feedback.kind} text={feedback.text} />}

              <div className="flex gap-2 pt-1">
                <Button
                  onClick={handleSave}
                   disabled={!provider || !model || !apiKey || saving || testing}
                  className="flex-1 h-10 bg-ink text-paper text-xs hover:bg-ink/90"
                >
                  {saving && <Loader2 size={14} className="animate-spin" />}
                  Save
                </Button>
                <Button
                  onClick={handleTest}
                  disabled={saving || testing}
                  variant="outline"
                  className="flex-1 h-10 text-xs border-rule bg-paper hover:bg-canvas text-ink"
                >
                  {testing && <Loader2 size={14} className="animate-spin" />}
                  Test connection
                </Button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function FormFeedback({ kind, text }: { kind: "success" | "error"; text: string }) {
  return (
    <p
      className={cn(
        "rounded-sm border px-3 py-2 font-mono text-xs",
        kind === "success"
          ? "border-ink/20 bg-canvas text-ink"
          : "border-destructive/20 bg-destructive/5 text-destructive"
      )}
    >
      {text}
    </p>
  );
}

function errorMessage(e: unknown, fallback: string): string {
  if (e instanceof Error && "response" in e) {
    const data = (e as { response?: { data?: { error?: string } } }).response?.data;
    if (data?.error) return data.error;
  }
  return fallback;
}

function formatTokenCount(value: number): string {
  return new Intl.NumberFormat("en-US").format(value);
}

function formatUsd(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 3,
  }).format(value);
}

const selectClass =
  "h-10 w-full rounded-sm border border-rule bg-paper px-3 text-xs text-ink focus:outline-none focus:border-ink disabled:bg-canvas disabled:text-ink-faint font-mono";
