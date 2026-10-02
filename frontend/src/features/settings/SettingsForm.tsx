import { Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import type { IModelCapabilities } from "@/services/settings";
import FormFeedback from "@/features/settings/FormFeedback";
import type { SettingsFeedback } from "@/hooks/useSettingsForm";
import ModelCapabilities from "./ModelCapabilities";

const selectClass =
  "h-11 min-h-[44px] w-full rounded-sm border border-rule bg-paper px-3 text-xs text-ink focus:outline-none focus:border-ink disabled:bg-canvas disabled:text-ink-faint font-mono";

type Props = {
  providers: string[];
  provider: string;
  onProvider: (value: string) => void;
  models: string[];
  model: string;
  onModel: (value: string) => void;
  selectedModel: IModelCapabilities | undefined;
  apiKey: string;
  onApiKey: (value: string) => void;
  savedMaskedKey: string | null;
  hasLocalChatModel: boolean;
  feedback: SettingsFeedback;
  busy: boolean;
  saving: boolean;
  testing: boolean;
  loadErrorText: string | null;
  onRetryLoad: () => void;
  onSave: () => void;
  onTest: () => void;
};

/**
 * The settings a reader can change, in the order they matter: which provider
 * answers, which model it answers with, and the key that pays for it.
 *
 * Nothing here decides whether a value is acceptable. A save and a connection
 * test are separate acts on purpose, so a working key can be checked before it
 * replaces the one already stored.
 */
export default function SettingsForm({
  providers,
  provider,
  onProvider,
  models,
  model,
  onModel,
  selectedModel,
  apiKey,
  onApiKey,
  savedMaskedKey,
  hasLocalChatModel,
  feedback,
  busy,
  saving,
  testing,
  loadErrorText,
  onRetryLoad,
  onSave,
  onTest,
}: Props) {
  return (
    <form
      className="space-y-3"
      onSubmit={(e) => {
        e.preventDefault();
        onSave();
      }}
    >
      {loadErrorText && (
        <div className="space-y-2">
          <FormFeedback kind="error" text={loadErrorText} />
          <button
            type="button"
            onClick={onRetryLoad}
            className="font-mono text-[0.65rem] text-ink-soft underline underline-offset-2 hover:text-ink"
            data-testid="settings-retry"
          >
            Retry loading saved settings
          </button>
        </div>
      )}
      <label className="block">
        <span className="label-meta">Provider</span>
        <select
          value={provider}
          onChange={(e) => onProvider(e.target.value)}
          className={selectClass}
          data-testid="settings-provider"
        >
          <option value="">Select a provider…</option>
          {providers.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
      </label>

      <label className="block">
        <span className="label-meta">Model</span>
        <select
          value={model}
          onChange={(e) => onModel(e.target.value)}
          disabled={!provider}
          className={selectClass}
          data-testid="settings-model"
        >
          <option value="">Select a model…</option>
          {models.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
      </label>

      {selectedModel && <ModelCapabilities model={selectedModel} />}

      <label className="block">
        <span className="label-meta">API key</span>
        <Input
          type="password"
          placeholder={
            savedMaskedKey
              ? `Saved key ${savedMaskedKey} — paste a new key to replace`
              : "Paste your API key"
          }
          value={apiKey}
          onChange={(e) => onApiKey(e.target.value)}
          className="h-11 min-h-[44px] bg-paper border-rule text-xs focus-visible:border-ink focus-visible:ring-0"
          autoComplete="off"
          data-testid="settings-api-key"
        />
      </label>

      <p className="rounded-sm border border-rule/70 bg-canvas px-3 py-2 text-xs leading-5 text-ink-soft">
        {selectedModel?.data_location === "local"
          ? "Questions and retrieved passages stay on this machine."
          : `Questions and retrieved passages leave this machine when you use ${
              provider || "the selected cloud provider"
            }.`}{" "}
        {hasLocalChatModel
          ? "A local-only chat option is available in the catalog."
          : "This build has no local-only chat option."}{" "}
        Embeddings and reranking stay on this machine.
      </p>

      {feedback && <FormFeedback kind={feedback.kind} text={feedback.text} />}

      <div className="flex gap-2 pt-1">
        <Button
          type="submit"
          disabled={!provider || !model || !apiKey || busy}
          className="flex-1 h-11 min-h-[44px] bg-ink text-paper text-xs hover:bg-ink/90"
          data-testid="settings-save"
        >
          {saving && <Loader2 size={14} className="animate-spin" aria-hidden="true" />}
          {saving ? "Saving…" : "Save"}
        </Button>
        <Button
          type="button"
          onClick={onTest}
          disabled={busy}
          variant="outline"
          className="flex-1 h-11 min-h-[44px] text-xs border-rule bg-paper hover:bg-canvas text-ink"
          data-testid="settings-test"
        >
          {testing && <Loader2 size={14} className="animate-spin" aria-hidden="true" />}
          {testing ? "Testing…" : "Test connection"}
        </Button>
      </div>
    </form>
  );
}
