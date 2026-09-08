import { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Loader2, Settings, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { clearToken, getToken, login } from "@/services/auth";
import {
  getSettings,
  saveSettings,
  verifySettings,
} from "@/services/settings";
import { cn } from "@/lib/utils";
import useSettingsUi from "@/store/settings-ui";

type Feedback = { kind: "success" | "error"; text: string } | null;

export default function SettingsDialog() {
  const { isOpen, close } = useSettingsUi();
  // Demo mode serves settings with no token, so we always attempt the load
  // first and only fall back to the login form when the backend answers 401.
  const [loginRequired, setLoginRequired] = useState(false);

  useEffect(() => {
    if (isOpen) setLoginRequired(false);
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-ink/40 p-4 backdrop-blur-sm">
      <div className="w-full max-w-md border border-rule bg-paper shadow-sheet p-6 relative">
        <button
          onClick={close}
          aria-label="Close settings"
          className="absolute right-4 top-4 text-ink-faint hover:text-ink cursor-pointer"
        >
          <X size={16} />
        </button>
        <div className="flex items-center gap-2 mb-5">
          <Settings size={16} className="text-ink-soft" />
          <h2 className="font-mono text-xs font-semibold uppercase tracking-widest text-ink">
            Settings
          </h2>
        </div>
        {loginRequired ? (
          <LoginForm onLoggedIn={() => setLoginRequired(false)} />
        ) : (
          <SettingsForm
            onUnauthorized={() => {
              clearToken();
              setLoginRequired(true);
            }}
          />
        )}
      </div>
    </div>
  );
}

function LoginForm({ onLoggedIn }: { onLoggedIn: () => void }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async () => {
    if (!email.trim() || !password || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      await login(email.trim(), password);
      onLoggedIn();
    } catch (e) {
      setError(errorMessage(e, "Login failed. Check your credentials."));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        handleSubmit();
      }}
      className="space-y-3"
    >
      <p className="font-serif text-xs leading-relaxed text-ink-soft">
        Sign in to manage your own chat provider and API key. Demo mode runs
        on the shared demo user instead.
      </p>
      <Input
        type="email"
        placeholder="Email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        className="h-10 bg-paper border-rule text-xs focus-visible:border-ink focus-visible:ring-0"
        autoComplete="email"
      />
      <Input
        type="password"
        placeholder="Password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        className="h-10 bg-paper border-rule text-xs focus-visible:border-ink focus-visible:ring-0"
        autoComplete="current-password"
      />
      {error && <FormFeedback kind="error" text={error} />}
      <Button type="submit" className="w-full h-10 bg-ink text-paper text-xs hover:bg-ink/90 border-ink" disabled={submitting}>
        {submitting && <Loader2 size={14} className="animate-spin" />}
        Sign in
      </Button>
    </form>
  );
}

function SettingsForm({ onUnauthorized }: { onUnauthorized: () => void }) {
  const queryClient = useQueryClient();
  const settingsQuery = useQuery({
    queryKey: ["settings"],
    queryFn: getSettings,
    retry: false,
  });

  useEffect(() => {
    if (axiosStatus(settingsQuery.error) === 401) onUnauthorized();
    // eslint-disable-next-line react-hooks/exhaustive-deps -- react only to the query error, not the callback identity
  }, [settingsQuery.error]);

  const [provider, setProvider] = useState("");
  const [model, setModel] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [feedback, setFeedback] = useState<Feedback>(null);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);

  const settings = settingsQuery.data;

  useEffect(() => {
    if (settings) {
      setProvider(settings.provider ?? "");
      setModel(settings.model ?? "");
    }
  }, [settings]);

  const providers = Object.keys(settings?.supported_models ?? {});
  const models = provider ? settings?.supported_models[provider] ?? [] : [];
  const savedMaskedKey = settings?.masked_key ?? null;
  const dirty = provider !== (settings?.provider ?? "") || model !== (settings?.model ?? "") || apiKey !== "";

  const handleSave = async () => {
    if (!provider || !model || !apiKey || saving) return;
    setSaving(true);
    setFeedback(null);
    try {
      await saveSettings({ provider, model, api_key: apiKey });
      setApiKey("");
      await queryClient.invalidateQueries({ queryKey: ["settings"] });
      setFeedback({ kind: "success", text: "Settings saved." });
    } catch (e) {
      setFeedback({ kind: "error", text: errorMessage(e, "Could not save settings.") });
    } finally {
      setSaving(false);
    }
  };

  const handleTest = async () => {
    if (testing) return;
    setTesting(true);
    setFeedback(null);
    try {
      // The verify route runs against saved settings, so unsaved edits go
      // to storage first; otherwise the test would check the old key.
      if (dirty) {
        if (!provider || !model || !apiKey) {
          setFeedback({ kind: "error", text: "Provider, model, and API key are required before testing." });
          return;
        }
        await saveSettings({ provider, model, api_key: apiKey });
        setApiKey("");
        await queryClient.invalidateQueries({ queryKey: ["settings"] });
      }
      const result = await verifySettings();
      setFeedback(
        result.ok
          ? { kind: "success", text: dirty ? "Settings saved and the connection works." : "Connection works." }
          : { kind: "error", text: result.error ?? "Connection failed." }
      );
    } catch (e) {
      setFeedback({ kind: "error", text: errorMessage(e, "Could not run the connection test.") });
    } finally {
      setTesting(false);
    }
  };

  const handleLogout = () => {
    clearToken();
    window.location.reload();
  };

  return (
    <div className="space-y-3">
      {settingsQuery.isLoading ? (
        <div className="flex justify-center py-6">
          <Loader2 size={18} className="animate-spin text-ink-faint" />
        </div>
      ) : settingsQuery.isError ? (
        <FormFeedback kind="error" text={errorMessage(settingsQuery.error, "Could not load settings.")} />
      ) : (
        <>
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
              {providers.map((p) => (
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

          <label className="block">
            <span className="label-meta">API key</span>
            <Input
              type="password"
              placeholder={savedMaskedKey ? `Saved key ${savedMaskedKey} — paste a new key to replace` : "Paste your API key"}
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              className="h-10 bg-paper border-rule text-xs focus-visible:border-ink focus-visible:ring-0"
              autoComplete="off"
            />
          </label>

          {feedback && <FormFeedback kind={feedback.kind} text={feedback.text} />}

          <div className="flex gap-2 pt-1">
            <Button
              onClick={handleSave}
              disabled={!provider || !model || !apiKey || saving}
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
          {getToken() && (
            <button
              onClick={handleLogout}
              className="font-mono text-[0.62rem] font-medium uppercase tracking-widest text-ink-faint hover:text-ink cursor-pointer"
            >
              Sign out
            </button>
          )}
        </>
      )}
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

function axiosStatus(e: unknown): number | null {
  if (e instanceof Error && "response" in e) {
    return (e as { response?: { status?: number } }).response?.status ?? null;
  }
  return null;
}

function errorMessage(e: unknown, fallback: string): string {
  if (e instanceof Error && "response" in e) {
    const data = (e as { response?: { data?: { error?: string } } }).response?.data;
    if (data?.error) return data.error;
  }
  return fallback;
}

const selectClass =
  "h-10 w-full rounded-sm border border-rule bg-paper px-3 text-xs text-ink focus:outline-none focus:border-ink disabled:bg-canvas disabled:text-ink-faint font-mono";
