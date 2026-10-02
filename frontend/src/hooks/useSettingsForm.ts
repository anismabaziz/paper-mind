import { useCallback, useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  saveSettings,
  verifySettings,
  type IModelCapabilities,
  type ISettings,
} from "@/services/settings";
import { errorMessage } from "@/lib/settings-format";

export type SettingsFeedback = { kind: "success" | "error"; text: string } | null;

/**
 * What App Settings is edited with, and the two requests it can make.
 *
 * Both requests run under one abort controller, because closing the dialog has
 * to stop the one that is running and refuse to stop the one that writes: a
 * save interrupted halfway would leave the stored outcome ambiguous, while a
 * verification is read-only and simply has nothing left to report.
 *
 * Every state update is guarded by whether the dialog is still mounted and the
 * request still live, so a response that arrives after the reader closed
 * cannot put feedback back on screen.
 */
export function useSettingsForm(
  settings: ISettings | undefined,
  catalog: IModelCapabilities[],
  close: () => void,
) {
  const queryClient = useQueryClient();
  const [provider, setProvider] = useState("");
  const [model, setModel] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [feedback, setFeedback] = useState<SettingsFeedback>(null);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const requestControllerRef = useRef<AbortController | null>(null);
  const mountedRef = useRef(true);
  const savingRef = useRef(false);
  savingRef.current = saving;

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      requestControllerRef.current?.abort();
      requestControllerRef.current = null;
    };
  }, []);

  const closeSettings = useCallback(() => {
    if (requestControllerRef.current && savingRef.current) return;
    requestControllerRef.current?.abort();
    requestControllerRef.current = null;
    setApiKey("");
    setFeedback(null);
    close();
  }, [close]);

  useEffect(() => {
    if (settings) {
      setProvider(settings.provider ?? "");
      setModel(settings.model ?? "");
    }
  }, [settings]);

  // A model the chosen provider does not carry is dropped rather than left
  // selected: the catalog can change under a deployment, and a form that
  // offers a model the server would refuse is worse than an empty one.
  useEffect(() => {
    if (!model) return;
    const supported = catalog.some(
      (entry) => entry.provider === provider && entry.id === model,
    );
    if (!supported) setModel("");
  }, [catalog, model, provider]);

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

  const busy = saving || testing;

  const save = async () => {
    if (!provider || !model || !apiKey || busy) return;
    const controller = beginRequest();
    setSaving(true);
    setFeedback(null);
    try {
      await saveSettings({ provider, model, api_key: apiKey }, controller.signal);
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

  const test = async () => {
    if (busy) return;
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
        setFeedback({
          kind: "error",
          text: errorMessage(e, "Could not run the connection test."),
        });
      }
    } finally {
      finishRequest(controller);
      if (mountedRef.current) setTesting(false);
    }
  };

  return {
    provider,
    model,
    apiKey,
    feedback,
    busy,
    saving,
    testing,
    setApiKey,
    // Switching provider invalidates the chosen model, so it goes with it
    // rather than staying on screen as a model that provider does not have.
    selectProvider: (value: string) => {
      setProvider(value);
      setModel("");
    },
    selectModel: setModel,
    save,
    test,
    close: closeSettings,
  };
}