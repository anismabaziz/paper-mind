import client from "./client";

export interface IModelCapabilities {
  provider: string;
  id: string;
  context_window_tokens: number;
  max_output_tokens: number;
  structured_output: boolean;
  tool_use: boolean;
  input_cost_per_million_usd: number;
  output_cost_per_million_usd: number;
  pricing_tier: "standard";
  data_location: "cloud" | "local";
  timeout_seconds: number;
}

export interface IModelCatalog {
  [provider: string]: IModelCapabilities[];
}

export type ISupportedModels = IModelCatalog;

export interface ISettings {
  provider: string | null;
  model: string | null;
  masked_key: string | null;
  supported_models: IModelCatalog;
}

export async function getSettings() {
  return (await client.get<ISettings>("/settings")).data;
}

export type ISaveSettingsPayload = ISettingsCandidate;

export async function saveSettings(
  payload: ISettingsCandidate,
  signal?: AbortSignal,
) {
  if (signal) {
    return (await client.put<ISettings>("/settings", payload, { signal })).data;
  }
  return (await client.put<ISettings>("/settings", payload)).data;
}

export interface IVerifyResult {
  ok: boolean;
  error: string | null;
}

export interface ISettingsCandidate {
  provider: string;
  model: string;
  api_key: string;
}

export async function verifySettings(
  payload: ISettingsCandidate,
  signal?: AbortSignal,
) {
  if (signal) {
    return (
      await client.post<IVerifyResult>("/settings/verify", payload, { signal })
    ).data;
  }
  return (await client.post<IVerifyResult>("/settings/verify", payload)).data;
}
