import client from "./client";

export interface ISupportedModels {
  [provider: string]: string[];
}

export interface ISettings {
  provider: string | null;
  model: string | null;
  masked_key: string | null;
  supported_models: ISupportedModels;
}

export async function getSettings() {
  return (await client.get<ISettings>("/settings")).data;
}

export interface ISaveSettingsPayload {
  provider: string;
  model: string;
  api_key: string;
}

export async function saveSettings(payload: ISaveSettingsPayload) {
  return (await client.put<ISettings>("/settings", payload)).data;
}

export interface IVerifyResult {
  ok: boolean;
  error: string | null;
}

export async function verifySettings() {
  return (await client.post<IVerifyResult>("/settings/verify")).data;
}
