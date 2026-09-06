import client from "./client";

const TOKEN_STORAGE_KEY = "papermind.auth.token";

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_STORAGE_KEY);
}

export function setToken(token: string) {
  localStorage.setItem(TOKEN_STORAGE_KEY, token);
}

export function clearToken() {
  localStorage.removeItem(TOKEN_STORAGE_KEY);
}

interface ILoginResponse {
  token: string;
}

export async function login(email: string, password: string) {
  const { data } = await client.post<ILoginResponse>("/auth/login", {
    email,
    password,
  });
  setToken(data.token);
  return data;
}
