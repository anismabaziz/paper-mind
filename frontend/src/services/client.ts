import axios from "axios";
import { getToken } from "./auth";

const client = axios.create({
  baseURL: import.meta.env.VITE_API_URL,
});

// Demo mode needs no token; outside it every protected route expects a
// Bearer token from /auth/login.
client.interceptors.request.use((config) => {
  const token = getToken();
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

export default client;
