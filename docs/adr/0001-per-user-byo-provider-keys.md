# Per-user BYO provider keys, encrypted at rest, no cross-provider fallback

> **Superseded by [0005-single-instance-no-auth.md](0005-single-instance-no-auth.md)** — the multi-user `users` + `user_settings` model and `JWT_SECRET`/`DEMO_MODE` machinery were removed. The app is now a single-instance workspace with a global `app_settings` row encrypted via `APP_SECRET`. No fallback and masked-key behavior described below is retained.

Each user brings their own chat provider (Google or Groq), model, and API key,
configured in the app's Settings dialog and stored in a per-user `user_settings`
row with the key encrypted via Fernet (key derived from `JWT_SECRET`). The API
never returns the plaintext key — only a masked suffix. Chat always runs on the
requester's own credentials, including demo mode, which resolves to a seeded
demo user. When the primary provider fails mid-stream, the failure is surfaced
to the user; there is no fallback to another provider, because with
user-supplied keys a silent fallback would bill someone else's account without
their consent. Provider, model, and API-key environment variables are gone:
"no user has configured settings yet" is a valid state, and chat without saved
settings returns a clear error pointing at Settings instead of an env default.
