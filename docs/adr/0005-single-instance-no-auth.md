# Single-instance workspace, no auth, global app settings

Chosen: PaperMind runs as a single-instance open-source workspace with no
users, no JWT, and no demo-mode gate. A sole `app_settings` row holds
`provider`, `model`, and `encrypted_api_key`; the key is encrypted with
Fernet using a key derived from `APP_SECRET` through HKDF-SHA256 and the
versioned info string `papermind/api-key/v1`. Every endpoint
(`POST /upload`, `GET /files`, `POST /response`, `GET /settings`,
`PUT /settings`, `POST /settings/verify`, etc.) is open without an
`Authorization` header. The `users` and `user_settings` tables were dropped in
migration `b7c9e2f4a1d6`. Downgrade re-creates the tables on a best-effort basis
without reconstructing encrypted keys.
`DEMO_MODE` and `JWT_SECRET` were removed from code, `.env.example`, and docs.
`APP_SECRET` is the sole application secret. Provider keys cannot be encrypted
or decrypted until it is set.

Rejected: keeping per-user BYO keys (ADR 0001). It doubled the concept count
(user + settings + JWT + demo bypass) for a single-person clone-and-run app,
required auth plumbing on every route and in the frontend (`LoginForm`,
Bearer interceptor, `localStorage` token), and made the Qdrant `pdf_name`
already stable on uuid filenames harder to reason about. Keeping a demo user
as a proxy for anonymous requests added a special case to every settings/chat
path. JWT with bcrypt added dependencies and a token lifecycle for a local app
that never leaves the laptop.

Hard to reverse: the repo contract is one `app_settings` row (`id = "app"`),
the `AppSettingsRepository.get_app_settings` / `upsert_app_settings` interface,
and the open HTTP contract (no `require_auth`, no `current_user()`). Re-adding
multi-user would mean reintroducing `users` tables, a JWT or session layer, and a
per-request user resolution on every route, plus a migration to split the
global row back into per-user rows.

Trade-off: one encrypted row is trivial to back up and fits BYO-keys
open-source distribution (clone → set `DATABASE_URL` + `QDRANT_URL` + optional
`APP_SECRET` → paste key in Settings). The cost is no multi-tenancy; any
future sharing would need a new auth/workspace layer.
