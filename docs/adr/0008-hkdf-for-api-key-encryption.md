# HKDF-SHA256 for API key encryption with versioned info string

Chosen: provider API keys are encrypted with Fernet where the 32-byte key is derived from `APP_SECRET` via HKDF-SHA256 with a versioned info string `papermind/api-key/v1` (salt is empty, HKDF is deterministic for the same secret and info). The helper in `services/accounts/secrets_service.py` builds the Fernet key as `base64.urlsafe_b64encode(HKDF(...).derive(APP_SECRET))`. The version is part of the info so a future rotation can use `v2` without ambiguity. Changing `APP_SECRET` still invalidates previously encrypted keys by design.

Rejected: plain SHA-256 `hashlib.sha256(APP_SECRET).digest()` as a key. It produces a fixed 32 bytes but is not a KDF. It has no domain separation, no versioning, and no standard construction for key derivation. It is easy to reuse the same hash for another purpose by accident and it carries no context about what the key is for.

Also rejected: PBKDF2-HMAC-SHA256. PBKDF2 is built for low-entropy passwords where brute force is the threat, so it pays for iterations and needs a stored salt. `APP_SECRET` is expected to be a long random secret, not a password. HKDF is the right tool for deriving a key from high-entropy input. It is fast, needs no iteration tuning, and its `info` parameter gives us versioning for free. Using PBKDF2 here would add latency and salt management for no security gain.

One-time break: rows encrypted with the old SHA-256 derivation cannot be decrypted with the HKDF key. The fallback reader tries the old SHA-256 key first; if that succeeds it does not return the plaintext but raises a typed `SecretsError` with a clear re-save message: "Stored API key was encrypted with an older method. Please re-save your key in Settings." The settings and chat routes map this error to a client error (400 range) with the same message, no stack trace, and no key material. After the migration window the fallback is removed and only HKDF remains. The operator fixes this by opening Settings and saving the key again, which re-encrypts under HKDF.

Hard to reverse: the on-disk `encrypted_api_key` is opaque Fernet ciphertext. Any change to the KDF changes every stored key. The versioned info string is the seam for future changes. Without it a second migration would need to guess which derivation was used.

Trade-off: operators who upgrade from the SHA-256 build must re-save once. That cost buys a standard KDF, domain separation, and a clean upgrade path for future rotations.
