"""
Symmetric encryption for provider API keys.

Keys are encrypted with Fernet; the encryption key is derived from
``APP_SECRET`` via HKDF-SHA256 with the versioned info string
``papermind/api-key/v1`` (urlsafe base64). The derivation is lazy:
importing this module without ``APP_SECRET`` is fine, only an actual
encrypt/decrypt call fails.

Rows written before the HKDF upgrade (plain SHA-256 derivation) are
recognized on read: they raise :class:`SecretsResaveRequiredError`
instead of returning plaintext, so routes can ask the operator to
re-save the key in Settings.
"""

import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from settings import get_settings

API_KEY_INFO = b"papermind/api-key/v1"

RESAVE_MESSAGE = (
    "Stored API key was encrypted with an older method. "
    "Please re-save your key in Settings."
)


class SecretsError(Exception):
    """SecretsError."""

    pass


class SecretsResaveRequiredError(SecretsError):
    """Old-derivation row; the operator must re-save the key."""

    pass


def _require_secret() -> str:
    secret = get_settings().auth.app_secret
    if not secret:
        raise SecretsError(
            "APP_SECRET is required to encrypt stored API keys; set it in backend/.env"
        )
    return secret


def _fernet() -> Fernet:
    secret = _require_secret()
    derived = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b"",
        info=API_KEY_INFO,
    ).derive(secret.encode())
    return Fernet(base64.urlsafe_b64encode(derived))


def _legacy_fernet() -> Fernet:
    secret = _require_secret()
    digest = hashlib.sha256(secret.encode()).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def encrypt_api_key(plaintext: str) -> str:
    """Do encrypt api key."""
    return _fernet().encrypt(plaintext.encode()).decode()


def decrypt_api_key(ciphertext: str) -> str:
    """Do decrypt api key."""
    try:
        return _fernet().decrypt(ciphertext.encode()).decode()
    except InvalidToken:
        pass
    try:
        _legacy_fernet().decrypt(ciphertext.encode())
    except InvalidToken:
        raise SecretsError(
            "Stored API key could not be decrypted; was it encrypted with a "
            "different APP_SECRET?"
        )
    raise SecretsResaveRequiredError(RESAVE_MESSAGE)
