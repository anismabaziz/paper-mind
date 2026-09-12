"""
Symmetric encryption for provider API keys.

Keys are encrypted with Fernet; the encryption key is derived from
``APP_SECRET`` (SHA-256, urlsafe base64). ``JWT_SECRET`` is still accepted
as a legacy env alias via Settings, but the derivation always reads
``APP_SECRET``. The derivation is lazy: importing this module without
``APP_SECRET`` is fine, only an actual encrypt/decrypt call fails.
"""

import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken

from settings import get_settings


class SecretsError(Exception):
    """SecretsError."""

    pass


def _fernet() -> Fernet:
    secret = get_settings().auth.app_secret
    if not secret:
        raise SecretsError(
            "APP_SECRET is required to encrypt stored API keys; set it in backend/.env"
        )
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
        raise SecretsError(
            "Stored API key could not be decrypted; was it encrypted with a "
            "different APP_SECRET?"
        )
