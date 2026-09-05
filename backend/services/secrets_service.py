"""
Symmetric encryption for user-supplied API keys.

Keys are encrypted with Fernet; the encryption key is derived from
``JWT_SECRET`` (SHA-256, urlsafe base64), so no extra secret is stored or
configured. The derivation is lazy: importing this module without
``JWT_SECRET`` is fine, only an actual encrypt/decrypt call fails.
"""

import base64
import hashlib
import os

from cryptography.fernet import Fernet, InvalidToken


class SecretsError(Exception):
    """SecretsError."""

    pass


def _fernet() -> Fernet:
    secret = os.getenv("JWT_SECRET")
    if not secret:
        raise SecretsError(
            "JWT_SECRET is required to encrypt stored API keys; set it in backend/.env"
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
            "different JWT_SECRET?"
        )
