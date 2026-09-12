"""
Authentication removed.

PaperMind now runs as a single-instance open-source app with no JWT or
demo-mode gate. Kept for migration compatibility: ``hash_password`` and
``verify_password`` remain so the Alembic revision can still import them.
All JWT helpers have been removed — every endpoint is open without a token.
"""

import bcrypt


def hash_password(password: str) -> str:
    """Hash a password with bcrypt (legacy, for migrations)."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its bcrypt hash (legacy, for migrations)."""
    return bcrypt.checkpw(password.encode(), password_hash.encode())
