"""
Vendor-neutral authentication: bcrypt password hashing and JWT issuing.

The ``require_auth`` decorator protects endpoints. When demo mode is on it
lets every request through so a local run needs zero clicks; the decorator
itself stays in place either way.
"""

import datetime
import os
import secrets
from functools import wraps

import bcrypt
import jwt
from flask import g, jsonify, request

TOKEN_TTL = datetime.timedelta(hours=24)


_random_secret = None


def _secret():
    # JWT_SECRET is optional so a demo run boots without one; a per-process
    # random fallback is fine there since tokens need not survive restarts.
    global _random_secret
    env_secret = os.getenv("JWT_SECRET")
    if env_secret:
        return env_secret
    if _random_secret is None:
        _random_secret = secrets.token_hex(32)
    return _random_secret


def is_demo_mode() -> bool:
    """Do is demo mode."""
    return os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes")


def hash_password(password: str) -> str:
    """Do hash password."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    """Do verify password."""
    return bcrypt.checkpw(password.encode(), password_hash.encode())


def issue_token(email: str) -> str:
    """Do issue token."""
    now = datetime.datetime.now(datetime.timezone.utc)
    payload = {"sub": email, "iat": now, "exp": now + TOKEN_TTL}
    return jwt.encode(payload, _secret(), algorithm="HS256")


class AuthError(Exception):
    """AuthError."""

    pass


def verify_token(token: str) -> str:
    """Return the token's subject email, or raise AuthError."""
    try:
        payload = jwt.decode(token, _secret(), algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise AuthError("Token has expired")
    except jwt.InvalidTokenError:
        raise AuthError("Invalid token")
    return payload.get("sub")


def require_auth(fn):
    """Do require auth."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        """Do wrapper."""
        if is_demo_mode():
            return fn(*args, **kwargs)

        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return jsonify({"error": "Missing bearer token"}), 401
        try:
            g.user_email = verify_token(header.removeprefix("Bearer "))
        except AuthError as e:
            return jsonify({"error": str(e)}), 401
        return fn(*args, **kwargs)

    return wrapper
