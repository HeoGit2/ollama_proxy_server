import secrets
from typing import MutableMapping

from passlib.context import CryptContext

CSRF_SESSION_KEY = "csrf_token"

# Hashing context for user passwords
# --- FIX: Add "sha256_crypt" as a legacy scheme.
# This allows the system to verify passwords that were hashed with this older method.
# The 'deprecated="auto"' setting ensures that bcrypt is used for all new hashes.
pwd_context = CryptContext(
    schemes=["bcrypt", "sha256_crypt"], 
    deprecated="auto"
)

# Hashing context for API keys. Using a different scheme for domain separation.
api_key_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    # No changes needed here. Passlib automatically handles the verification
    # against all schemes listed in the context.
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def verify_api_key(plain_key: str, hashed_key: str) -> bool:
    """Verifies a plain API key against its hash."""
    return api_key_context.verify(plain_key, hashed_key)


def get_api_key_hash(api_key: str) -> str:
    """Hashes an API key."""
    return api_key_context.hash(api_key)


def generate_secure_api_key() -> (str, str, str):
    """Generates a new secure API key with a prefix and a secret part."""
    prefix = f"op_{secrets.token_urlsafe(8)}"
    secret = secrets.token_urlsafe(32)
    return f"{prefix}_{secret}", prefix, secret


def get_or_create_csrf_token(session: MutableMapping) -> str:
    """Returns the session's CSRF token, creating one if it does not exist yet."""
    if CSRF_SESSION_KEY not in session:
        session[CSRF_SESSION_KEY] = secrets.token_hex(32)
    return session[CSRF_SESSION_KEY]


def csrf_token_is_valid(session: MutableMapping, submitted_token: str) -> bool:
    """Constant-time comparison of a submitted CSRF token with the session token."""
    stored_token = get_or_create_csrf_token(session)
    return bool(stored_token) and secrets.compare_digest(submitted_token, stored_token)
