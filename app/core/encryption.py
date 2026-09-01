# app/core/encryption.py
import base64
import hashlib
import logging

from cryptography.fernet import Fernet, MultiFernet

from app.core.config import settings

logger = logging.getLogger(__name__)


def _derive_key(secret: str) -> bytes:
    """Derives a 32-byte Fernet key from the configured secret."""
    digest = hashlib.sha256(secret.encode()).digest()
    return base64.urlsafe_b64encode(digest)


def _legacy_key(secret: str) -> bytes:
    """
    Reproduces the original key derivation, which used the first 32 characters
    of SECRET_KEY verbatim. Kept only so previously encrypted values (backend
    API keys) can still be decrypted and transparently re-encrypted.
    """
    return base64.urlsafe_b64encode(secret.encode()[:32].ljust(32, b"\0"))


try:
    fernet = MultiFernet(
        [Fernet(_derive_key(settings.SECRET_KEY)), Fernet(_legacy_key(settings.SECRET_KEY))]
    )
except Exception as e:
    logger.critical(
        f"Failed to initialize Fernet for encryption: {e}. "
        "Backend API keys can neither be stored nor used until SECRET_KEY is fixed."
    )
    fernet = None

def encrypt_data(data: str) -> str:
    """Encrypts a string."""
    if not fernet:
        raise RuntimeError("Encryption service is not initialized.")
    if not data:
        return ""
    return fernet.encrypt(data.encode()).decode()

def decrypt_data(encrypted_data: str) -> str:
    """Decrypts a string."""
    if not fernet:
        raise RuntimeError("Encryption service is not initialized.")
    if not encrypted_data:
        return ""
    try:
        return fernet.decrypt(encrypted_data.encode()).decode()
    except Exception:
        # If decryption fails (e.g., key changed, data corrupted), return empty
        logger.warning(
            "Failed to decrypt data. Key may have changed or data is invalid.", exc_info=True
        )
        return ""
