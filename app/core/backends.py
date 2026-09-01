# app/core/backends.py
"""Shared helpers for talking to Ollama/vLLM backend servers."""
import json
import logging
from typing import Dict

import httpx

from app.core.encryption import decrypt_data
from app.database.models import OllamaServer

logger = logging.getLogger(__name__)


def auth_headers(server: OllamaServer) -> Dict[str, str]:
    """Bearer authorization headers for a backend server, if it has an API key."""
    headers: Dict[str, str] = {}
    if server.encrypted_api_key:
        api_key = decrypt_data(server.encrypted_api_key)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        else:
            logger.error(
                f"Server '{server.name}' has a stored API key that could not be decrypted; "
                "the request will be sent unauthenticated. Re-enter the key or restore SECRET_KEY."
            )
    return headers


def backend_url(server: OllamaServer, path: str = "") -> str:
    """Joins a path onto a server URL, normalizing the slashes between them."""
    base_url = server.url.rstrip("/")
    if not path:
        return base_url
    return f"{base_url}/{path.lstrip('/')}"


def response_error_detail(response: httpx.Response) -> str:
    """Extracts the 'error' field of a backend error response, falling back to the raw body."""
    try:
        return response.json().get("error", response.text)
    except (json.JSONDecodeError, ValueError):
        return response.text
