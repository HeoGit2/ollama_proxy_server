import base64

import pytest
from cryptography.fernet import Fernet

from app.core import encryption


def test_encrypt_decrypt_roundtrip():
    token = encryption.encrypt_data("my-api-key")
    assert token != "my-api-key"
    assert encryption.decrypt_data(token) == "my-api-key"


def test_encrypt_data_is_non_deterministic():
    assert encryption.encrypt_data("dup") != encryption.encrypt_data("dup")


def test_empty_values_pass_through():
    assert encryption.encrypt_data("") == ""
    assert encryption.decrypt_data("") == ""


def test_decrypt_invalid_token_returns_empty_string():
    assert encryption.decrypt_data("not-a-fernet-token") == ""


def test_decrypt_with_foreign_key_returns_empty_string():
    foreign_token = Fernet(Fernet.generate_key()).encrypt(b"secret").decode()
    assert encryption.decrypt_data(foreign_token) == ""


def test_unicode_roundtrip():
    assert (
        encryption.decrypt_data(encryption.encrypt_data("clé-privée-🔐"))
        == "clé-privée-🔐"
    )


def test_key_derivation_matches_secret_key():
    from app.core.config import settings

    expected = base64.urlsafe_b64encode(settings.SECRET_KEY.encode()[:32])
    assert Fernet(expected).decrypt(encryption.encrypt_data("abc").encode()) == b"abc"


def test_operations_raise_when_service_uninitialised(monkeypatch):
    monkeypatch.setattr(encryption, "fernet", None)

    with pytest.raises(RuntimeError):
        encryption.encrypt_data("abc")
    with pytest.raises(RuntimeError):
        encryption.decrypt_data("abc")
