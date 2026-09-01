from passlib.hash import sha256_crypt

from app.core import security


def test_password_hash_roundtrip():
    hashed = security.get_password_hash("correct horse")
    assert hashed != "correct horse"
    assert security.verify_password("correct horse", hashed)
    assert not security.verify_password("wrong horse", hashed)


def test_password_hash_is_salted():
    assert security.get_password_hash("same") != security.get_password_hash("same")


def test_verify_password_accepts_legacy_sha256_crypt():
    legacy = sha256_crypt.hash("legacy-pass")
    assert security.verify_password("legacy-pass", legacy)
    assert not security.verify_password("other", legacy)


def test_new_password_hashes_use_bcrypt():
    assert security.get_password_hash("x").startswith("$2")


def test_api_key_hash_roundtrip():
    hashed = security.get_api_key_hash("secret-part")
    assert security.verify_api_key("secret-part", hashed)
    assert not security.verify_api_key("secret-part2", hashed)


def test_generate_secure_api_key_structure():
    plain, prefix, secret = security.generate_secure_api_key()

    assert plain == f"{prefix}_{secret}"
    assert prefix.startswith("op_")
    assert secret
    assert security.verify_api_key(secret, security.get_api_key_hash(secret))


def test_generate_secure_api_key_is_unique():
    keys = {security.generate_secure_api_key()[0] for _ in range(5)}
    assert len(keys) == 5
