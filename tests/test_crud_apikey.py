import pytest
from sqlalchemy.exc import IntegrityError

from app.core.security import verify_api_key
from app.crud import apikey_crud


async def test_create_api_key_returns_plain_key_and_stores_hash(db, user):
    plain_key, key = await apikey_crud.create_api_key(
        db, user_id=user.id, key_name="ci"
    )

    prefix, secret = plain_key.rsplit("_", 1)

    assert prefix == key.key_prefix
    assert prefix.startswith("op_")
    assert key.hashed_key not in plain_key
    assert verify_api_key(secret, key.hashed_key)
    assert key.is_active is True
    assert key.is_revoked is False
    assert key.rate_limit_requests is None


async def test_created_key_prefix_has_no_extra_underscores(db, user):
    plain_key, key = await apikey_crud.create_api_key(
        db, user_id=user.id, key_name="ci"
    )

    assert key.key_prefix.count("_") == 1
    assert plain_key.count("_") == 2


async def test_create_api_key_stores_per_key_rate_limits(db, user):
    _, key = await apikey_crud.create_api_key(
        db,
        user_id=user.id,
        key_name="limited",
        rate_limit_requests=10,
        rate_limit_window_minutes=5,
    )

    assert (key.rate_limit_requests, key.rate_limit_window_minutes) == (10, 5)


async def test_duplicate_key_name_per_user_is_rejected(db, user):
    await apikey_crud.create_api_key(db, user_id=user.id, key_name="dup")

    with pytest.raises(IntegrityError):
        await apikey_crud.create_api_key(db, user_id=user.id, key_name="dup")


async def test_lookup_by_prefix_and_id(db, api_key):
    assert (
        await apikey_crud.get_api_key_by_prefix(db, api_key.key_prefix)
    ).id == api_key.id
    assert await apikey_crud.get_api_key_by_prefix(db, "op_missing") is None
    assert (await apikey_crud.get_api_key_by_id(db, api_key.id)).key_name == "default"
    assert await apikey_crud.get_api_key_by_id(db, 999) is None


async def test_lookup_by_name_is_scoped_to_user(db, user, api_key):
    from app.crud import user_crud
    from app.schema.user import UserCreate

    other = await user_crud.create_user(db, UserCreate(username="bob", password="pw"))

    found = await apikey_crud.get_api_key_by_name_and_user_id(
        db, key_name="default", user_id=user.id
    )
    not_found = await apikey_crud.get_api_key_by_name_and_user_id(
        db, key_name="default", user_id=other.id
    )

    assert found.id == api_key.id
    assert not_found is None


async def test_get_api_keys_for_user_returns_only_own_keys(db, user, api_key):
    from app.crud import user_crud
    from app.schema.user import UserCreate

    other = await user_crud.create_user(db, UserCreate(username="bob", password="pw"))
    await apikey_crud.create_api_key(db, user_id=other.id, key_name="theirs")
    await apikey_crud.create_api_key(db, user_id=user.id, key_name="mine")

    names = {
        key.key_name for key in await apikey_crud.get_api_keys_for_user(db, user.id)
    }

    assert names == {"default", "mine"}
    assert await apikey_crud.get_api_keys_for_user(db, 999) == []


async def test_revoke_api_key_also_deactivates(db, api_key):
    revoked = await apikey_crud.revoke_api_key(db, api_key.id)

    assert revoked.is_revoked is True
    assert revoked.is_active is False


async def test_revoke_missing_api_key_returns_none(db):
    assert await apikey_crud.revoke_api_key(db, 999) is None


async def test_toggle_api_key_active_flips_state(db, api_key):
    disabled = await apikey_crud.toggle_api_key_active(db, api_key.id)
    assert disabled.is_active is False

    reenabled = await apikey_crud.toggle_api_key_active(db, api_key.id)
    assert reenabled.is_active is True


async def test_toggle_is_refused_for_revoked_or_missing_keys(db, api_key):
    await apikey_crud.revoke_api_key(db, api_key.id)

    assert await apikey_crud.toggle_api_key_active(db, api_key.id) is None
    assert await apikey_crud.toggle_api_key_active(db, 999) is None
