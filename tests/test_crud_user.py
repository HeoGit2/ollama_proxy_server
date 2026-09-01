import datetime

import pytest
from sqlalchemy.exc import IntegrityError

from app.core.security import verify_password
from app.crud import apikey_crud, log_crud, user_crud
from app.schema.user import UserCreate


async def test_create_user_hashes_password(db):
    user = await user_crud.create_user(db, UserCreate(username="bob", password="pw123"))

    assert user.id is not None
    assert user.hashed_password != "pw123"
    assert verify_password("pw123", user.hashed_password)
    assert user.is_admin is False


async def test_create_user_can_be_admin(db):
    user = await user_crud.create_user(
        db, UserCreate(username="root", password="pw"), is_admin=True
    )
    assert user.is_admin is True


async def test_duplicate_usernames_are_rejected(db):
    await user_crud.create_user(db, UserCreate(username="dup", password="pw"))

    with pytest.raises(IntegrityError):
        await user_crud.create_user(db, UserCreate(username="dup", password="pw"))


async def test_get_user_by_username(db, user):
    assert (await user_crud.get_user_by_username(db, "alice")).id == user.id
    assert await user_crud.get_user_by_username(db, "nobody") is None


async def test_get_user_by_id(db, user):
    assert (await user_crud.get_user_by_id(db, user.id)).username == "alice"
    assert await user_crud.get_user_by_id(db, 4242) is None


async def test_update_user_renames_and_keeps_password(db, user):
    original_hash = user.hashed_password

    updated = await user_crud.update_user(db, user.id, username="alice2")

    assert updated.username == "alice2"
    assert updated.hashed_password == original_hash


async def test_update_user_changes_password_when_provided(db, user):
    updated = await user_crud.update_user(
        db, user.id, username="alice", password="newpw"
    )

    assert verify_password("newpw", updated.hashed_password)
    assert not verify_password("s3cret", updated.hashed_password)


async def test_update_missing_user_returns_none(db):
    assert await user_crud.update_user(db, 999, username="ghost") is None


async def test_delete_user_cascades_to_api_keys(db, user, api_key):
    deleted = await user_crud.delete_user(db, user.id)

    assert deleted.id == user.id
    assert await user_crud.get_user_by_id(db, user.id) is None
    assert await apikey_crud.get_api_key_by_id(db, api_key.id) is None


async def test_delete_missing_user_returns_none(db):
    assert await user_crud.delete_user(db, 999) is None


async def test_get_users_aggregates_key_and_request_counts(db, user, api_key):
    await apikey_crud.create_api_key(db, user_id=user.id, key_name="second")
    await log_crud.create_usage_log(
        db, api_key_id=api_key.id, endpoint="/api/chat", status_code=200, server_id=None
    )
    await user_crud.create_user(db, UserCreate(username="zoe", password="pw"))

    rows = {row.username: row for row in await user_crud.get_users(db)}

    assert rows["alice"].key_count == 2
    assert rows["alice"].request_count == 1
    assert isinstance(rows["alice"].last_used, datetime.datetime)
    assert rows["zoe"].key_count == 0
    assert rows["zoe"].request_count == 0
    assert rows["zoe"].last_used is None


async def test_get_users_sorting_and_pagination(db):
    for name in ("carol", "alice", "bob"):
        await user_crud.create_user(db, UserCreate(username=name, password="pw"))

    ascending = [row.username for row in await user_crud.get_users(db)]
    descending = [
        row.username for row in await user_crud.get_users(db, sort_order="DESC")
    ]
    paginated = [row.username for row in await user_crud.get_users(db, skip=1, limit=1)]

    assert ascending == ["alice", "bob", "carol"]
    assert descending == ["carol", "bob", "alice"]
    assert paginated == ["bob"]


async def test_get_users_unknown_sort_key_falls_back_to_username(db):
    for name in ("carol", "alice"):
        await user_crud.create_user(db, UserCreate(username=name, password="pw"))

    rows = await user_crud.get_users(db, sort_by="not_a_column")
    assert [row.username for row in rows] == ["alice", "carol"]


async def test_get_users_can_sort_by_request_count(db, user, api_key):
    await log_crud.create_usage_log(
        db, api_key_id=api_key.id, endpoint="/api/chat", status_code=200, server_id=None
    )
    await user_crud.create_user(db, UserCreate(username="zoe", password="pw"))

    rows = await user_crud.get_users(db, sort_by="request_count", sort_order="desc")
    assert rows[0].username == "alice"
