import datetime

import pytest_asyncio

from app.crud import apikey_crud, log_crud, user_crud
from app.database.models import OllamaServer, UsageLog
from app.schema.user import UserCreate


@pytest_asyncio.fixture
async def server(db):
    srv = OllamaServer(name="srv-a", url="http://localhost:11434")
    db.add(srv)
    await db.commit()
    await db.refresh(srv)
    return srv


async def _log(
    db,
    api_key_id,
    *,
    endpoint="/api/chat",
    status_code=200,
    server_id=None,
    model=None,
    timestamp=None
):
    log = await log_crud.create_usage_log(
        db,
        api_key_id=api_key_id,
        endpoint=endpoint,
        status_code=status_code,
        server_id=server_id,
        model=model,
    )
    if timestamp is not None:
        log.request_timestamp = timestamp
        await db.commit()
        await db.refresh(log)
    return log


async def test_create_usage_log_persists_all_fields(db, api_key, server):
    log = await log_crud.create_usage_log(
        db,
        api_key_id=api_key.id,
        endpoint="/api/generate",
        status_code=201,
        server_id=server.id,
        model="llama3",
    )

    assert log.id is not None
    assert log.endpoint == "/api/generate"
    assert log.status_code == 201
    assert log.server_id == server.id
    assert log.model == "llama3"
    assert isinstance(log.request_timestamp, datetime.datetime)


async def test_create_usage_log_allows_missing_server_and_model(db, api_key):
    log = await log_crud.create_usage_log(
        db, api_key_id=api_key.id, endpoint="/api/tags", status_code=200, server_id=None
    )

    assert log.server_id is None
    assert log.model is None


async def test_get_usage_statistics_counts_per_key(db, user, api_key):
    _, unused_key = await apikey_crud.create_api_key(
        db, user_id=user.id, key_name="unused"
    )
    await _log(db, api_key.id)
    await _log(db, api_key.id)

    rows = {row.key_name: row for row in await log_crud.get_usage_statistics(db)}

    assert rows["default"].request_count == 2
    assert rows["default"].username == "alice"
    assert rows["default"].key_prefix == api_key.key_prefix
    assert rows["unused"].request_count == 0
    assert rows["unused"].is_revoked is False


async def test_get_usage_statistics_sorting(db, user, api_key):
    _, other_key = await apikey_crud.create_api_key(
        db, user_id=user.id, key_name="aaa-busy"
    )
    await _log(db, other_key.id)
    await _log(db, other_key.id)
    await _log(db, api_key.id)

    by_count_desc = await log_crud.get_usage_statistics(db)
    by_name_asc = await log_crud.get_usage_statistics(
        db, sort_by="key_name", sort_order="asc"
    )
    unknown_column = await log_crud.get_usage_statistics(db, sort_by="bogus")

    assert by_count_desc[0].key_name == "aaa-busy"
    assert [row.key_name for row in by_name_asc] == ["aaa-busy", "default"]
    assert unknown_column[0].key_name == "aaa-busy"


async def test_get_daily_usage_stats_respects_window(db, api_key):
    now = datetime.datetime.utcnow()
    await _log(db, api_key.id, timestamp=now)
    await _log(db, api_key.id, timestamp=now)
    await _log(db, api_key.id, timestamp=now - datetime.timedelta(days=40))

    recent = await log_crud.get_daily_usage_stats(db, days=30)
    everything = await log_crud.get_daily_usage_stats(db, days=365)

    assert len(recent) == 1
    assert recent[0].request_count == 2
    assert recent[0].date == now.date()
    assert sum(row.request_count for row in everything) == 3


async def test_get_hourly_usage_stats_returns_all_24_hours(db, api_key):
    stamp = datetime.datetime(2024, 5, 1, 13, 30)
    await _log(db, api_key.id, timestamp=stamp)

    stats = await log_crud.get_hourly_usage_stats(db)

    assert len(stats) == 24
    assert stats[0] == {"hour": "00:00", "request_count": 0}
    assert {"hour": "13:00", "request_count": 1} in stats


async def test_get_server_load_stats(db, api_key, server):
    idle = OllamaServer(name="srv-idle", url="http://localhost:11435")
    db.add(idle)
    await db.commit()
    await _log(db, api_key.id, server_id=server.id)

    stats = {
        row.server_name: row.request_count
        for row in await log_crud.get_server_load_stats(db)
    }

    assert stats == {"srv-a": 1, "srv-idle": 0}


async def test_get_model_usage_stats_ignores_null_models(db, api_key):
    await _log(db, api_key.id, model="llama3")
    await _log(db, api_key.id, model="llama3")
    await _log(db, api_key.id, model="mistral")
    await _log(db, api_key.id, model=None)

    stats = await log_crud.get_model_usage_stats(db)

    assert [(row.model_name, row.request_count) for row in stats] == [
        ("llama3", 2),
        ("mistral", 1),
    ]


class TestPerUserStats:
    @pytest_asyncio.fixture
    async def other_user_key(self, db):
        other = await user_crud.create_user(
            db, UserCreate(username="bob", password="pw")
        )
        _, key = await apikey_crud.create_api_key(db, user_id=other.id, key_name="bobs")
        return key

    async def test_daily_stats_are_scoped_to_the_user(
        self, db, user, api_key, other_user_key
    ):
        now = datetime.datetime.utcnow()
        await _log(db, api_key.id, timestamp=now)
        await _log(db, other_user_key.id, timestamp=now)
        await _log(db, api_key.id, timestamp=now - datetime.timedelta(days=90))

        stats = await log_crud.get_daily_usage_stats_for_user(db, user.id, days=30)

        assert len(stats) == 1
        assert stats[0].request_count == 1

    async def test_hourly_stats_are_scoped_to_the_user(
        self, db, user, api_key, other_user_key
    ):
        await _log(db, api_key.id, timestamp=datetime.datetime(2024, 5, 1, 9, 0))
        await _log(db, other_user_key.id, timestamp=datetime.datetime(2024, 5, 1, 9, 0))

        stats = await log_crud.get_hourly_usage_stats_for_user(db, user.id)

        assert len(stats) == 24
        assert {"hour": "09:00", "request_count": 1} in stats

    async def test_server_load_stats_are_scoped_to_the_user(
        self, db, user, api_key, other_user_key, server
    ):
        await _log(db, api_key.id, server_id=server.id)
        await _log(db, other_user_key.id, server_id=server.id)

        stats = await log_crud.get_server_load_stats_for_user(db, user.id)

        assert [(row.server_name, row.request_count) for row in stats] == [("srv-a", 1)]

    async def test_model_stats_are_scoped_to_the_user(
        self, db, user, api_key, other_user_key
    ):
        await _log(db, api_key.id, model="llama3")
        await _log(db, other_user_key.id, model="llama3")
        await _log(db, api_key.id, model=None)

        stats = await log_crud.get_model_usage_stats_for_user(db, user.id)

        assert [(row.model_name, row.request_count) for row in stats] == [("llama3", 1)]

    async def test_stats_for_user_without_activity_are_empty(self, db, other_user_key):
        assert await log_crud.get_daily_usage_stats_for_user(db, 999) == []
        assert await log_crud.get_model_usage_stats_for_user(db, 999) == []
        assert all(
            entry["request_count"] == 0
            for entry in await log_crud.get_hourly_usage_stats_for_user(db, 999)
        )


async def test_usage_logs_are_deleted_with_their_api_key(db, user, api_key):
    await _log(db, api_key.id)

    await user_crud.delete_user(db, user.id)

    remaining = (await db.execute(UsageLog.__table__.select())).all()
    assert remaining == []
