# app/crud/log_crud.py
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func, select, text, Date # <-- Import Date
from sqlalchemy.sql import Select
from app.database.models import UsageLog, APIKey, User, OllamaServer
import datetime
from typing import Optional

# Requests are aggregated as a count of usage log rows in every statistic below.
_REQUEST_COUNT = func.count(UsageLog.id)


def _date_column():
    """
    Truncates the request timestamp to a day.

    --- CRITICAL FIX: Cast the date function output to a Date type ---
    This ensures that we get a date object back, not just a string,
    which is required for the strftime formatting in the admin route.
    """
    return func.date(UsageLog.request_timestamp, type_=Date).label("date")


def _hour_column():
    # This uses strftime which is specific to SQLite.
    # For PostgreSQL, you would use: func.extract('hour', UsageLog.request_timestamp)
    return func.strftime('%H', UsageLog.request_timestamp).label("hour")


def _scope_to_user(stmt: Select, user_id: Optional[int]) -> Select:
    """Restricts a usage log statement to the API keys owned by a single user."""
    if user_id is None:
        return stmt
    return stmt.join(APIKey, UsageLog.api_key_id == APIKey.id).filter(APIKey.user_id == user_id)


def _fill_missing_hours(rows) -> list[dict]:
    """Expands hourly aggregates into all 24 hours of the day, defaulting to zero requests."""
    stats_dict = {row.hour: row.request_count for row in rows}
    return [{"hour": f"{h:02d}:00", "request_count": stats_dict.get(f"{h:02d}", 0)} for h in range(24)]


async def create_usage_log(
    db: AsyncSession, *, api_key_id: int, endpoint: str, status_code: int, server_id: int | None, model: str | None = None
) -> UsageLog:
    db_log = UsageLog(
        api_key_id=api_key_id,
        endpoint=endpoint,
        status_code=status_code,
        server_id=server_id,
        model=model
    )
    db.add(db_log)
    await db.commit()
    await db.refresh(db_log)
    return db_log

async def get_usage_statistics(db: AsyncSession, sort_by: str = "request_count", sort_order: str = "desc"):
    """
    Returns aggregated usage statistics for all API keys, with sorting.
    """
    sort_column_map = {
        "username": User.username,
        "key_name": APIKey.key_name,
        "key_prefix": APIKey.key_prefix,
        "request_count": _REQUEST_COUNT,
    }

    # Default to request_count if an invalid column is provided for safety
    sort_column = sort_column_map.get(sort_by, _REQUEST_COUNT)

    # Determine sort order
    if sort_order.lower() == "asc":
        order_modifier = sort_column.asc()
    else:
        order_modifier = sort_column.desc()

    stmt = (
        select(
            User.username,
            APIKey.key_name,
            APIKey.key_prefix,
            APIKey.is_revoked,
            _REQUEST_COUNT.label("request_count"),
        )
        .select_from(APIKey)
        .join(User, APIKey.user_id == User.id)
        .outerjoin(UsageLog, APIKey.id == UsageLog.api_key_id)
        .group_by(User.username, APIKey.key_name, APIKey.key_prefix, APIKey.is_revoked)
        .order_by(order_modifier)
    )
    result = await db.execute(stmt)
    return result.all()

# --- STATISTICS FUNCTIONS ---
# Each of these accepts an optional user_id to scope the statistic to a single user.

async def get_daily_usage_stats(db: AsyncSession, days: int = 30, user_id: Optional[int] = None):
    """Returns total requests per day for the last N days."""
    start_date = datetime.datetime.utcnow() - datetime.timedelta(days=days)
    date_column = _date_column()

    stmt = _scope_to_user(
        select(date_column, _REQUEST_COUNT.label("request_count")),
        user_id,
    ).filter(UsageLog.request_timestamp >= start_date).group_by(date_column).order_by(date_column.asc())

    result = await db.execute(stmt)
    return result.all()

async def get_hourly_usage_stats(db: AsyncSession, user_id: Optional[int] = None):
    """Returns total requests aggregated by the hour of the day (UTC)."""
    stmt = _scope_to_user(
        select(_hour_column(), _REQUEST_COUNT.label("request_count")),
        user_id,
    ).group_by("hour").order_by("hour")

    result = await db.execute(stmt)
    return _fill_missing_hours(result.all())

async def get_server_load_stats(db: AsyncSession, user_id: Optional[int] = None):
    """Returns total requests per backend server."""
    columns = (OllamaServer.name.label("server_name"), _REQUEST_COUNT.label("request_count"))

    if user_id is None:
        stmt = (
            select(*columns)
            .select_from(OllamaServer)
            .outerjoin(UsageLog, OllamaServer.id == UsageLog.server_id)
        )
    else:
        # Scoped per user the join has to start from the logs, so that only the
        # servers used by that user's keys are reported.
        stmt = _scope_to_user(
            select(*columns).select_from(UsageLog),
            user_id,
        ).outerjoin(OllamaServer, UsageLog.server_id == OllamaServer.id)

    stmt = stmt.group_by(OllamaServer.name).order_by(_REQUEST_COUNT.desc())
    result = await db.execute(stmt)
    return result.all()

async def get_model_usage_stats(db: AsyncSession, user_id: Optional[int] = None):
    """Returns total requests per model."""
    stmt = _scope_to_user(
        select(UsageLog.model.label("model_name"), _REQUEST_COUNT.label("request_count")),
        user_id,
    ).filter(UsageLog.model.isnot(None)).group_by(UsageLog.model).order_by(_REQUEST_COUNT.desc())

    result = await db.execute(stmt)
    return result.all()

# --- USER-SPECIFIC STATISTICS FUNCTIONS ---

async def get_daily_usage_stats_for_user(db: AsyncSession, user_id: int, days: int = 30):
    """Returns total requests per day for the last N days for a specific user."""
    return await get_daily_usage_stats(db, days=days, user_id=user_id)

async def get_hourly_usage_stats_for_user(db: AsyncSession, user_id: int):
    """Returns total requests aggregated by the hour for a specific user."""
    return await get_hourly_usage_stats(db, user_id=user_id)

async def get_server_load_stats_for_user(db: AsyncSession, user_id: int):
    """Returns total requests per backend server for a specific user."""
    return await get_server_load_stats(db, user_id=user_id)

async def get_model_usage_stats_for_user(db: AsyncSession, user_id: int):
    """Returns total requests per model for a specific user."""
    return await get_model_usage_stats(db, user_id=user_id)
