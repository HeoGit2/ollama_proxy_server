import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database.base import Base
from app.database.models import APIKey, OllamaServer, User
from app.schema.user import UserCreate


@pytest_asyncio.fixture
async def db() -> AsyncSession:
    """An isolated in-memory SQLite session with all tables created."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )
    async with session_factory() as session:
        yield session

    await engine.dispose()


@pytest_asyncio.fixture
async def user(db: AsyncSession) -> User:
    from app.crud import user_crud

    return await user_crud.create_user(
        db, UserCreate(username="alice", password="s3cret"), is_admin=False
    )


@pytest_asyncio.fixture
async def api_key(db: AsyncSession, user: User) -> APIKey:
    from app.crud import apikey_crud

    _, key = await apikey_crud.create_api_key(db, user_id=user.id, key_name="default")
    return key


@pytest.fixture
def make_server():
    """Builds a detached OllamaServer instance (no DB round-trip required)."""

    def _make(**overrides) -> OllamaServer:
        defaults = {
            "id": 1,
            "name": "srv",
            "url": "http://localhost:11434",
            "server_type": "ollama",
            "is_active": True,
        }
        defaults.update(overrides)
        return OllamaServer(**defaults)

    return _make
