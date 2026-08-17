import httpx
import pytest
from fastapi import FastAPI

from app.api.v1.routes import health


@pytest.fixture()
def client() -> httpx.AsyncClient:
    app = FastAPI()
    app.include_router(health.router)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


async def test_health_check_reports_ok(client: httpx.AsyncClient) -> None:
    async with client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
