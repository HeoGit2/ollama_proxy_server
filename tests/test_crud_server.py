import datetime
import json

import httpx
import pytest

from app.core.encryption import decrypt_data
from app.crud import server_crud
from app.database.models import OllamaServer
from app.schema.server import ServerCreate, ServerUpdate

_REAL_ASYNC_CLIENT = httpx.AsyncClient


def _mock_client(handler) -> httpx.AsyncClient:
    """A client whose requests are answered by ``handler`` instead of the network."""
    return _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(handler))


def _responder(response: httpx.Response, recorder: list | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if recorder is not None:
            recorder.append(request)
        return response

    return handler


class TestAuthHeaders:
    def test_no_headers_without_api_key(self, make_server):
        assert server_crud._get_auth_headers(make_server()) == {}

    def test_bearer_header_from_encrypted_key(self, make_server):
        from app.core.encryption import encrypt_data

        server = make_server(encrypted_api_key=encrypt_data("sk-123"))
        assert server_crud._get_auth_headers(server) == {
            "Authorization": "Bearer sk-123"
        }

    def test_undecryptable_key_yields_no_headers(self, make_server):
        assert (
            server_crud._get_auth_headers(make_server(encrypted_api_key="garbage"))
            == {}
        )


class TestServerCrud:
    async def test_create_server_encrypts_api_key(self, db):
        server = await server_crud.create_server(
            db,
            ServerCreate(
                name="local", url="http://localhost:11434", api_key="sk-secret"
            ),
        )

        assert server.id is not None
        assert server.url == "http://localhost:11434/"
        assert server.encrypted_api_key != "sk-secret"
        assert decrypt_data(server.encrypted_api_key) == "sk-secret"
        assert server.has_api_key is True

    async def test_create_server_without_api_key(self, db):
        server = await server_crud.create_server(
            db, ServerCreate(name="local", url="http://localhost:11434")
        )

        assert server.encrypted_api_key is None
        assert server.has_api_key is False
        assert server.server_type == "ollama"

    async def test_lookup_helpers(self, db):
        server = await server_crud.create_server(
            db, ServerCreate(name="local", url="http://localhost:11434")
        )

        assert (await server_crud.get_server_by_id(db, server.id)).name == "local"
        assert (await server_crud.get_server_by_name(db, "local")).id == server.id
        assert (await server_crud.get_server_by_url(db, server.url)).id == server.id
        assert await server_crud.get_server_by_id(db, 999) is None
        assert await server_crud.get_server_by_name(db, "ghost") is None
        assert await server_crud.get_server_by_url(db, "http://ghost") is None

    async def test_get_servers_pagination(self, db):
        for index in range(3):
            db.add(
                OllamaServer(
                    name=f"srv-{index}",
                    url=f"http://localhost:1000{index}",
                    created_at=datetime.datetime(2024, 1, index + 1),
                )
            )
        await db.commit()

        newest_first = [server.name for server in await server_crud.get_servers(db)]
        limited = [server.name for server in await server_crud.get_servers(db, limit=1)]
        skipped = [server.name for server in await server_crud.get_servers(db, skip=2)]

        assert newest_first == ["srv-2", "srv-1", "srv-0"]
        assert limited == ["srv-2"]
        assert skipped == ["srv-0"]

    async def test_update_server_fields(self, db):
        server = await server_crud.create_server(
            db, ServerCreate(name="local", url="http://localhost:11434")
        )

        updated = await server_crud.update_server(
            db,
            server.id,
            ServerUpdate(name="renamed", url="http://remote:11434", server_type="vllm"),
        )

        assert updated.name == "renamed"
        assert updated.url == "http://remote:11434/"
        assert isinstance(updated.url, str)
        assert updated.server_type == "vllm"

    async def test_update_server_sets_and_clears_api_key(self, db):
        server = await server_crud.create_server(
            db, ServerCreate(name="local", url="http://localhost:11434", api_key="old")
        )

        rotated = await server_crud.update_server(
            db, server.id, ServerUpdate(api_key="new")
        )
        assert decrypt_data(rotated.encrypted_api_key) == "new"

        cleared = await server_crud.update_server(
            db, server.id, ServerUpdate(api_key="")
        )
        assert cleared.encrypted_api_key is None

    async def test_update_server_leaves_untouched_fields_alone(self, db):
        server = await server_crud.create_server(
            db, ServerCreate(name="local", url="http://localhost:11434", api_key="keep")
        )

        updated = await server_crud.update_server(
            db, server.id, ServerUpdate(name="renamed")
        )

        assert updated.url == "http://localhost:11434/"
        assert decrypt_data(updated.encrypted_api_key) == "keep"

    async def test_update_missing_server_returns_none(self, db):
        assert await server_crud.update_server(db, 999, ServerUpdate(name="x")) is None

    async def test_delete_server(self, db):
        server = await server_crud.create_server(
            db, ServerCreate(name="local", url="http://localhost:11434")
        )

        deleted = await server_crud.delete_server(db, server.id)

        assert deleted.id == server.id
        assert await server_crud.get_server_by_id(db, server.id) is None
        assert await server_crud.delete_server(db, 999) is None


class TestFetchAndUpdateModels:
    async def test_fetches_ollama_tags(self, db, monkeypatch):
        server = await server_crud.create_server(
            db, ServerCreate(name="local", url="http://localhost:11434")
        )
        requests = []
        response = httpx.Response(
            200, json={"models": [{"name": "llama3:8b", "size": 42}]}
        )
        monkeypatch.setattr(
            server_crud.httpx,
            "AsyncClient",
            lambda **kwargs: _mock_client(_responder(response, requests)),
        )

        result = await server_crud.fetch_and_update_models(db, server.id)

        assert result["success"] is True
        assert result["models"] == [{"name": "llama3:8b", "size": 42}]
        assert str(requests[0].url) == "http://localhost:11434/api/tags"
        refreshed = await server_crud.get_server_by_id(db, server.id)
        assert refreshed.available_models == result["models"]
        assert refreshed.last_error is None
        assert refreshed.models_last_updated is not None

    async def test_fetches_and_normalises_vllm_models(self, db, monkeypatch):
        server = await server_crud.create_server(
            db,
            ServerCreate(name="vllm", url="http://localhost:8000", server_type="vllm"),
        )
        requests = []
        response = httpx.Response(
            200,
            json={
                "data": [{"id": "meta-llama/Llama-3-8b", "created": 0}, {"no_id": True}]
            },
        )
        monkeypatch.setattr(
            server_crud.httpx,
            "AsyncClient",
            lambda **kwargs: _mock_client(_responder(response, requests)),
        )

        result = await server_crud.fetch_and_update_models(db, server.id)

        assert str(requests[0].url) == "http://localhost:8000/v1/models"
        assert len(result["models"]) == 1
        model = result["models"][0]
        assert model["name"] == "meta-llama/Llama-3-8b"
        assert model["digest"] == "meta-llama/Llama-3-8b"
        assert model["details"]["format"] == "vllm"
        assert model["details"]["family"] == "meta"

    async def test_records_http_errors_on_the_server(self, db, monkeypatch):
        server = await server_crud.create_server(
            db, ServerCreate(name="local", url="http://localhost:11434")
        )
        monkeypatch.setattr(
            server_crud.httpx,
            "AsyncClient",
            lambda **kwargs: _mock_client(_responder(httpx.Response(500))),
        )

        result = await server_crud.fetch_and_update_models(db, server.id)

        assert result["success"] is False
        assert "HTTP error" in result["error"]
        refreshed = await server_crud.get_server_by_id(db, server.id)
        assert refreshed.available_models is None
        assert "HTTP error" in refreshed.last_error

    async def test_records_unexpected_errors(self, db, monkeypatch):
        server = await server_crud.create_server(
            db, ServerCreate(name="local", url="http://localhost:11434")
        )

        def boom(request):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(
            server_crud.httpx, "AsyncClient", lambda **kwargs: _mock_client(boom)
        )

        result = await server_crud.fetch_and_update_models(db, server.id)

        assert result["success"] is False
        assert "Unexpected error" in result["error"]

    async def test_missing_server_is_reported(self, db):
        result = await server_crud.fetch_and_update_models(db, 999)
        assert result == {"success": False, "error": "Server not found", "models": []}


class TestModelManagementOnServer:
    @pytest.mark.parametrize(
        "operation",
        [
            server_crud.pull_model_on_server,
            server_crud.delete_model_on_server,
            server_crud.load_model_on_server,
            server_crud.unload_model_on_server,
        ],
    )
    async def test_operations_are_rejected_for_vllm(self, operation, make_server):
        async with _mock_client(_responder(httpx.Response(200))) as client:
            result = await operation(client, make_server(server_type="vllm"), "llama3")

        assert result["success"] is False
        assert "vLLM" in result["message"]

    async def test_pull_model_success(self, make_server):
        requests = []
        response = httpx.Response(200, text=json.dumps({"status": "success"}))

        async with _mock_client(_responder(response, requests)) as client:
            result = await server_crud.pull_model_on_server(
                client, make_server(), "llama3"
            )

        assert result["success"] is True
        assert str(requests[0].url) == "http://localhost:11434/api/pull"
        assert json.loads(requests[0].content) == {"name": "llama3", "stream": False}

    async def test_pull_model_ignores_non_json_progress_chunks(self, make_server):
        async with _mock_client(
            _responder(httpx.Response(200, text="not json"))
        ) as client:
            result = await server_crud.pull_model_on_server(
                client, make_server(), "llama3"
            )

        assert result["success"] is True

    async def test_pull_model_reports_http_status_errors(self, make_server):
        async with _mock_client(_responder(httpx.Response(404, text="{}"))) as client:
            result = await server_crud.pull_model_on_server(
                client, make_server(), "llama3"
            )

        assert result["success"] is False
        assert "status 404" in result["message"]

    async def test_pull_model_reports_unexpected_errors(self, make_server):
        def boom(request):
            raise RuntimeError("kaboom")

        async with _mock_client(boom) as client:
            result = await server_crud.pull_model_on_server(
                client, make_server(), "llama3"
            )

        assert result["success"] is False
        assert "kaboom" in result["message"]

    async def test_delete_model_success(self, make_server):
        requests = []

        async with _mock_client(_responder(httpx.Response(200), requests)) as client:
            result = await server_crud.delete_model_on_server(
                client, make_server(), "llama3"
            )

        assert result["success"] is True
        assert requests[0].method == "DELETE"
        assert json.loads(requests[0].content) == {"name": "llama3"}

    async def test_delete_missing_model_is_treated_as_success(self, make_server):
        async with _mock_client(_responder(httpx.Response(404))) as client:
            result = await server_crud.delete_model_on_server(
                client, make_server(), "llama3"
            )

        assert result["success"] is True
        assert "not found" in result["message"]

    async def test_delete_model_other_errors_fail(self, make_server):
        async with _mock_client(_responder(httpx.Response(500))) as client:
            result = await server_crud.delete_model_on_server(
                client, make_server(), "llama3"
            )

        assert result["success"] is False
        assert "status 500" in result["message"]

    async def test_load_model_success(self, make_server):
        requests = []

        async with _mock_client(_responder(httpx.Response(200), requests)) as client:
            result = await server_crud.load_model_on_server(
                client, make_server(), "llama3"
            )

        assert result["success"] is True
        assert str(requests[0].url) == "http://localhost:11434/api/generate"
        assert json.loads(requests[0].content)["stream"] is False

    async def test_load_model_surfaces_server_error_detail(self, make_server):
        response = httpx.Response(400, json={"error": "no such model"})

        async with _mock_client(_responder(response)) as client:
            result = await server_crud.load_model_on_server(
                client, make_server(), "llama3"
            )

        assert result["success"] is False
        assert "no such model" in result["message"]

    async def test_load_model_falls_back_to_response_text(self, make_server):
        async with _mock_client(
            _responder(httpx.Response(400, text="plain failure"))
        ) as client:
            result = await server_crud.load_model_on_server(
                client, make_server(), "llama3"
            )

        assert "plain failure" in result["message"]

    async def test_unload_model_sets_zero_keep_alive(self, make_server):
        requests = []

        async with _mock_client(_responder(httpx.Response(200), requests)) as client:
            result = await server_crud.unload_model_on_server(
                client, make_server(), "llama3"
            )

        assert result["success"] is True
        assert json.loads(requests[0].content)["keep_alive"] == "0s"

    async def test_unload_model_treats_404_as_not_loaded(self, make_server):
        async with _mock_client(_responder(httpx.Response(404))) as client:
            result = await server_crud.unload_model_on_server(
                client, make_server(), "llama3"
            )

        assert result["success"] is True
        assert "not loaded" in result["message"]

    async def test_unload_model_reports_errors(self, make_server):
        async with _mock_client(_responder(httpx.Response(500, text="boom"))) as client:
            result = await server_crud.unload_model_on_server(
                client, make_server(), "llama3"
            )

        assert result["success"] is False
        assert "status 500" in result["message"]


class TestModelDiscovery:
    @pytest.mark.parametrize(
        "model_name, expected",
        [("nomic-embed-text", True), ("EMBEDDING-model", True), ("llama3", False)],
    )
    def test_is_embedding_model(self, model_name, expected):
        assert server_crud.is_embedding_model(model_name) is expected

    async def _seed(self, db):
        db.add_all(
            [
                OllamaServer(
                    name="ollama-a",
                    url="http://a:11434",
                    is_active=True,
                    available_models=[
                        {"name": "llama3:8b"},
                        {"name": "nomic-embed-text"},
                    ],
                    created_at=datetime.datetime(2024, 1, 2),
                ),
                OllamaServer(
                    name="vllm-b",
                    url="http://b:8000",
                    server_type="vllm",
                    is_active=True,
                    available_models=[
                        {"name": "models--meta-llama--Llama-2-7b-chat-hf"}
                    ],
                    created_at=datetime.datetime(2024, 1, 1),
                ),
                OllamaServer(
                    name="inactive",
                    url="http://c:11434",
                    is_active=False,
                    available_models=[{"name": "hidden-model"}],
                    created_at=datetime.datetime(2024, 1, 3),
                ),
            ]
        )
        await db.commit()

    async def test_get_servers_with_model_matching_rules(self, db):
        await self._seed(db)

        exact = await server_crud.get_servers_with_model(db, "llama3:8b")
        prefix = await server_crud.get_servers_with_model(db, "llama3")
        vllm_substring = await server_crud.get_servers_with_model(db, "Llama-2-7b")
        hidden = await server_crud.get_servers_with_model(db, "hidden-model")

        assert [s.name for s in exact] == ["ollama-a"]
        assert [s.name for s in prefix] == ["ollama-a"]
        assert [s.name for s in vllm_substring] == ["vllm-b"]
        assert hidden == []

    async def test_get_servers_with_model_ignores_servers_without_models(self, db):
        db.add(OllamaServer(name="empty", url="http://d:11434", is_active=True))
        await db.commit()

        assert await server_crud.get_servers_with_model(db, "llama3") == []

    async def test_get_all_available_model_names_with_filters(self, db):
        await self._seed(db)

        all_names = await server_crud.get_all_available_model_names(db)
        chat = await server_crud.get_all_available_model_names(db, filter_type="chat")
        embedding = await server_crud.get_all_available_model_names(
            db, filter_type="embedding"
        )

        assert all_names == sorted(
            ["llama3:8b", "nomic-embed-text", "models--meta-llama--Llama-2-7b-chat-hf"]
        )
        assert "nomic-embed-text" not in chat
        assert embedding == ["nomic-embed-text"]

    async def test_get_all_available_model_names_parses_json_strings(self, db):
        db.add(
            OllamaServer(
                name="json-str",
                url="http://e:11434",
                is_active=True,
                available_models=json.dumps([{"name": "llama3"}]),
            )
        )
        await db.commit()

        assert await server_crud.get_all_available_model_names(db) == ["llama3"]

    async def test_get_all_available_model_names_skips_invalid_json(self, db):
        db.add(
            OllamaServer(
                name="broken",
                url="http://f:11434",
                is_active=True,
                available_models="{not json",
            )
        )
        await db.commit()

        assert await server_crud.get_all_available_model_names(db) == []

    async def test_get_all_models_grouped_by_server(self, db):
        await self._seed(db)

        grouped = await server_crud.get_all_models_grouped_by_server(db)

        assert list(grouped)[0] == "Proxy Features"
        assert grouped["Proxy Features"] == ["auto"]
        assert grouped["ollama-a"] == ["llama3:8b", "nomic-embed-text"]
        assert "inactive" not in grouped

    async def test_grouped_embedding_filter_excludes_proxy_features(self, db):
        await self._seed(db)

        grouped = await server_crud.get_all_models_grouped_by_server(
            db, filter_type="embedding"
        )

        assert "Proxy Features" not in grouped
        assert grouped == {"ollama-a": ["nomic-embed-text"]}

    async def test_grouped_chat_filter(self, db):
        await self._seed(db)

        grouped = await server_crud.get_all_models_grouped_by_server(
            db, filter_type="chat"
        )

        assert grouped["ollama-a"] == ["llama3:8b"]

    async def test_grouped_skips_invalid_json(self, db):
        db.add(
            OllamaServer(
                name="broken",
                url="http://f:11434",
                is_active=True,
                available_models="{not json",
            )
        )
        await db.commit()

        assert await server_crud.get_all_models_grouped_by_server(db) == {
            "Proxy Features": ["auto"]
        }


class TestHealthAndActiveModels:
    async def test_get_active_models_combines_ollama_ps_and_vllm(self, db):
        db.add_all(
            [
                OllamaServer(name="ollama-a", url="http://a:11434", is_active=True),
                OllamaServer(
                    name="vllm-b",
                    url="http://b:8000",
                    server_type="vllm",
                    is_active=True,
                    available_models=[{"name": "llama-2", "size": 10}],
                ),
                OllamaServer(name="inactive", url="http://c:11434", is_active=False),
            ]
        )
        await db.commit()
        response = httpx.Response(200, json={"models": [{"name": "llama3:8b"}]})

        async with _mock_client(_responder(response)) as client:
            models = await server_crud.get_active_models_all_servers(db, client)

        by_name = {model["name"]: model for model in models}
        assert by_name["llama3:8b"]["server_name"] == "ollama-a"
        assert by_name["llama-2"]["server_name"] == "vllm-b"
        assert by_name["llama-2"]["expires_at"] == "N/A (Always Active)"

    async def test_get_active_models_tolerates_unreachable_servers(self, db):
        db.add(OllamaServer(name="ollama-a", url="http://a:11434", is_active=True))
        await db.commit()

        def boom(request):
            raise httpx.ConnectError("refused")

        async with _mock_client(boom) as client:
            assert await server_crud.get_active_models_all_servers(db, client) == []

    async def test_check_server_health_online(self, make_server):
        requests = []

        async with _mock_client(_responder(httpx.Response(200), requests)) as client:
            result = await server_crud.check_server_health(client, make_server())

        assert result["status"] == "Online"
        assert result["reason"] is None
        assert str(requests[0].url) == "http://localhost:11434"

    async def test_check_server_health_uses_vllm_health_endpoint(self, make_server):
        requests = []

        async with _mock_client(_responder(httpx.Response(200), requests)) as client:
            await server_crud.check_server_health(
                client, make_server(server_type="vllm", url="http://localhost:8000")
            )

        assert str(requests[0].url) == "http://localhost:8000/health"

    async def test_check_server_health_offline_on_bad_status(self, make_server):
        async with _mock_client(_responder(httpx.Response(503))) as client:
            result = await server_crud.check_server_health(client, make_server())

        assert result["status"] == "Offline"
        assert result["reason"] == "Status 503"

    async def test_check_server_health_offline_on_connection_error(self, make_server):
        def boom(request):
            raise httpx.ConnectError("refused")

        async with _mock_client(boom) as client:
            result = await server_crud.check_server_health(client, make_server())

        assert result["status"] == "Offline"
        assert "refused" in result["reason"]

    async def test_check_all_servers_health(self, db):
        db.add_all(
            [
                OllamaServer(name="a", url="http://a:11434", is_active=True),
                OllamaServer(name="b", url="http://b:11434", is_active=False),
            ]
        )
        await db.commit()

        async with _mock_client(_responder(httpx.Response(200))) as client:
            results = await server_crud.check_all_servers_health(db, client)

        assert {result["name"] for result in results} == {"a", "b"}

    async def test_check_all_servers_health_without_servers(self, db):
        async with _mock_client(_responder(httpx.Response(200))) as client:
            assert await server_crud.check_all_servers_health(db, client) == []

    async def test_refresh_all_server_models_counts_successes_and_failures(
        self, db, monkeypatch
    ):
        db.add_all(
            [
                OllamaServer(name="ok", url="http://a:11434", is_active=True),
                OllamaServer(name="bad", url="http://b:11434", is_active=True),
                OllamaServer(name="off", url="http://c:11434", is_active=False),
            ]
        )
        await db.commit()

        async def fake_fetch(session, server_id):
            server = await server_crud.get_server_by_id(session, server_id)
            if server.name == "ok":
                return {"success": True, "models": [], "error": None}
            return {"success": False, "models": [], "error": "unreachable"}

        monkeypatch.setattr(server_crud, "fetch_and_update_models", fake_fetch)

        results = await server_crud.refresh_all_server_models(db)

        assert results["total"] == 2
        assert results["success"] == 1
        assert results["failed"] == 1
        assert results["errors"][0]["server_name"] == "bad"
        assert results["errors"][0]["error"] == "unreachable"
