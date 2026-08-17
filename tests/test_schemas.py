import datetime

import pytest
from pydantic import ValidationError

from app.schema.apikey import APIKey as APIKeySchema
from app.schema.server import ServerCreate, ServerUpdate
from app.schema.settings import AppSettingsModel
from app.schema.user import User as UserSchema
from app.schema.user import UserCreate


class TestAppSettingsModel:
    def test_defaults(self):
        settings = AppSettingsModel()

        assert settings.branding_title == "Ollama Proxy"
        assert settings.branding_logo_url is None
        assert settings.ui_style == "dark-glass"
        assert settings.rate_limit_requests == 100
        assert settings.max_retries == 5
        assert settings.retry_total_timeout_seconds == 2.0
        assert settings.retry_base_delay_ms == 50
        assert settings.blocked_ollama_endpoints == "pull,delete,create,copy,push"

    def test_available_themes_expose_shade_palette(self):
        themes = AppSettingsModel().available_themes

        assert "indigo" in themes
        for palette in themes.values():
            assert set(palette) == {"500", "600", "700", "800"}
            assert all(shade.startswith("#") for shade in palette.values())

    @pytest.mark.parametrize(
        "field", ["branding_logo_url", "ssl_keyfile", "ssl_certfile"]
    )
    def test_empty_strings_are_normalised_to_none(self, field):
        assert getattr(AppSettingsModel(**{field: ""}), field) is None

    @pytest.mark.parametrize(
        "field, value",
        [
            ("max_retries", -1),
            ("max_retries", 21),
            ("retry_total_timeout_seconds", 0.0),
            ("retry_total_timeout_seconds", 30.1),
            ("retry_base_delay_ms", 9),
            ("retry_base_delay_ms", 5001),
        ],
    )
    def test_retry_bounds_are_enforced(self, field, value):
        with pytest.raises(ValidationError):
            AppSettingsModel(**{field: value})

    @pytest.mark.parametrize(
        "field, value",
        [("max_retries", 0), ("max_retries", 20), ("retry_total_timeout_seconds", 0.1)],
    )
    def test_retry_bounds_accept_edges(self, field, value):
        assert getattr(AppSettingsModel(**{field: value}), field) == value

    def test_ssl_content_fields_are_excluded_from_dumps(self):
        settings = AppSettingsModel(
            ssl_keyfile_content="KEY", ssl_certfile_content="CERT"
        )

        dumped = settings.model_dump()

        assert "ssl_keyfile_content" not in dumped
        assert "ssl_certfile_content" not in dumped
        assert settings.ssl_keyfile_content == "KEY"

    def test_roundtrip_through_json(self):
        settings = AppSettingsModel(branding_title="Fortress", redis_port=6380)
        assert (
            AppSettingsModel.model_validate_json(settings.model_dump_json()) == settings
        )


class TestServerSchemas:
    def test_server_create_defaults_to_ollama(self):
        server = ServerCreate(name="local", url="http://localhost:11434")
        assert server.server_type == "ollama"
        assert server.api_key is None

    def test_server_create_rejects_invalid_url(self):
        with pytest.raises(ValidationError):
            ServerCreate(name="bad", url="not-a-url")

    def test_server_create_rejects_unknown_server_type(self):
        with pytest.raises(ValidationError):
            ServerCreate(name="bad", url="http://localhost:8000", server_type="tgi")

    def test_server_update_tracks_unset_fields(self):
        update = ServerUpdate(name="renamed")
        assert update.model_dump(exclude_unset=True) == {"name": "renamed"}

    def test_server_update_allows_empty_api_key_to_clear(self):
        assert ServerUpdate(api_key="").model_dump(exclude_unset=True) == {
            "api_key": ""
        }


class TestUserAndApiKeySchemas:
    def test_user_create_requires_password(self):
        with pytest.raises(ValidationError):
            UserCreate(username="bob")

    def test_user_schema_reads_from_orm_attributes(self):
        from app.database.models import User

        user = User(
            id=7, username="bob", hashed_password="x", is_active=True, is_admin=True
        )
        assert UserSchema.model_validate(user).model_dump() == {
            "id": 7,
            "username": "bob",
            "is_active": True,
            "is_admin": True,
        }

    def test_api_key_schema_reads_from_orm_attributes(self):
        from app.database.models import APIKey

        created = datetime.datetime(2024, 1, 1, 12, 0, 0)
        key = APIKey(
            id=1,
            key_name="ci",
            key_prefix="op_abc",
            hashed_key="hash",
            user_id=7,
            expires_at=None,
            is_revoked=False,
            created_at=created,
        )

        schema = APIKeySchema.model_validate(key)

        assert schema.key_prefix == "op_abc"
        assert schema.expires_at is None
        assert schema.created_at == created
