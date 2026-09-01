from app.crud import model_metadata_crud, settings_crud
from app.schema.settings import AppSettingsModel


class TestSettingsCrud:
    async def test_get_app_settings_returns_none_when_empty(self, db):
        assert await settings_crud.get_app_settings(db) is None

    async def test_create_initial_settings_writes_defaults(self, db):
        created = await settings_crud.create_initial_settings(db)

        assert created.id == 1
        assert created.settings_data["branding_title"] == "Ollama Proxy"
        assert created.settings_data["max_retries"] == 5

    async def test_create_initial_settings_is_idempotent(self, db):
        first = await settings_crud.create_initial_settings(db)
        first.settings_data = {**first.settings_data, "branding_title": "Kept"}
        await db.commit()

        second = await settings_crud.create_initial_settings(db)

        assert second.settings_data["branding_title"] == "Kept"

    async def test_update_app_settings_creates_row_when_missing(self, db):
        updated = await settings_crud.update_app_settings(
            db, AppSettingsModel(branding_title="Fortress", redis_port=6380)
        )

        assert updated.id == 1
        assert updated.settings_data["branding_title"] == "Fortress"
        assert updated.settings_data["redis_port"] == 6380

    async def test_update_app_settings_overwrites_existing_row(self, db):
        await settings_crud.create_initial_settings(db)

        await settings_crud.update_app_settings(
            db, AppSettingsModel(branding_title="New")
        )
        stored = await settings_crud.get_app_settings(db)

        assert stored.settings_data["branding_title"] == "New"

    async def test_update_app_settings_excludes_ssl_content(self, db):
        await settings_crud.update_app_settings(
            db, AppSettingsModel(ssl_keyfile_content="KEY", ssl_certfile_content="CERT")
        )
        stored = await settings_crud.get_app_settings(db)

        assert "ssl_keyfile_content" not in stored.settings_data
        assert "ssl_certfile_content" not in stored.settings_data

    async def test_stored_settings_reload_into_the_pydantic_model(self, db):
        await settings_crud.update_app_settings(
            db, AppSettingsModel(rate_limit_requests=7)
        )
        stored = await settings_crud.get_app_settings(db)

        assert AppSettingsModel(**stored.settings_data).rate_limit_requests == 7


class TestModelMetadataCrud:
    async def test_get_metadata_returns_none_for_unknown_model(self, db):
        assert await model_metadata_crud.get_metadata_by_model_name(db, "ghost") is None

    async def test_get_or_create_creates_default_entry(self, db):
        metadata = await model_metadata_crud.get_or_create_metadata(db, "llama3:8b")

        assert metadata.id is not None
        assert metadata.description == "Auto-discovered model."
        assert metadata.supports_images is False
        assert metadata.is_chat_model is True
        assert metadata.priority == 10

    async def test_get_or_create_is_idempotent(self, db):
        first = await model_metadata_crud.get_or_create_metadata(db, "llama3:8b")
        second = await model_metadata_crud.get_or_create_metadata(db, "llama3:8b")

        assert first.id == second.id
        assert len(await model_metadata_crud.get_all_metadata(db)) == 1

    async def test_vision_models_default_to_image_support(self, db):
        llava = await model_metadata_crud.get_or_create_metadata(db, "llava:13b")
        bakllava = await model_metadata_crud.get_or_create_metadata(db, "bakllava")

        assert llava.supports_images is True
        assert bakllava.supports_images is True

    async def test_get_all_metadata_sorts_by_priority_then_name(self, db):
        await model_metadata_crud.get_or_create_metadata(db, "zeta")
        await model_metadata_crud.get_or_create_metadata(db, "alpha")
        await model_metadata_crud.get_or_create_metadata(db, "high-priority")
        await model_metadata_crud.update_metadata(db, "high-priority", priority=1)

        names = [
            metadata.model_name
            for metadata in await model_metadata_crud.get_all_metadata(db)
        ]

        assert names == ["high-priority", "alpha", "zeta"]

    async def test_update_metadata_sets_known_fields(self, db):
        await model_metadata_crud.get_or_create_metadata(db, "codellama")

        updated = await model_metadata_crud.update_metadata(
            db, "codellama", is_code_model=True, description="Code model", priority=2
        )

        assert updated.is_code_model is True
        assert updated.description == "Code model"
        assert updated.priority == 2

    async def test_update_metadata_ignores_unknown_fields(self, db):
        await model_metadata_crud.get_or_create_metadata(db, "llama3")

        updated = await model_metadata_crud.update_metadata(
            db, "llama3", not_a_field="x"
        )

        assert not hasattr(updated, "not_a_field")

    async def test_update_missing_metadata_returns_none(self, db):
        assert (
            await model_metadata_crud.update_metadata(db, "ghost", priority=1) is None
        )
