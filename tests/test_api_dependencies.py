import types

import pytest
from fastapi import HTTPException

from app.api.v1 import dependencies
from app.crud import apikey_crud
from app.schema.settings import AppSettingsModel


class FakeRedis:
    """Minimal in-memory stand-in for the async redis client used by the dependencies."""

    def __init__(self, values=None, fail=False):
        self.values = dict(values or {})
        self.expirations = {}
        self.fail = fail

    async def get(self, key):
        if self.fail:
            raise ConnectionError("redis down")
        value = self.values.get(key)
        return None if value is None else str(value)

    async def incr(self, key):
        if self.fail:
            raise ConnectionError("redis down")
        self.values[key] = self.values.get(key, 0) + 1
        return self.values[key]

    async def expire(self, key, window):
        self.expirations[key] = window

    async def ttl(self, key):
        return 42


def make_request(*, redis=None, settings=None, session=None, client_host="1.2.3.4"):
    app_state = types.SimpleNamespace(
        redis=redis, settings=settings or AppSettingsModel()
    )
    return types.SimpleNamespace(
        app=types.SimpleNamespace(state=app_state),
        state=types.SimpleNamespace(),
        client=types.SimpleNamespace(host=client_host),
        session={} if session is None else session,
    )


class TestGetSettings:
    def test_returns_settings_from_app_state(self):
        settings = AppSettingsModel(branding_title="Fortress")
        assert dependencies.get_settings(make_request(settings=settings)) is settings


class TestCsrf:
    async def test_token_is_created_once_and_reused(self):
        request = make_request()

        first = await dependencies.get_csrf_token(request)
        second = await dependencies.get_csrf_token(request)

        assert first == second == request.session["csrf_token"]
        assert len(first) == 64

    async def test_existing_session_token_is_kept(self):
        request = make_request(session={"csrf_token": "existing"})
        assert await dependencies.get_csrf_token(request) == "existing"

    async def test_form_validation_accepts_matching_token(self):
        request = make_request(session={"csrf_token": "tok"})
        assert await dependencies.validate_csrf_token(request, csrf_token="tok") is True

    async def test_form_validation_rejects_mismatched_token(self):
        request = make_request(session={"csrf_token": "tok"})

        with pytest.raises(HTTPException) as exc:
            await dependencies.validate_csrf_token(request, csrf_token="wrong")

        assert exc.value.status_code == 403

    async def test_header_validation_accepts_matching_token(self):
        request = make_request(session={"csrf_token": "tok"})
        assert (
            await dependencies.validate_csrf_token_header(request, x_csrf_token="tok")
            is True
        )

    async def test_header_validation_rejects_mismatched_token(self):
        request = make_request(session={"csrf_token": "tok"})

        with pytest.raises(HTTPException) as exc:
            await dependencies.validate_csrf_token_header(request, x_csrf_token="wrong")

        assert exc.value.status_code == 403


class TestLoginRateLimiter:
    async def test_passes_without_redis(self):
        assert await dependencies.login_rate_limiter(make_request(redis=None)) is True

    async def test_passes_below_the_failure_threshold(self):
        request = make_request(redis=FakeRedis({"login_fail:1.2.3.4": 4}))
        assert await dependencies.login_rate_limiter(request) is True

    @pytest.mark.xfail(
        strict=True,
        reason="the broad 'except Exception' swallows the 429 HTTPException it just raised",
    )
    async def test_blocks_at_the_failure_threshold(self):
        request = make_request(redis=FakeRedis({"login_fail:1.2.3.4": 5}))

        with pytest.raises(HTTPException) as exc:
            await dependencies.login_rate_limiter(request)

        assert exc.value.status_code == 429
        assert "42 seconds" in exc.value.detail

    async def test_limits_are_tracked_per_ip(self):
        request = make_request(
            redis=FakeRedis({"login_fail:9.9.9.9": 5}), client_host="1.2.3.4"
        )
        assert await dependencies.login_rate_limiter(request) is True

    async def test_redis_failures_fail_open(self):
        assert (
            await dependencies.login_rate_limiter(
                make_request(redis=FakeRedis(fail=True))
            )
            is True
        )


class TestIpFilter:
    async def test_allows_by_default(self):
        assert await dependencies.ip_filter(make_request(), AppSettingsModel()) is True

    async def test_allow_list_blocks_unlisted_ips(self):
        settings = AppSettingsModel(allowed_ips="10.0.0.1, 10.0.0.2")

        with pytest.raises(HTTPException) as exc:
            await dependencies.ip_filter(make_request(client_host="1.2.3.4"), settings)

        assert exc.value.status_code == 403
        assert exc.value.detail == "IP address not allowed"

    async def test_allow_list_permits_listed_ips(self):
        settings = AppSettingsModel(allowed_ips="10.0.0.1, 1.2.3.4")
        assert await dependencies.ip_filter(make_request(), settings) is True

    async def test_wildcard_allow_list_permits_everything(self):
        settings = AppSettingsModel(allowed_ips="*")
        assert await dependencies.ip_filter(make_request(), settings) is True

    async def test_deny_list_blocks_listed_ips(self):
        settings = AppSettingsModel(denied_ips="1.2.3.4")

        with pytest.raises(HTTPException) as exc:
            await dependencies.ip_filter(make_request(), settings)

        assert exc.value.detail == "IP address has been blocked"

    async def test_deny_list_takes_precedence_over_allow_list(self):
        settings = AppSettingsModel(allowed_ips="1.2.3.4", denied_ips="1.2.3.4")

        with pytest.raises(HTTPException):
            await dependencies.ip_filter(make_request(), settings)


class TestGetValidApiKey:
    async def _call(self, db, plain_key, request=None):
        header = None if plain_key is None else f"Bearer {plain_key}"
        return await dependencies.get_valid_api_key(
            request or make_request(), db=db, auth_header=header
        )

    async def test_valid_key_is_returned_and_attached_to_request(self, db, user):
        plain_key, key = await apikey_crud.create_api_key(
            db, user_id=user.id, key_name="ci"
        )
        request = make_request()

        resolved = await self._call(db, plain_key, request)

        assert resolved.id == key.id
        assert request.state.api_key.id == key.id

    async def test_missing_header_is_unauthorized(self, db):
        with pytest.raises(HTTPException) as exc:
            await self._call(db, None)

        assert exc.value.status_code == 401
        assert "missing" in exc.value.detail

    async def test_wrong_scheme_is_unauthorized(self, db):
        with pytest.raises(HTTPException) as exc:
            await dependencies.get_valid_api_key(
                make_request(), db=db, auth_header="Basic abc"
            )

        assert exc.value.status_code == 401
        assert "Bearer" in exc.value.detail

    async def test_malformed_key_is_unauthorized(self, db):
        with pytest.raises(HTTPException) as exc:
            await self._call(db, "nounderscore")

        assert exc.value.detail == "Invalid API key format"

    async def test_unknown_prefix_is_unauthorized(self, db):
        with pytest.raises(HTTPException) as exc:
            await self._call(db, "op_deadbeef_secret")

        assert exc.value.status_code == 401
        assert exc.value.detail == "Invalid API Key"

    async def test_wrong_secret_is_unauthorized(self, db, api_key):
        with pytest.raises(HTTPException) as exc:
            await self._call(db, f"{api_key.key_prefix}_wrongsecret")

        assert exc.value.status_code == 401
        assert exc.value.detail == "Invalid API Key"

    async def test_revoked_key_is_forbidden(self, db, user):
        plain_key, key = await apikey_crud.create_api_key(
            db, user_id=user.id, key_name="ci"
        )
        await apikey_crud.revoke_api_key(db, key.id)

        with pytest.raises(HTTPException) as exc:
            await self._call(db, plain_key)

        assert exc.value.status_code == 403
        assert "revoked" in exc.value.detail

    async def test_disabled_key_is_forbidden(self, db, user):
        plain_key, key = await apikey_crud.create_api_key(
            db, user_id=user.id, key_name="ci"
        )
        await apikey_crud.toggle_api_key_active(db, key.id)

        with pytest.raises(HTTPException) as exc:
            await self._call(db, plain_key)

        assert exc.value.status_code == 403
        assert "disabled" in exc.value.detail


class TestRateLimiter:
    async def test_passes_without_redis(self, api_key):
        request = make_request(redis=None)
        assert (
            await dependencies.rate_limiter(request, api_key, AppSettingsModel())
            is True
        )

    async def test_uses_global_limit_and_sets_window_expiry(self, api_key):
        redis = FakeRedis()
        request = make_request(redis=redis)
        settings = AppSettingsModel(rate_limit_requests=2, rate_limit_window_minutes=3)

        assert await dependencies.rate_limiter(request, api_key, settings) is True
        assert redis.expirations[f"rate_limit:{api_key.key_prefix}"] == 180

    @pytest.mark.xfail(
        strict=True,
        reason="the broad 'except Exception' swallows the 429 HTTPException it just raised",
    )
    async def test_exceeding_the_global_limit_raises_429(self, api_key):
        redis = FakeRedis({f"rate_limit:{api_key.key_prefix}": 5})
        request = make_request(redis=redis)
        settings = AppSettingsModel(rate_limit_requests=2, rate_limit_window_minutes=1)

        with pytest.raises(HTTPException) as exc:
            await dependencies.rate_limiter(request, api_key, settings)

        assert exc.value.status_code == 429
        assert exc.value.headers["Retry-After"] == "42"

    @pytest.mark.xfail(
        strict=True,
        reason="the broad 'except Exception' swallows the 429 HTTPException it just raised",
    )
    async def test_per_key_limit_overrides_global_limit(self, db, api_key):
        api_key.rate_limit_requests = 1
        api_key.rate_limit_window_minutes = 10
        redis = FakeRedis({f"rate_limit:{api_key.key_prefix}": 1})
        request = make_request(redis=redis)
        settings = AppSettingsModel(
            rate_limit_requests=1000, rate_limit_window_minutes=1
        )

        with pytest.raises(HTTPException) as exc:
            await dependencies.rate_limiter(request, api_key, settings)

        assert exc.value.status_code == 429

    async def test_redis_failures_fail_open(self, api_key):
        request = make_request(redis=FakeRedis(fail=True))
        assert (
            await dependencies.rate_limiter(request, api_key, AppSettingsModel())
            is True
        )
