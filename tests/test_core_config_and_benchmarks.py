import base64
import sys

import pytest

from app.core.benchmarks import PREBUILT_BENCHMARKS
from app.core.config import Settings, settings


class TestSettings:
    def test_defaults(self):
        fresh = Settings(_env_file=None)
        assert fresh.DATABASE_URL.startswith("sqlite+aiosqlite:///")
        assert fresh.ADMIN_USER == "admin"
        assert fresh.PROXY_PORT == 8080
        assert fresh.APP_NAME == "Ollama Proxy Server"

    def test_reads_values_from_environment(self, monkeypatch):
        monkeypatch.setenv("ADMIN_USER", "root")
        monkeypatch.setenv("PROXY_PORT", "9999")

        fresh = Settings(_env_file=None)

        assert fresh.ADMIN_USER == "root"
        assert fresh.PROXY_PORT == 9999

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="os.environ folds variable names to a single case at the OS level on "
        "Windows, so monkeypatch.setenv('admin_user', ...) is indistinguishable from "
        "setting ADMIN_USER before pydantic-settings ever sees it — this test cannot "
        "observe case_sensitive=True on this platform",
    )
    def test_is_case_sensitive(self, monkeypatch):
        monkeypatch.setenv("admin_user", "lowercase")
        assert Settings(_env_file=None).ADMIN_USER == "admin"

    def test_ignores_unknown_environment_variables(self, monkeypatch):
        monkeypatch.setenv("TOTALLY_UNRELATED_SETTING", "value")
        assert Settings(_env_file=None).ADMIN_USER == "admin"

    def test_invalid_port_is_rejected(self, monkeypatch):
        monkeypatch.setenv("PROXY_PORT", "not-a-port")
        with pytest.raises(Exception):
            Settings(_env_file=None)

    def test_secret_key_is_usable_as_a_fernet_key(self):
        assert len(base64.urlsafe_b64encode(settings.SECRET_KEY.encode()[:32])) == 44


class TestPrebuiltBenchmarks:
    def test_benchmarks_are_present_and_named(self):
        assert PREBUILT_BENCHMARKS
        names = [benchmark["name"] for benchmark in PREBUILT_BENCHMARKS]
        assert len(names) == len(set(names))

    @pytest.mark.parametrize("benchmark", PREBUILT_BENCHMARKS, ids=lambda b: b["name"])
    def test_benchmark_groups_are_well_formed(self, benchmark):
        groups = benchmark["groups"]
        assert len(groups) >= 2

        group_ids = [group["id"] for group in groups]
        assert len(group_ids) == len(set(group_ids))

        for group in groups:
            assert group["name"]
            assert group["color"].startswith("#") and len(group["color"]) == 7
            assert len(group["texts"]) >= 2
            assert all(
                isinstance(text, str) and text.strip() for text in group["texts"]
            )
