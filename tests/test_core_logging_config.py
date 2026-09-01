import json
import logging

from app.core.logging_config import (
    LOGGING_CONFIG,
    HumanReadableFormatter,
    JsonFormatter,
    _build_logging_config,
    setup_logging,
)


def _record(level=logging.INFO, msg="hello"):
    return logging.LogRecord(
        name="test.logger",
        level=level,
        pathname=__file__,
        lineno=10,
        msg=msg,
        args=(),
        exc_info=None,
    )


def test_human_readable_formatter_output():
    formatted = HumanReadableFormatter().format(_record())
    assert "[INFO]" in formatted
    assert "test.logger" in formatted
    assert formatted.endswith("hello")


def test_json_formatter_adds_timestamp_and_uppercase_level():
    formatter = JsonFormatter("%(timestamp)s %(level)s %(name)s %(message)s")
    record = _record(level=logging.WARNING, msg="watch out")

    payload = json.loads(formatter.format(record))

    assert payload["level"] == "WARNING"
    assert payload["message"] == "watch out"
    assert isinstance(payload["timestamp"], float)


def test_default_config_uses_human_formatter(monkeypatch):
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    config = _build_logging_config()

    assert config["handlers"]["default"]["formatter"] == "human"
    assert config["formatters"]["human"]["()"].endswith("HumanReadableFormatter")


def test_json_format_selected_via_environment(monkeypatch):
    monkeypatch.setenv("LOG_FORMAT", "JSON")
    config = _build_logging_config()

    assert config["handlers"]["default"]["formatter"] == "json"
    assert config["formatters"]["json"]["()"].endswith("JsonFormatter")


def test_log_level_is_normalised_across_loggers():
    config = _build_logging_config("debug")

    assert config["root"]["level"] == "DEBUG"
    assert all(logger["level"] == "DEBUG" for logger in config["loggers"].values())


def test_library_loggers_do_not_propagate():
    config = _build_logging_config()

    for name in (
        "uvicorn.error",
        "uvicorn.access",
        "gunicorn.error",
        "gunicorn.access",
    ):
        assert config["loggers"][name]["propagate"] is False
        assert config["loggers"][name]["handlers"] == ["default"]


def test_module_level_config_has_root_logger():
    assert LOGGING_CONFIG["root"]["handlers"] == ["default"]
    assert LOGGING_CONFIG["version"] == 1


def test_setup_logging_applies_configuration():
    try:
        setup_logging("WARNING")
        root = logging.getLogger()
        assert root.level == logging.WARNING
        assert root.handlers
    finally:
        logging.getLogger().setLevel(logging.WARNING)
