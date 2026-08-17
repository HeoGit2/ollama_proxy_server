from pydantic import model_validator
from pydantic_settings import BaseSettings

# The secret key that used to ship as the default in .env.example / config.py.
# It is public knowledge, so it must never be accepted at runtime: it would let
# anyone forge admin session cookies and decrypt stored backend API keys.
COMPROMISED_SECRET_KEY = (
    "dd2a57833f4a2115b02644c3c332822d5b6e405d542a2258c422fb39a8e97b10"
)
MIN_SECRET_KEY_LENGTH = 32

SECRET_KEY_HELP = (
    "SECRET_KEY must be set to a private, random value of at least "
    f"{MIN_SECRET_KEY_LENGTH} characters. Generate one with:\n"
    '    python -c "import secrets; print(secrets.token_hex(32))"\n'
    "then put it in your .env file (or run setup_wizard.py). Changing it "
    "invalidates existing admin sessions."
)


class Settings(BaseSettings):
    # --- Bootstrap Settings ---
    # These are the only settings read from the .env file.
    DATABASE_URL: str = "sqlite+aiosqlite:///./ollama_proxy.db"
    ADMIN_USER: str = "admin"
    ADMIN_PASSWORD: str = "changeme"
    PROXY_PORT: int = 8080
    SECRET_KEY: str = ""

    # Send the session cookie only over HTTPS. Enable this whenever the proxy
    # is reachable over TLS (directly or behind a TLS-terminating reverse proxy).
    SESSION_COOKIE_SECURE: bool = False
    # Lifetime of an admin session cookie, in seconds.
    SESSION_MAX_AGE_SECONDS: int = 8 * 60 * 60

    # --- App Info (Hardcoded) ---
    APP_NAME: str = "Ollama Proxy Server"
    APP_VERSION: str = "9.0.0"
    LOG_LEVEL: str = "info"

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = 'ignore'

    @model_validator(mode="after")
    def validate_secret_key(self) -> "Settings":
        if not self.SECRET_KEY:
            raise ValueError(f"SECRET_KEY is not configured. {SECRET_KEY_HELP}")
        if self.SECRET_KEY == COMPROMISED_SECRET_KEY:
            raise ValueError(
                "SECRET_KEY is set to the value that was published in this "
                f"project's example configuration. {SECRET_KEY_HELP}"
            )
        if len(self.SECRET_KEY) < MIN_SECRET_KEY_LENGTH:
            raise ValueError(f"SECRET_KEY is too short. {SECRET_KEY_HELP}")
        return self


# This `settings` object is now only used for bootstrapping.
# The rest of the app will use settings loaded from the DB.
settings = Settings()
