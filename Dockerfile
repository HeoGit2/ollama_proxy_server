# --- Build Stage ---
FROM python:3.13-slim AS builder

# Set working directory
WORKDIR /app

# Install poetry
RUN pip install poetry gunicorn

# Copy only dependency-defining files
COPY pyproject.toml ./

# Install dependencies, without dev dependencies, into a virtual environment
RUN poetry config virtualenvs.create false && \
    poetry install --without dev --no-root --no-interaction --no-ansi

# Set a non-root user
RUN addgroup --system app && adduser --system --group app

# Set working directory
WORKDIR /home/app

COPY --chown=app:app ./app ./app
COPY --chown=app:app gunicorn_conf.py .

# Runtime-writable dirs the app creates/uses at startup (uploads, SSL cert
# storage, benchmarks, default sqlite data dir). Must be owned by the
# non-root `app` user before we drop root below — COPY without --chown (and
# any mkdir() the app does at runtime) default to root ownership otherwise,
# which crashes startup with PermissionError under this non-root USER.
RUN mkdir -p app/static/uploads .ssl benchmarks data && \
    chown -R app:app app/static/uploads .ssl benchmarks data

USER app

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application using our custom Gunicorn config file.
# This ensures structured JSON logging is used in production.
CMD ["gunicorn", "-c", "./gunicorn_conf.py", "app.main:app"]
