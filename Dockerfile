# # ── Build stage ──────────────────────────────────────────────────────────────
# FROM python:3.13-slim AS builder

# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1 \
#     UV_COMPILE_BYTECODE=1 \
#     UV_LINK_MODE=copy

# WORKDIR /app

# # Install uv
# COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# # Copy lockfile and project metadata first (layer cache)
# COPY pyproject.toml uv.lock ./

# # Install dependencies into /app/.venv — no project code yet
# RUN uv sync --frozen --no-install-project --no-dev

# # Copy source and install the project itself
# COPY . .
# RUN uv sync --frozen --no-dev


# # ── Runtime stage ─────────────────────────────────────────────────────────────
# FROM python:3.13-slim

# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1 \
#     PYTHONPATH=/app \
#     PATH="/app/.venv/bin:$PATH"

# WORKDIR /app

# # Create non-root user before copying files
# RUN adduser -u 5678 --disabled-password --gecos "" appuser

# # Copy only the installed venv and app code from builder
# COPY --from=builder --chown=appuser:appuser /app /app

# USER appuser

# EXPOSE 8000

# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#     CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# # Use venv python directly — no uv overhead at runtime
# CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# AWS Lambda Python 3.12 Base Image
FROM public.ecr.aws/lambda/python:3.13
 
# Install 'uv' for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
 
# Copy Dependency Files
COPY pyproject.toml uv.lock ${LAMBDA_TASK_ROOT}/
 
# Install Python Dependencies directly to system Python (required for Lambda)
RUN cd ${LAMBDA_TASK_ROOT} && uv pip install --system --no-cache-dir .
 
# Copy Your Code
COPY . ${LAMBDA_TASK_ROOT}
 
# Lambda Handler
CMD [ "main.handler" ]