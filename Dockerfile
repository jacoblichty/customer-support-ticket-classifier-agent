# Multi-stage Docker build for production deployment
FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# Set work directory and change ownership
WORKDIR /app
RUN chown appuser:appuser /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base AS development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-mock \
    pytest-cov \
    black \
    flake8 \
    isort \
    mypy

# Copy source code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Create logs directory as appuser
RUN mkdir -p logs

# Expose port
EXPOSE 8000

# Command for development
CMD ["python", "main.py", "--server", "--host", "0.0.0.0", "--port", "8000"]

# Production stage
FROM base AS production

# Copy only necessary files
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser main.py .
COPY --chown=appuser:appuser requirements.txt .

# Switch to non-root user
USER appuser

# Create logs directory as appuser
RUN mkdir -p logs

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Command for production
CMD ["python", "main.py", "--server", "--env", "production"]

# Testing stage
FROM development AS testing

# Copy test files
COPY --chown=appuser:appuser tests/ ./tests/
COPY --chown=appuser:appuser pyproject.toml .

# Switch to non-root user for tests
USER appuser

# Run tests with cache disabled to eliminate warnings
RUN python -m pytest tests/ -v --tb=short -p no:cacheprovider -p no:stepwise

# Default to production stage
FROM development