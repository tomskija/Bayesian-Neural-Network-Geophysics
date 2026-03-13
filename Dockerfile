# Multi-stage build for production optimization
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Development stage (for devcontainer compatibility)
FROM base as development
RUN apt-get update && apt-get install -y \
    git \
    vim \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for development
RUN useradd -m -s /bin/bash developer \
    && echo "developer ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set up development workspace
USER developer
WORKDIR /workspace

# Production stage
FROM base as production

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code (fix path issues)
COPY weatherForcastingCalculator/ ./weatherForcastingCalculator/
COPY *.py ./
COPY README.md .

# Create non-root user for security
RUN useradd -m -s /bin/bash appuser
RUN chown -R appuser:appuser /app
USER appuser

# Set Python path
ENV PYTHONPATH=/app

# Create necessary directories
RUN mkdir -p /app/weatherForcastingCalculator/Data \
    && mkdir -p /app/logs

# Expose port
EXPOSE 8000

# Health check (assuming FastAPI is used)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - Update this based on your actual entry point
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]NUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Development stage (optional - for devcontainer compatibility)
FROM base as development
RUN apt-get update && apt-get install -y \
    git \
    vim \
    sudo \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for development
RUN useradd -m -s /bin/bash developer \
    && echo "developer ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER developer
WORKDIR /workspace

# Production stage
FROM base as production

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY weatherForcastingCalculator/ ./weatherForcastingCalculator/
COPY models/ ./models/
COPY data/ ./data/
COPY notebooks/ ./notebooks/
COPY readme.md .

# Create non-root user for security
RUN useradd -m -s /bin/bash appuser
RUN chown -R appuser:appuser /app
USER appuser

# Set Python path
ENV PYTHONPATH=/app

# Expose port (adjust based on your app)
EXPOSE 8000

# Health check (optional)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - adjust based on your main application file
CMD ["python", "weatherForcastingCalculator/Calculator.py"]