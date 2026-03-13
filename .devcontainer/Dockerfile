# Use Python 3.11 slim base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (including those needed for scientific computing)
RUN apt-get update && apt-get install -y \
    curl \
    git \
    wget \
    vim \
    sudo \
    build-essential \
    pkg-config \
    libpq-dev \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Create workspace directory
WORKDIR /workspace

# Create a non-root user for development
RUN groupadd --gid 1000 developer \
    && useradd --uid 1000 --gid developer --shell /bin/bash --create-home developer \
    && echo "developer:developer" | chpasswd \
    && adduser developer sudo \
    && echo "developer ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Give developer ownership of workspace
RUN chown -R developer:developer /workspace

# Switch to developer user
USER developer

# Set the default shell
SHELL ["/bin/bash", "-c"]

# Install Python tools
RUN pip install --user --upgrade pip setuptools wheel

# Pre-install common dependencies for faster container startup
RUN pip install --user \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    scipy \
    requests \
    fastapi \
    uvicorn \
    pytest \
    black \
    flake8 \
    isort

# Set Python path and PATH
ENV PYTHONPATH=/workspace
ENV PATH="/home/developer/.local/bin:${PATH}"

# Set the working directory
WORKDIR /workspace

# Create directory structure for the application
RUN mkdir -p /workspace/rockPropCalculator/Data \
    && mkdir -p /workspace/rockPropCalculator/DataLoad \
    && mkdir -p /workspace/rockPropCalculator/FiguresAndResults \
    && mkdir -p /workspace/rockPropCalculator/data \
    && mkdir -p /workspace/rockPropCalculator/models \
    && mkdir -p /workspace/rockPropCalculator/utils \
    && mkdir -p /workspace/tests \
    && mkdir -p /workspace/logs

# Default command
CMD ["/bin/bash"]