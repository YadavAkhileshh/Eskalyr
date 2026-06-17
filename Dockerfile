# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENV PIP_DEFAULT_TIMEOUT=1000

# Set the working directory
WORKDIR /app

# Install system dependencies required for building some python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the pyproject.toml
COPY pyproject.toml .

# CRITICAL FIX: Force install the tiny CPU-only version of PyTorch first. 
# Without this, Docker tries to download the massive 2.5GB GPU version which causes Windows Docker Desktop to freeze.
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the application using standard pip (avoiding aggressive timeouts from uv)
RUN pip install .

# Copy the rest of the application code
COPY . .

# Create the data directory for SQLite databases
RUN mkdir -p /app/data

# Expose ports for both the main app (8501) and the dashboard (8502)
EXPOSE 8501
EXPOSE 8502

# The default command runs the main chat application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
