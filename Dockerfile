# Use official slim Python runtime
FROM python:3.11-slim

# Set environment variables: disable bytecode writes and force unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# Install system dependencies (build tools if needed for math/C-extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies first to take advantage of Docker layer caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and data files
COPY . /app/

# Default entry command runs the data loader module
CMD ["python", "-m", "src.data_loader"]