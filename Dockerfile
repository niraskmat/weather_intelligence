# Use official Python 3.11 slim base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libatlas-base-dev \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --prefer-binary -r requirements.txt


# Copy application source code
COPY . .

# Expose API port
EXPOSE 8000

# Set environment variable defaults (can be overridden)
ENV LOG_LEVEL=INFO
ENV DATA_SEED=42
#ENV ANALYSIS_WINDOW_DAYS=30
ENV CACHE_RESULTS=false

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
