# N'Ko Continuous Learning Pipeline
# Runs scheduled vocabulary enrichment and video processing

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Deno (required by yt-dlp for YouTube extraction)
RUN curl -fsSL https://deno.land/install.sh | sh
ENV DENO_INSTALL="/root/.deno"
ENV PATH="${DENO_INSTALL}/bin:${PATH}"

# Install yt-dlp
RUN curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -o /usr/local/bin/yt-dlp \
    && chmod a+rx /usr/local/bin/yt-dlp

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training code
COPY . .

# Create data directories
RUN mkdir -p /app/data/frames /app/data/audio /app/data/checkpoints /app/logs

# Environment variables (override at runtime)
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Health check endpoint (optional - for monitoring)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "print('healthy')" || exit 1

# Default command: run the continuous learning scheduler
CMD ["python", "scripts/scheduled_exploration.py", "--interval", "300"]

