# ---------- Base image ----------
FROM python:3.11-slim

# ---------- Install system dependencies ----------
# ffmpeg is required for MP3 / FLAC / OGG exports
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ---------- Set working directory ----------
WORKDIR /app

# ---------- Install Python dependencies ----------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Copy application code ----------
COPY . .

# ---------- Runtime config ----------
ENV PORT=5890
EXPOSE 5890

# ---------- Start production server ----------
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5890", "brainwave:app"]