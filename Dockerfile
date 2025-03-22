FROM python:3.10.16-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p db temp

# Copy application files
COPY backend/ /app/
COPY frontend/ /app/frontend/
COPY models/ /app/models/

ENV PORT=8000
ENV WEBSITES_PORT=8000

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "-u", "main.py"]
