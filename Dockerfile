FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install minimal system deps
RUN apt-get update && apt-get install -y curl git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first so Docker cache helps during development
COPY requirements.txt .

# Install python deps
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure model & assets directories exist (will be mounted in dev)
RUN mkdir -p /app/models /app/assets

# Expose Streamlit default port
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
