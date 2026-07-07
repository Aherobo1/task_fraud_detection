FROM python:3.9-slim

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .

# Create necessary directories
RUN mkdir -p data/processed models

# Expose ports for API and Web UI
EXPOSE 8001 8501

# Set environment variables
ENV PYTHONPATH=/app

# Command to run the API and Web UI
CMD ["sh", "-c", "python -m src.api.app & python -m src.web.app"]
