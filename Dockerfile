FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the port that Gradio will run on
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8000
ENV DEBUG=false
ENV SHARE=false

# Create a non-root user and switch to it
RUN useradd -m appuser
USER appuser

# Run the application
CMD ["python", "main.py"]
