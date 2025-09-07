# Start from an official Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /code

# Create a non-root user for security
RUN useradd -m appuser
USER appuser

# Add the user's local bin to the PATH
# This ensures that executables installed by pip (like uvicorn) can be found.
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Copy the requirements file first to leverage Docker's caching
COPY --chown=appuser:appuser requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# CRITICAL STEP (CORRECTED): Install the browser binaries WITHOUT system dependencies
RUN python -m playwright install chromium

# Download the embedding model TO THE CORRECT CACHE DIRECTORY during build
RUN python -c "import os; from langchain_community.embeddings import FastEmbedEmbeddings; \
    cache_dir = os.path.join(os.path.expanduser('~'), '.embeddings_cache'); \
    os.makedirs(cache_dir, exist_ok=True); \
    FastEmbedEmbeddings(model_name='BAAI/bge-small-en-v1.5', cache_dir=cache_dir); \
    print('Embedding model downloaded successfully to:', cache_dir)"

# Copy the rest of your application code into the container
COPY --chown=appuser:appuser . .

# Expose the port that Cloud Run will assign (this is for documentation)
EXPOSE 8080

# The command to run your FastAPI server
# This "shell form" correctly interprets the $PORT environment variable.
CMD uvicorn mcp_server:app --host 0.0.0.0 --port ${PORT:-8080}