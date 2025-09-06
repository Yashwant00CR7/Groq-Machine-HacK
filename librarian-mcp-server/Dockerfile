# Start from an official Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /code

# --- INSTALLATION STEPS (as root) ---

# Copy the requirements file first to leverage Docker's caching
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# CRITICAL STEP: Install browser binaries and system dependencies as root
RUN python -m playwright install --with-deps chromium

# Download the embedding model TO THE CORRECT CACHE DIRECTORY during build
RUN python -c "import os; from langchain_community.embeddings import FastEmbedEmbeddings; \
    cache_dir = os.path.join(os.getcwd(), '.embeddings_cache'); \
    os.makedirs(cache_dir, exist_ok=True); \
    FastEmbedEmbeddings(model_name='BAAI/bge-small-en-v1.5', cache_dir=cache_dir); \
    print('Embedding model downloaded successfully to:', cache_dir)"

# --- APPLICATION SETUP (as non-root) ---

# Create a non-root user
RUN useradd -m appuser

# Copy the rest of your application code, giving ownership to the new user
COPY --chown=appuser:appuser . .

# Switch to the non-root user for running the application
USER appuser

# Expose the port that your application will run on
EXPOSE 8080

# The command to run your server when the container starts
CMD ["uvicorn", "mcp_server:app", "--host", "0.0.0.0", "--port", "8080"]