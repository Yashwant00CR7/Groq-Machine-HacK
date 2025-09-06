# --- Configuration and Schema Definitions ---
# This file contains all the static settings, environment variables,
# and data structures for the documentation fetcher.

import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv()

# --- Global Configuration ---
PINECONE_INDEX_NAME = "mcp-documentation-index-groq"
JINA_READER_API_URL = "https://r.jina.ai/"

# --- Pydantic Schema for Rich Data Extraction ---
class LibraryInfo(BaseModel):
    """A structured representation of key information about a software library."""
    library_name: Optional[str] = Field(description="The official name of the library (e.g., 'Pinecone')")
    package_name: Optional[str] = Field(description="The correct, current package name for installation (e.g., 'pinecone')")
    latest_version: Optional[str] = Field(description="The latest stable version number found (e.g., '4.1.0')")
    summary: Optional[str] = Field(description="A brief, one-sentence summary of the library's primary purpose.")
    installation_command: Optional[str] = Field(description="The standard installation command (e.g., 'pip install pinecone' or 'npm install express')")
    deprecation_notice: Optional[str] = Field(description="A notice if the original library name is deprecated or has been renamed.")
    ecosystem: Optional[str] = Field(description="The software ecosystem the library belongs to (e.g., 'Python/PyPI', 'JavaScript/npm')")

# --- Environment Variable Validation ---
def validate_environment():
    """Validates that required environment variables are set."""
    required_vars = ["GROQ_API_KEY", "PINECONE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Check for optional Tavily API key (used as fallback)
    if not os.getenv("TAVILY_API_KEY"):
        print("⚠️  TAVILY_API_KEY not set. Tavily AI search fallback will not be available.")
        print("   Get your free API key from: https://tavily.com/")
    
    return True 