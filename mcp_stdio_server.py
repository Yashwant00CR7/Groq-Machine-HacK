# --- Librarian: The Stdio to HTTP Cloud Relay ---
# This script creates a lightweight, stdio-based MCP server that acts as a
# bridge between a local VS Code client and your deployed HTTP server on Google Cloud.

import os
import traceback
import requests
from typing import Dict, Any

from dotenv import load_dotenv
import asyncio
import logging

# === MCP server imports (stdio) ===
from mcp.server.fastmcp import FastMCP

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global State for the Relay ---
server_state: Dict[str, Any] = {}

app = FastMCP(
    "librarian-cloud-relay",
    "A relay server that connects to the deployed Librarian AI agent on Google Cloud."
)

async def on_startup() -> None:
    """
    On startup, this function loads the URL of the deployed cloud server
    and performs a quick health check to ensure it's reachable.
    """
    load_dotenv()
    logger.info("[MCP Relay] Startup: Initializing cloud connection...")
    
    cloud_url = os.getenv("LIBRARIAN_CLOUD_URL")
    if not cloud_url:
        logger.critical("[MCP Relay] CRITICAL STARTUP FAILED: LIBRARIAN_CLOUD_URL is not set in the .env file.")
        raise ValueError("LIBRARIAN_CLOUD_URL environment variable is not set.")
        
    server_state["base_url"] = cloud_url.rstrip('/') + '/'
    
    try:
        logger.info(f"[MCP Relay] Pinging cloud server at {server_state['base_url']}...")
        response = await asyncio.to_thread(requests.get, server_state['base_url'], timeout=10)
        response.raise_for_status()
        logger.info(f"[MCP Relay] Cloud server responded successfully: {response.json().get('message')}")
        logger.info("[MCP Relay] Startup complete. Relay is ready.")
    except requests.RequestException as e:
        logger.critical(f"[MCP Relay] CRITICAL STARTUP FAILED: Could not connect to the cloud server at {server_state['base_url']}. Error: {e}")
        raise

@app.tool()
async def process_library(library_name: str) -> Dict[str, Any]:
    """
    Relays the request to the /process_library endpoint on the deployed cloud server.
    All heavy lifting is done in the cloud.
    """
    logger.info(f"[MCP Relay] Relaying 'process_library' request for '{library_name}' to the cloud...")
    
    try:
        endpoint_url = server_state["base_url"] + "process_library"
        payload = {"library_name": library_name}
        
        response = await asyncio.to_thread(
            requests.post, endpoint_url, json=payload, timeout=600
        )
        
        response.raise_for_status()
        
        logger.info(f"[MCP Relay] Received successful response from cloud for '{library_name}'.")
        return response.json()

    except requests.HTTPError as e:
        error_details = e.response.json().get("detail", str(e))
        logger.error(f"[MCP Relay] Cloud server returned an error: {e.response.status_code} - {error_details}")
        return {"error": f"Cloud server error: {error_details}"}
    except Exception as e:
        logger.error(f"[MCP Relay] 'process_library' tool failed: {e}\n{traceback.format_exc()}")
        return {
            "error": f"An internal relay error occurred: {str(e)}",
        }


@app.tool()
async def ask(library_name: str, question: str) -> Dict[str, Any]:
    """
    Relays a question to the /ask RAG endpoint on the deployed cloud server.
    """
    logger.info(f"[MCP Relay] Relaying 'ask' request for '{library_name}' to the cloud...")

    try:
        endpoint_url = server_state["base_url"] + "ask"
        payload = {"library_name": library_name, "question": question}

        response = await asyncio.to_thread(
            requests.post, endpoint_url, json=payload, timeout=120
        )
        response.raise_for_status()
        
        logger.info(f"[MCP Relay] Received successful RAG answer from cloud for '{library_name}'.")
        return response.json()

    except requests.HTTPError as e:
        error_details = e.response.json().get("detail", str(e))
        logger.error(f"[MCP Relay] Cloud server returned an error on /ask: {e.response.status_code} - {error_details}")
        return {"error": f"Cloud server RAG error: {error_details}"}
    except Exception as e:
        logger.error(f"[MCP Relay] 'ask' relay failed: {e}\n{traceback.format_exc()}")
        return {
            "error": f"An internal relay error occurred during 'ask': {str(e)}",
        }


@app.tool()
async def ping() -> str:
    """A simple health check tool to confirm the local relay is running."""
    return "pong from local relay"


if __name__ == "__main__":
    # This is the main coroutine that sets up and runs the server.
    async def main():
        await on_startup()
        # We call the internal async method directly to avoid the library
        # trying to start a new, conflicting event loop.
        await app.run_stdio_async()

    # This logic robustly handles running the server in different environments.
    try:
        # Check if an event loop is already running (like in VS Code).
        loop = asyncio.get_running_loop()
        # If so, schedule the main coroutine as a task on the existing loop.
        loop.create_task(main())
    except RuntimeError:
        # If no event loop is running (like when you run `python mcp_stdio_server.py`
        # from a normal terminal), start a new one.
        asyncio.run(main())
