# --- MCP Server: Main Entry Point ---
# This file creates a FastAPI web server to expose the Librarian's capabilities
# as a Model Context Protocol (MCP) server.
import asyncio
import sys
import signal

# FIX: For Playwright/asyncio issues on Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
import os
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# --- Local Imports from your project ---
# We now import the main orchestrator functions, not the individual components.
from services import (
    initialize_services,
    ensure_pinecone_index_ready,
    run_full_pipeline,
    process_image_and_identify_library,
    answer_question,
    logger,
    PineconeVectorStore,
    _sanitize_filename,
    PINECONE_INDEX_NAME
)

# --- Data Models for API Requests ---
class ProcessRequest(BaseModel):
    """The request model for processing a new library."""
    library_name: str

class AskRequest(BaseModel):
    """The request model for asking a question about a library."""
    library_name: str
    question: str

# --- Global State Management ---
# This dictionary will hold our initialized services.
server_state = {}

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print(f"\n--- Received signal {signum}, shutting down gracefully ---")
    server_state.clear()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
if hasattr(signal, 'SIGTERM'):
    signal.signal(signal.SIGTERM, signal_handler)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This function runs on server startup to initialize models and services.
    """
    print("--- Server is starting up... ---")
    load_dotenv()
    
    try:
        text_llm, vision_llm, embeddings_model, pc = initialize_services()
        ensure_pinecone_index_ready(pc, embeddings_model)
        
        # Store the initialized services in the global state
        server_state["text_llm"] = text_llm
        server_state["vision_llm"] = vision_llm
        server_state["embeddings_model"] = embeddings_model
        server_state["pc"] = pc
        
        print("--- Models and services initialized. Server is ready. ---")
        print("🧼 Text Model: llama-3.1-8b-instant (basic, fast)")
        print("🖼️ Vision Model: llama-3.2-11b-vision-preview (image processing)")
        print("🌐 Server URL: http://localhost:8080")
        print("📖 API Documentation: http://localhost:8080/docs")
        
        yield  # Server is running
        
    except Exception as e:
        logger.critical(f"FATAL: Server startup failed during initialization: {e}")
        raise
    finally:
        print("--- Server is shutting down... ---")
        server_state.clear()


# --- FastAPI Application ---
app = FastAPI(
    title="Librarian MCP Server",
    description="An AI-powered service to find, process, and query software documentation.",
    lifespan=lifespan
)

# --- API Endpoints ---

@app.get("/")
async def root():
    """Root endpoint to check if the server is running."""
    return {"message": "Librarian MCP Server running", "status": "healthy"}

@app.post("/process", response_model=dict)
async def process_text_endpoint(request: ProcessRequest):
    """
    This endpoint runs the full Librarian pipeline for a given library name from text.
    """
    library_name = request.library_name
    logger.info(f"Received text request to process library: {library_name}")

    try:
        result = await run_full_pipeline(
            library_name=library_name,
            text_llm=server_state["text_llm"],
            embeddings_model=server_state["embeddings_model"],
            pc=server_state["pc"]
        )
        if not result:
            raise HTTPException(status_code=404, detail=f"Failed to process library '{library_name}'. See logs for details.")
        return result
    except Exception as e:
        logger.error(f"An unexpected error occurred in /process for '{library_name}': {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@app.post("/process_image", response_model=dict)
async def process_image_endpoint(prompt: str = Form("Identify the primary software library in this image."), image: UploadFile = File(...)):
    """
    This endpoint identifies a library from an image, then runs the full pipeline.
    """
    logger.info(f"Received image request with prompt: '{prompt}'")
    
    try:
        # Read image bytes from the uploaded file
        image_bytes = await image.read()

        # 1. Identify the library name from the image using vision model
        identified_name = await process_image_and_identify_library(
            vision_llm=server_state["vision_llm"],
            image_bytes=image_bytes,
            prompt=prompt
        )

        if not identified_name:
            raise HTTPException(status_code=400, detail="Could not identify a library name from the provided image.")

        logger.info(f"Image identified as '{identified_name}'. Starting full pipeline...")

        # 2. Run the full text-based pipeline with the identified name using text model
        result = await run_full_pipeline(
            library_name=identified_name,
            text_llm=server_state["text_llm"],
            embeddings_model=server_state["embeddings_model"],
            pc=server_state["pc"]
        )

        if not result:
            raise HTTPException(status_code=404, detail=f"Successfully identified '{identified_name}' from image, but failed to process it.")
        return result

    except Exception as e:
        logger.error(f"An unexpected error occurred in /process_image: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")


@app.post("/ask", response_model=dict)
async def ask_endpoint(request: AskRequest):
    """
    This is the RAG endpoint. It answers a question about a library using the text model.
    """
    library_name = request.library_name
    question = request.question
    logger.info(f"Received question about '{library_name}': '{question}' (using text model)")

    try:
        result = await answer_question(
            library_name=library_name,
            question=question,
            text_llm=server_state["text_llm"],
            embeddings_model=server_state["embeddings_model"]
        )
        return result

    except Exception as e:
        logger.error(f"Error during RAG retrieval for '{library_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve information. Error: {str(e)}")


# --- Run the Server ---
if __name__ == "__main__":
    """This block allows you to run the server directly for development."""
    try:
        print("Starting Librarian MCP Server with Uvicorn...")
        uvicorn.run(
            "mcp_server:app", 
            host="0.0.0.0", 
            port=8080, 
            reload=True,
            access_log=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n--- Server stopped by user ---")
    except Exception as e:
        print(f"--- Server failed to start: {e} ---")
        raise