# --- MCP Server: Main Entry Point ---
# This file creates a FastAPI web server to expose the Librarian's capabilities
# as a Model Context Protocol (MCP) server.
import asyncio
import sys

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This function runs on server startup to initialize models and services.
    """
    print("--- Server is starting up... ---")
    load_dotenv()
    
    try:
        llm, embeddings_model, pc = initialize_services()
        ensure_pinecone_index_ready(pc, embeddings_model)
        
        # Store the initialized services in the global state
        server_state["llm"] = llm
        server_state["embeddings_model"] = embeddings_model
        server_state["pc"] = pc
        
        print("--- Models and services initialized. Server is ready. ---")
        print("🌐 Server URL: http://localhost:8080")
        print("📖 API Documentation: http://localhost:8080/docs")
    except Exception as e:
        logger.critical(f"FATAL: Server startup failed during initialization: {e}")
    
    yield
    
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
            llm=server_state["llm"],
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

        # 1. Identify the library name from the image
        identified_name = await process_image_and_identify_library(
            llm=server_state["llm"],
            image_bytes=image_bytes,
            prompt=prompt
        )

        if not identified_name:
            raise HTTPException(status_code=400, detail="Could not identify a library name from the provided image.")

        logger.info(f"Image identified as '{identified_name}'. Starting full pipeline...")

        # 2. Run the full text-based pipeline with the identified name
        result = await run_full_pipeline(
            library_name=identified_name,
            llm=server_state["llm"],
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
    This is the RAG endpoint. It answers a question about a library.
    """
    library_name = request.library_name
    question = request.question
    logger.info(f"Received question about '{library_name}': '{question}'")

    try:
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=server_state["embeddings_model"]
        )

        doc_id = f"lib-{_sanitize_filename(library_name)}"
        retriever = vectorstore.as_retriever(
            search_kwargs={'k': 5, 'filter': {'doc_id': doc_id}}
        )

        docs = await retriever.ainvoke(question)

        if not docs:
            return {"answer": "I could not find any relevant information in the documentation for that library. It might not have been processed yet or the name is incorrect."}

        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        
        # Here you would typically pass the context and question to an LLM to generate a natural language answer.
        # For this example, we return the raw context.
        return {
            "answer": f"Based on the documentation for {library_name}, here is some relevant context for your question: '{question}'",
            "context": context
        }

    except Exception as e:
        logger.error(f"Error during RAG retrieval for '{library_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve information. Error: {str(e)}")


# --- Run the Server ---
if __name__ == "__main__":
    """This block allows you to run the server directly for development."""
    print("Starting Librarian MCP Server with Uvicorn...")
    uvicorn.run("mcp_server:app", host="0.0.0.0", port=8080, reload=True)