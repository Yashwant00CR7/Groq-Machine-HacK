# --- MCP Server: Main Entry Point ---
# This file creates a FastAPI web server to expose the Librarian's capabilities
# as a Model Context Protocol (MCP) server.

import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# --- Local Imports from your project ---
# These functions contain the core intelligence of the Librarian.
from services import (
    initialize_services,
    create_universal_agent,
    ensure_pinecone_index_ready,
    extract_structured_info,
    load_from_cache,
    save_to_cache,
    interpret_confidence_score,
    logger,
    PineconeVectorStore,
    _sanitize_filename,
    find_documentation_url
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
server_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This function runs on server startup and shutdown to initialize models and services.
    """
    print("--- Server is starting up... ---")
    load_dotenv()
    
    try:
        llm, embeddings_model, pc = initialize_services()
        ensure_pinecone_index_ready(pc, embeddings_model)
        agent_executor = create_universal_agent(llm)
        
        server_state["llm"] = llm
        server_state["embeddings_model"] = embeddings_model
        server_state["agent_executor"] = agent_executor
        
        print("--- Models and services initialized. Server is ready. ---")
        print("🌐 Server URL: http://localhost:8080")
        print("📖 API Documentation: http://localhost:8080/docs")
    except Exception as e:
        # If initialization fails, log the error. The server won't start correctly.
        logger.critical(f"FATAL: Server startup failed during initialization: {e}")
        # In a real production scenario, you might want to exit or handle this differently.
    
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
    """
    Root endpoint to check if the server is running.
    """
    return {"message": "Librarian MCP Server running", "status": "healthy"}

@app.post("/process_library", response_model=dict)
async def process_library_endpoint(request: ProcessRequest):
    """
    This endpoint runs the full Librarian pipeline for a given library name.
    """
    library_name = request.library_name
    logger.info(f"Received request to process library: {library_name}")

    try:
        # 1. Check Cache
        cached_data = load_from_cache(library_name)
        if cached_data:
            logger.info(f"Cache hit for '{library_name}'. Returning cached data.")
            return cached_data

        # 2. Find URL with the Agent
        doc_url = find_documentation_url(server_state["agent_executor"], library_name)
        if not doc_url:
            raise HTTPException(status_code=404, detail=f"Agent failed to find a valid URL for '{library_name}'.")

        # 3. Extract Info (this function contains the full ingestion and fallback logic)
        # CORRECTED: Added 'await' to properly call the asynchronous function
        library_info = await extract_structured_info(
            library_name=library_name,
            llm=server_state["llm"],
            embeddings_model=server_state["embeddings_model"],
            doc_url=doc_url
        )

        if not library_info:
            raise HTTPException(status_code=500, detail="Failed to extract structured information after ingestion.")

        # 4. Save to cache ONLY if confidence is high or medium
        info_dict = library_info.model_dump()
        if library_info.confidence_score and library_info.confidence_score.lower() in ["high", "medium"]:
            save_to_cache(library_name, info_dict)
        else:
            logger.warning(f"Skipping cache for '{library_name}' due to low or unknown confidence.")

        return info_dict
    
    except HTTPException as http_exc:
        # Re-raise HTTP exceptions directly
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors and return a proper 500 error
        logger.error(f"An unexpected error occurred in /process_library for '{library_name}': {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@app.post("/ask", response_model=dict)
async def ask_endpoint(request: AskRequest):
    """
    This is the RAG endpoint. It answers a question about a library
    by retrieving relevant context from the Pinecone database.
    """
    library_name = request.library_name
    question = request.question
    logger.info(f"Received question about '{library_name}': '{question}'")

    try:
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=os.getenv("PINECONE_INDEX_NAME", "mcp-documentation-index"),
            embedding=server_state["embeddings_model"]
        )

        doc_id = f"lib-{_sanitize_filename(library_name)}"
        retriever = vectorstore.as_retriever(
            search_kwargs={'k': 5, 'filter': {'doc_id': doc_id}}
        )

        docs = retriever.invoke(question)

        if not docs:
            return {"answer": "I could not find any relevant information in the documentation for that library. It might not have been processed yet."}

        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        
        return {
            "library": library_name,
            "question": question,
            "context": context
        }

    except Exception as e:
        logger.error(f"Error during RAG retrieval for '{library_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve information. Error: {str(e)}")


# --- Run the Server ---
if __name__ == "__main__":
    """
    This block allows you to run the server directly from the command line
    for testing and development.
    """
    print("Starting Librarian MCP Server with Uvicorn...")
    uvicorn.run("mcp_server:app", host="0.0.0.0", port=8000, reload=True)
