# --- MCP Server: Main Orchestrator (Async) ---
# This file is the entry point for the application. It has been updated
# to use Python's asyncio to correctly manage the asynchronous crawling
# and data processing pipeline.

import os
import json
import re
import asyncio # Import the asyncio library
import sys
import traceback # Import the traceback module
from dotenv import load_dotenv

# Fix for Windows asyncio + Playwright compatibility
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# --- Local Imports from our services module ---
from services import (
    create_universal_agent,
    ensure_pinecone_index_ready,
    extract_structured_info,
    initialize_services,
    save_to_cache,
    load_from_cache,
    interpret_confidence_score,
    find_documentation_url # Re-added for clarity in the main loop
)

# The main function is now an async function
async def main():
    """Main async function to orchestrate the full documentation pipeline."""
    load_dotenv()

    # --- Explicitly check and verify the loaded API key ---
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        print(f"✅ GROQ API Key loaded. Starts with: '{groq_api_key[:5]}', ends with: '{groq_api_key[-4:]}'.")
        print("   Please verify this matches the key in your Groq dashboard.")
    else:
        print("❌ FATAL: GROQ_API_KEY not found. The .env file might be missing or in the wrong directory.")
        return # Exit if no key is found
    
    try:
        llm, embeddings_model, pc = initialize_services()
    except Exception as e:
        print(f"❌ Failed to initialize core services: {e}")
        print("\n--- Full Error Traceback ---")
        traceback.print_exc() # Print the full traceback
        print("--------------------------\n")
        return

    # --- Libraries to test across different ecosystems ---
    libraries_to_process = ["requests"]

    ensure_pinecone_index_ready(pc, embeddings_model)
    agent_executor = create_universal_agent(llm)

    for target_library in libraries_to_process:
        print("\n" + "=" * 60)
        print(f"🚀 Starting MCP Server Pipeline for: '{target_library}'")
        print("=" * 60)

        cached_data = load_from_cache(target_library)
        if cached_data:
            print("✅ Cache Hit! Skipping agent and using cached data.")
            library_data_dict = cached_data
        else:
            print("⚠️ Cache Miss. Deploying agent to find documentation URL...")
            
            # Use the dedicated function to run the agent and get the URL
            doc_url = find_documentation_url(agent_executor, target_library)
            
            if not doc_url:
                print(f"\n❌ Halting pipeline: Agent failed to find a valid URL for '{target_library}'.")
                continue
            
            # CORRECTED: Use 'await' for the async extract_structured_info function
            # This function now handles ingestion internally.
            library_data_model = await extract_structured_info(target_library, llm, embeddings_model, doc_url)

            if library_data_model:
                save_to_cache(target_library, library_data_model)
                library_data_dict = library_data_model.model_dump()
            else:
                library_data_dict = None

        if library_data_dict:
            print("\n" + "-"*60)
            print(f"📊 Final Data for '{target_library}':")
            
            confidence = library_data_dict.get('confidence_score')
            insights = library_data_dict.get('additional_insights')

            if confidence:
                print(f"\n🔍 Confidence Assessment: {interpret_confidence_score(confidence)}")
            if insights:
                print(f"\n💡 Additional Insights:\n{insights}")

            print("\n--- Full JSON Output ---")
            print(json.dumps(library_data_dict, indent=2))
            print("-" * 60)
        else:
            print(f"\n❌ Failed to get any data for '{target_library}'.")


if __name__ == "__main__":
    # CORRECTED: Use asyncio.run() to start the async main function
    # This is how you run the script from the command line for testing.
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "cannot run in a running event loop" in str(e):
            print("ERROR: This script is being run in an environment that already has an event loop (like Jupyter or a web server).")
            print("In that case, you should call 'await main()' from an async cell or function.")
        else:
            raise
