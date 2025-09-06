# --- SSL Certificate Configuration ---
# This MUST be at the top, before any library that uses httpx/requests is imported.
import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
# --- End SSL Configuration ---

# --- MCP Server: Services Module (Async-Aware) ---
# This file contains the core logic of the agent. Key data pipeline functions
# have been updated to be fully asynchronous to support async libraries like crawl4ai.

import os
import getpass
import time
import json
import requests
import re
import logging
import asyncio
from typing import Optional, List
from datetime import datetime, timedelta

# --- Core LangChain & Pydantic Imports ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field

import certifi
import httpx

# --- Modern, Modular LangChain Integration Imports ---
from langchain_groq import ChatGroq
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_community.tools import TavilySearchResults

# --- Third-party imports for fallback search ---
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy


# --- Global Configuration ---
PINECONE_INDEX_NAME = "mcp-documentation-index-groq"
JINA_READER_API_URL = "https://r.jina.ai/"
MIN_CONTENT_LENGTH_THRESHOLD = 500

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Pydantic Schema for Rich Data Extraction ---
class LibraryInfo(BaseModel):
    """A structured representation of key information about a software library."""
    library_name: Optional[str] = Field(default=None, description="The official name of the library (e.g., 'scikit-learn')")
    package_name: Optional[str] = Field(default=None, description="The correct, current package name for installation (e.g., 'scikit-learn')")
    latest_version: Optional[str] = Field(default=None, description="The latest stable version number found (e.g., '1.5.0')")
    documentation_url: Optional[str] = Field(default=None, description="The direct URL to the main documentation page that was processed.")
    summary: Optional[str] = Field(default=None, description="A brief, one-sentence summary of the library's primary purpose.")
    installation_command: Optional[str] = Field(default=None, description="The standard installation command (e.g., 'pip install scikit-learn')")
    deprecation_notice: Optional[str] = Field(default=None, description="A notice if the original library name is deprecated or has been renamed.")
    ecosystem: Optional[str] = Field(default=None, description="The software ecosystem the library belongs to (e.g., 'Python/PyPI', 'JavaScript/npm')")
    confidence_score: Optional[str] = Field(default=None, description="Self-evaluated confidence level: 'High', 'Medium', or 'Low'.")
    additional_insights: Optional[str] = Field(default=None, description="Important information discovered during low-confidence rescraping.")

# --- Service Initialization, Caching, Confidence Score ---

def initialize_services():
    """Initializes and returns the core AI and database clients."""
    
    # Load API key from groq_api.py
    try:
        import groq_api
    except ImportError:
        raise ValueError("groq_api.py not found. Please ensure groq_api.py exists in this directory.")

    api_key = getattr(groq_api, "GROQ_API_KEY", None)
    if not api_key or "PASTE_YOUR_GROQ_API_KEY_HERE" in api_key:
        raise ValueError("GROQ_API_KEY not set in groq_api.py or still contains placeholder.")
    groq_api_key = api_key.strip()

    # Explicitly pass the API key to the constructor
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        temperature=0,
        model_name="llama-3.1-8b-instant"
    )
    
    # Specify a local cache directory for the embedding model and use a 768-dim model
    cache_dir = os.path.join(os.path.dirname(__file__), ".embeddings_cache")
    os.makedirs(cache_dir, exist_ok=True)
    embeddings_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5", cache_dir=cache_dir)
    
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY environment variable not set.")
    pc = Pinecone(api_key=pinecone_api_key)
    
    return llm, embeddings_model, pc

def _sanitize_filename(name: str) -> str:
    """Replaces characters that are invalid in file names."""
    return re.sub(r'[\\/*?:"<>|]', "_", name)

def save_to_cache(library_name: str, library_data):
    """Saves library data to a local cache file with timestamp."""
    cache_dir = "cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    safe_filename = _sanitize_filename(library_name)
    cache_file = os.path.join(cache_dir, f"{safe_filename}_cache.json")

    if hasattr(library_data, 'model_dump'):
        data_dict = library_data.model_dump()
    else:
        data_dict = library_data

    cache_entry = {
        "timestamp": datetime.now().isoformat(),
        "data": data_dict
    }

    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_entry, f, indent=2)
        logger.info(f"💾 Cached data for '{library_name}'")
    except Exception as e:
        logger.error(f"⚠️ Failed to cache data for '{library_name}': {e}")

def load_from_cache(library_name: str, max_age_hours: int = 24):
    """Loads library data from cache if it exists and is not expired."""
    cache_dir = "cache"
    safe_filename = _sanitize_filename(library_name)
    cache_file = os.path.join(cache_dir, f"{safe_filename}_cache.json")

    if not os.path.exists(cache_file):
        return None

    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_entry = json.load(f)

        cache_time = datetime.fromisoformat(cache_entry["timestamp"])
        if datetime.now() - cache_time > timedelta(hours=max_age_hours):
            logger.warning(f"⚠️ Cache expired for '{library_name}' (older than {max_age_hours} hours)")
            return None

        logger.info(f"📋 Cache hit for '{library_name}' (age: {datetime.now() - cache_time})")
        return cache_entry["data"]
    except Exception as e:
        logger.error(f"⚠️ Failed to load cache for '{library_name}': {e}")
        return None

def interpret_confidence_score(confidence_score: str) -> str:
    """Provides a human-readable interpretation of the confidence score."""
    if not confidence_score or not confidence_score.strip():
        return "❓ Unknown Confidence Level"

    confidence_lower = confidence_score.lower()
    if confidence_lower == "high":
        return "✅ High Confidence - Data is comprehensive and reliable"
    elif confidence_lower == "medium":
        return "⚠️ Medium Confidence - Data is good but may have some uncertainties"
    elif confidence_lower == "low":
        return "❌ Low Confidence - Data is limited or potentially unreliable"
    else:
        return f"❓ Unknown Confidence Level: {confidence_score}"


# --- Specialist Agent Tools ---
@tool
def pypi_api_tool(package_name: str) -> str:
    """Queries the official PyPI API for a Python package's metadata."""
    logger.info(f"-> 🛠️ Using PyPI API Tool for '{package_name}'...")
    try:
        response = requests.get(f"https://pypi.org/pypi/{package_name}/json")
        response.raise_for_status()
        info = response.json().get("info", {})
        urls = info.get("project_urls", {})
        doc_url = urls.get("Homepage") or urls.get("Documentation") or next((url for url in urls.values() if url), None)
        return f"Success! Found documentation URL: {doc_url}" if doc_url else "PyPI API found the package but no documentation URL."
    except requests.exceptions.HTTPError as e:
        return f"Error: Package '{package_name}' not found on PyPI." if e.response.status_code == 404 else f"Error: HTTP error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

@tool
def npm_api_tool(package_name: str) -> str:
    """Queries the official npm registry API for a JavaScript/Node.js package's metadata."""
    logger.info(f"-> 🛠️ Using npm API Tool for '{package_name}'...")
    try:
        response = requests.get(f"https://registry.npmjs.org/{package_name}")
        response.raise_for_status()
        data = response.json()
        doc_url = data.get("homepage") or (data.get("bugs", {}).get("url"))
        return f"Success! Found documentation URL: {doc_url}" if doc_url else "npm API found the package but no documentation URL."
    except requests.exceptions.HTTPError as e:
        return f"Error: Package '{package_name}' not found on npm." if e.response.status_code == 404 else f"Error: HTTP error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

@tool
def crates_io_api_tool(package_name: str) -> str:
    """Queries the official Crates.io API for a Rust package's metadata."""
    logger.info(f"-> 🛠️ Using Crates.io API Tool for '{package_name}'...")
    try:
        response = requests.get(f"https://crates.io/api/v1/crates/{package_name}")
        response.raise_for_status()
        data = response.json()
        doc_url = data.get("crate", {}).get("documentation")
        return f"Success! Found documentation URL: {doc_url}" if doc_url else "Crates.io API found the package but no documentation URL."
    except requests.exceptions.HTTPError as e:
        return f"Error: Package '{package_name}' not found on Crates.io." if e.response.status_code == 404 else f"Error: HTTP error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

@tool
def web_search_tool(query: str) -> str:
    """Performs a web search using DuckDuckGo first, then falls back to Tavily AI."""
    logger.info(f"-> 🛠️ Using Web Search Tool for '{query}'...")
    try:
        logger.info("  -> 🔍 Attempting DuckDuckGo search...")
        wrapper = DuckDuckGoSearchAPIWrapper(time="y", max_results=5, region="us-en")
        ddg_results = wrapper.run(query)
        if ddg_results and 'http' in ddg_results:
            logger.info("  -> ✅ DuckDuckGo search successful!")
            return ddg_results
        logger.warning("  -> ⚠️ DuckDuckGo found no valid URLs, falling back to Tavily AI...")
    except Exception as e:
        logger.error(f"  -> ❌ DuckDuckGo search failed: {e}, falling back to Tavily AI...")

    try:
        if not os.getenv("TAVILY_API_KEY"):
            return "Error: Tavily API key not found. Please set TAVILY_API_KEY."
        logger.info("  -> 🤖 Using LangChain-aware Tavily AI as fallback...")
        tavily_tool = TavilySearchResults(max_results=5)
        results = tavily_tool.invoke({"query": query})
        logger.info("  -> ✅ Tavily AI search successful!")
        return str(results)
    except Exception as e:
        logger.error(f"  -> ❌ Tavily AI search also failed: {e}")
        return f"Both DuckDuckGo and Tavily AI searches failed. Error: {e}"

def create_universal_agent(llm):
    """Creates the multi-ecosystem agent with the most robust reasoning process."""
    tools = [pypi_api_tool, npm_api_tool, crates_io_api_tool, web_search_tool]
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert documentation research assistant. Your goal is to find the single, best, official documentation URL for a given software library. "
         "You must follow this exact strategy:\n"
         "1. **Analyze and Refine the Query:** Before using any tools, analyze the input library name. Is it a common alias or abbreviation? For example, if the input is 'sklearn', you must recognize the official name is 'scikit-learn' and use 'scikit-learn' for all subsequent steps.\n"
         "2. **Use Specialist Tools First:** Use the refined, official name with the appropriate specialist tool (e.g., pypi_api_tool for 'scikit-learn'). This is your preferred first action.\n"
         "3. **Fallback to Web Search:** If the specialist tool fails or returns no URL, you must then use the web_search_tool with the refined, official name.\n"
         "4. **Filter by Language:** You MUST ignore and discard any search results from non-English domains (e.g., .cn, .jp, .ru, .de). Only consider English-language results.\n"
         "5. **Self-Correct on Failure:** If your first web search with the refined name fails, do not give up. Try one more time with a broader query. For example, if a search for 'google photos picker api' fails, try a new search for 'google photos api developer documentation'."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


# --- Data Pipeline Functions ---

def ensure_pinecone_index_ready(pc: Pinecone, embeddings_model):
    """Checks if the Pinecone index exists. If not, creates it and waits until it is ready."""
    logger.info(f"Ensuring Pinecone index '{PINECONE_INDEX_NAME}' is ready...")
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        logger.info(f"Index not found. Creating new index '{PINECONE_INDEX_NAME}'...")
        try:
            dimension = 384  # Hardcoded dimension to match BAAI/bge-small-en-v1.5 embeddings
            pc.create_index(
                name=PINECONE_INDEX_NAME, dimension=dimension, metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
                logger.info("Waiting for index to become ready...")
                time.sleep(5)
            logger.info("✅ Index is ready.")
        except Exception as e:
            logger.error(f"❌ Error creating Pinecone index: {e}")
            raise
    else:
        logger.info("✅ Index already exists and is ready.")

def _fetch_with_jina(url: str) -> Optional[str]:
    """Fetches clean content using the Jina AI Reader API."""
    try:
        api_url = f"{JINA_READER_API_URL}{url}"
        logger.info(f"🧼 [Primary] Attempting to fetch content with Jina AI from: {url}")
        response = requests.get(api_url, timeout=30, headers={"User-Agent": "LibrarianAI/1.0"})
        response.raise_for_status()
        logger.info("✅ [Primary] Successfully fetched content with Jina AI.")
        return response.text
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ [Primary] Jina AI API request failed: {e}")
        return None

async def _deep_crawl_with_crawl4ai(url: str) -> Optional[str]:
    """
    Performs a deep crawl starting from the given URL to gather rich context.
    """
    try:
        logger.info(f"🤖 [Deep Crawl] Initiating deep crawl for: {url}")
        strategy = BFSDeepCrawlStrategy(max_depth=1)
        run_config = CrawlerRunConfig(
            deep_crawl_strategy=strategy,
            markdown_generator=DefaultMarkdownGenerator()
        )
        all_content = []
        async with AsyncWebCrawler() as crawler:
            results = await crawler.arun(url=url, config=run_config)
            if results:
                for result in results:
                    if result.success and result.markdown:
                        logger.info(f"  -> Crawled and extracted content from: {result.url}")
                        all_content.append(str(result.markdown))
        
        if not all_content:
            logger.error("❌ Deep crawl finished but found no markdown content.")
            return None
        
        logger.info(f"✅ Deep crawl successful. Aggregated content from {len(all_content)} pages.")
        return "\n\n--- (New Page Content) ---\n\n".join(all_content)
    except Exception as e:
        logger.error(f"❌ Deep crawl failed with a critical error: {e}")
        return None

async def get_clean_content(doc_url: str) -> Optional[str]:
    """
    Hybrid content fetcher. First tries Jina AI, then falls back to a deep
    crawl with Crawl4ai if Jina fails.
    """
    jina_content = _fetch_with_jina(doc_url)
    if jina_content and len(jina_content) >= MIN_CONTENT_LENGTH_THRESHOLD:
        logger.info("Jina AI content is valid and of sufficient length.")
        return jina_content

    logger.warning("Jina API failed or content was too short. Triggering deep crawl fallback.")
    return await _deep_crawl_with_crawl4ai(doc_url)

async def ingest_documentation(library_name: str, doc_url: str, embeddings_model, content_to_ingest: Optional[str] = None):
    """
    Deletes old vectors for the library, then gets clean content and stores the new vectors.
    """
    logger.info(f"\n📚 Starting ingestion for '{library_name}'...")
    doc_id = f"lib-{_sanitize_filename(library_name)}"
    
    try:
        logger.info(f"🧹 Deleting old vectors with doc_id: '{doc_id}'...")
        vectorstore = PineconeVectorStore.from_existing_index(PINECONE_INDEX_NAME, embeddings_model)
        vectorstore.delete(filter={"doc_id": doc_id})
        logger.info("✅ Old vectors deleted successfully.")
    except Exception as e:
        # Check if it's a "Namespace not found" error (Pinecone error code 5)
        if "Namespace not found" in str(e) or "code" in str(e) and "5" in str(e):
            logger.info("ℹ️ No existing vectors found (namespace doesn't exist yet). Skipping deletion.")
        else:
            logger.warning(f"⚠️ Could not delete old vectors: {e}")

    # Use pre-fetched content if provided, otherwise fetch it.
    if content_to_ingest is None:
        content_string = await get_clean_content(doc_url)
    else:
        logger.info("Using pre-fetched content for ingestion.")
        content_string = content_to_ingest

    if not content_string:
        logger.error("❌ Halting ingestion as no content was provided or could be fetched.")
        return False

    try:
        base_doc = Document(page_content=content_string, metadata={"source": doc_url})
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents([base_doc])
        
        for split in splits:
            split.metadata["doc_id"] = doc_id

        logger.info(f"Embedding {len(splits)} new document chunks and storing in Pinecone...")
        await PineconeVectorStore.afrom_documents(splits, embeddings_model, index_name=PINECONE_INDEX_NAME)
        logger.info("✅ New content ingested successfully.")
        return True
    except Exception as e:
        logger.error(f"❌ An error occurred during chunking or embedding: {e}")
        return False

async def extract_structured_info(library_name: str, llm, embeddings_model, doc_url: str) -> LibraryInfo:
    """
    Retrieves context from Pinecone and uses an LLM to extract rich, structured information.
    Implements a two-stage fallback for low-confidence results.
    """
    logger.info(f"\n⛏️ Starting structured info extraction for '{library_name}'...")

    try:
        # --- Stage 1: Quick Pass with Initial Content ---
        logger.info("--- Stage 1: Performing initial RAG extraction ---")
        initial_content = await get_clean_content(doc_url)
        if not initial_content:
            logger.error("Initial content fetch failed. Cannot proceed with extraction.")
            return LibraryInfo(library_name=library_name, confidence_score="Low", additional_insights="Initial content scraping failed.")

        ingestion_success = await ingest_documentation(library_name, doc_url, embeddings_model, content_to_ingest=initial_content)
        if not ingestion_success:
             return LibraryInfo(library_name=library_name, confidence_score="Low", additional_insights="Initial content ingestion failed.")


        vectorstore = PineconeVectorStore.from_existing_index(PINECONE_INDEX_NAME, embeddings_model)
        retriever = vectorstore.as_retriever(search_kwargs={'k': 8})
        
        docs = await retriever.ainvoke(f"Information about {library_name} including purpose, installation, version, and deprecation notices.")
        context_text = "\n\n".join([doc.page_content for doc in docs])

        if not context_text:
            logger.warning(f"⚠️ Could not retrieve any context for '{library_name}' from Pinecone.")
            return LibraryInfo(library_name=library_name, confidence_score="Low", documentation_url=doc_url)

        structured_llm = llm.with_structured_output(LibraryInfo)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at extracting key information..."),
            ("human", "Extract the required information about '{topic}'... Context:\n{context}")
        ])

        chain = prompt | structured_llm
        initial_response = await chain.ainvoke({"topic": library_name, "context": context_text})

        if initial_response and doc_url:
            initial_response.documentation_url = doc_url
        
        # --- Stage 2: Confidence Check and Deep Crawl Fallback ---
        is_confident = initial_response.confidence_score and initial_response.confidence_score.lower() in ["high", "medium"]

        if not is_confident:
            logger.warning(f"⚠️ Low or Unknown confidence for '{library_name}'. Triggering deep crawl fallback...")
            
            # This is where the deep crawl was intended but the logic was flawed.
            # We now perform the deep crawl and then re-run the extraction.
            rich_content = await _deep_crawl_with_crawl4ai(doc_url)
            
            if rich_content:
                logger.info("--- Stage 2: Re-ingesting with richer content from deep crawl ---")
                reingestion_success = await ingest_documentation(library_name, doc_url, embeddings_model, content_to_ingest=rich_content)
                
                if reingestion_success:
                    logger.info("--- Stage 2: Re-running structured extraction with new context ---")
                    # Retrieve the new, richer context
                    new_docs = await retriever.ainvoke(f"Information about {library_name} including purpose, installation, and version.")
                    new_context_text = "\n\n".join([doc.page_content for doc in new_docs])
                    
                    # Re-run the extraction chain
                    final_response = await chain.ainvoke({"topic": library_name, "context": new_context_text})
                    
                    if final_response:
                        final_response.documentation_url = doc_url
                        final_response.additional_insights = (
                            "Low confidence triggered a deep crawl to improve data quality.\n"
                            + (final_response.additional_insights or "")
                        )
                        logger.info("✅ Structured information extracted successfully after deep crawl.")
                        return final_response
            
            logger.error("❌ Deep crawl fallback failed to produce better results. Returning initial low-confidence data.")
            return initial_response # Return the first result if deep crawl fails

        logger.info("✅ Structured information extracted successfully on the first pass.")
        return initial_response

    except Exception as e:
        logger.error(f"❌ A critical error occurred during structured data extraction: {e}")
        return LibraryInfo(
            library_name=library_name,
            documentation_url=doc_url,
            confidence_score="Low",
            additional_insights=f"A critical error occurred: {str(e)}"
        )

def find_documentation_url(agent_executor, library_name: str) -> str | None:
    """
    Uses the agent to find the official documentation URL for a library.
    """
    logger.info(f"\n🕵️ Agent is searching for documentation for '{library_name}'...")
    try:
        # Agent invocation is synchronous
        response = agent_executor.invoke({"input": f"Find the documentation URL for the {library_name} library."})
        output = response.get("output", "")
        
        url_match = re.search(r'https?://[^\s,"]+', output)
        if url_match:
            doc_url = url_match.group(0).strip().strip(' ./\t\n')
            logger.info(f"✅ Agent found URL: {doc_url}")
            return doc_url
        else:
            logger.error(f"❌ Agent failed to find a valid URL in its output: {output}")
            return None
            
    except Exception as e:
        logger.error(f"❌ An error occurred while the agent was searching: {e}")
        return None
