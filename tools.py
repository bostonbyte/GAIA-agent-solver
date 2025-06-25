# tools.py

import os
import requests
import PyPDF2
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import uuid

# Imports for Tool classes and loaders
from smolagents.tools import Tool
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.document_loaders import WikipediaLoader, ArxivLoader

# --- NEW Imports for the Retriever Tool ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

class WebSearchTool(Tool):
    name = "web_search"
    description = "Searches the web and returns a list of results."
    inputs = {"query": {"type": "string", "description": "The search query to execute."}}
    output_type = "string"

    def forward(self, query: str) -> str:
        # ... (implementation unchanged)
        print(f"--- Executing Web Search with query: '{query}' ---")
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
            if not results:
                return "No results found for the query."
            formatted_results = [
                f"Result {i+1}:\n  Title: {r['title']}\n  URL: {r['href']}\n  Snippet: {r['body']}"
                for i, r in enumerate(results)
            ]
            return "\n\n".join(formatted_results)
        except Exception as e:
            return f"An error occurred during web search: {e}"

class ScrapeWebpageTool(Tool):
    name = "scrape_web_page"
    description = "Extracts the full, clean text content from a web page URL."
    inputs = {"url": {"type": "string", "description": "The URL of the web page to scrape."}}
    output_type = "string"

    def forward(self, url: str) -> str:
        # ... (same logic, but we REMOVED the truncation) ...
        print(f"--- Scraping Web Page at URL: {url} ---")
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            for script_or_style in soup(['script', 'style']):
                script_or_style.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            return '\n'.join(chunk for chunk in chunks if chunk) or "The page has no text content."
        except Exception as e:
            return f"An error occurred while scraping the URL: {e}"

class DownloadFileTool(Tool):
    name = "download_file"
    description = "Downloads a file from a URL (e.g., a PDF) and returns its local file path."
    inputs = {"url": {"type": "string", "description": "The URL of the file to download."}}
    output_type = "string"

    def forward(self, url: str) -> str:
        # ... (implementation unchanged)
        print(f"--- Downloading file from URL: {url} ---")
        try:
            if not os.path.exists("temp_files"):
                os.makedirs("temp_files")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            file_name = str(uuid.uuid4())
            local_path = os.path.join("temp_files", file_name)
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return f"File successfully downloaded and saved to: {local_path}"
        except Exception as e:
            return f"An error occurred while downloading the file: {e}"

class ReadPDFTool(Tool):
    name = "read_pdf_file"
    description = "Reads and returns the text content from a local PDF file."
    inputs = {"file_path": {"type": "string", "description": "The local path to the PDF file."}}
    output_type = "string"

    def forward(self, file_path: str) -> str:
        # ... (implementation unchanged)
        print(f"--- Reading PDF file at path: {file_path} ---")
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
            return text or "No text could be extracted from the PDF."
        except FileNotFoundError:
            return f"Error: The file at {file_path} was not found."
        except Exception as e:
            return f"An error occurred reading the PDF: {e}"

class ReadTextFileTool(Tool):
    name = "read_text_file"
    description = "Reads and returns the content from a local plain text file."
    inputs = {"file_path": {"type": "string", "description": "The local path to the text file."}}
    output_type = "string"
    
    def forward(self, file_path: str) -> str:
        # ... (implementation unchanged)
        print(f"--- Reading text file at path: {file_path} ---")
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            return f"Error: The file at {file_path} was not found."
        except Exception as e:
            return f"An error occurred reading the text file: {e}"

class YouTubeTranscriptionTool(Tool):
    name = "youtube_transcription_tool"
    description = "This tool returns the full text transcript of a YouTube video."
    inputs = {"url": {"type": "string", "description": "The full URL of the YouTube video."}}
    output_type = "string"

    def forward(self, url: str) -> str:
        # ... (implementation unchanged)
        try:
            proxy_user = os.getenv("proxy_username")
            proxy_pass = os.getenv("proxy_password")
            
            proxies = None
            if proxy_user and proxy_pass:
                proxy_url = f"http://{proxy_user}:{proxy_pass}@p.webshare.io:80"
                proxies = {"http": proxy_url, "https": proxy_url}

            if "v=" not in url:
                return "Error: Invalid YouTube URL. Must contain 'v='."
            
            video_id = url.split("v=")[-1].split("&")[0]
            
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, proxies=proxies)
            
            return " ".join([d['text'] for d in transcript_list])
        except Exception as e:
            return f"Error fetching transcript: {e}"

class WikiSearchTool(Tool):
    name = "wiki_search"
    description = "Search Wikipedia for a query and return the content of up to 2 results."
    inputs = {"query": {"type": "string", "description": "The search query for Wikipedia."}}
    output_type = "string"

    def forward(self, query: str) -> str:
        # ... (implementation unchanged)
        search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
        return "\n\n---\n\n".join([doc.page_content for doc in search_docs])

class ArxivSearchTool(Tool):
    name = "arvix_search"
    description = "Search Arxiv for a query and return the content of up to 2 results."
    inputs = {"query": {"type": "string", "description": "The search query for Arxiv."}}
    output_type = "string"

    def forward(self, query: str) -> str:
        # ... (implementation unchanged)
        search_docs = ArxivLoader(query=query, load_max_docs=2).load()
        return "\n\n---\n\n".join(
            [f"Title: {doc.metadata.get('Title', 'N/A')}\nSummary: {doc.page_content}" for doc in search_docs]
        )

class SemanticSearchTool(Tool):
    name = "semantic_search"
    description = "Performs a semantic search for a query within a large body of text. Use this to find specific information inside a document or large text that you have already retrieved."
    inputs = {
        "text_to_search": {"type": "string", "description": "The large block of text to search within."},
        "query": {"type": "string", "description": "The specific question or query to find information about."}
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        # Initialize components needed for the retriever
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def forward(self, text_to_search: str, query: str) -> str:
        print(f"--- Performing semantic search for query: '{query[:50]}...' ---")
        try:
            # 1. Split the large text into chunks
            docs = self.text_splitter.create_documents([text_to_search])
            
            # 2. Create a temporary, in-memory vector store on the fly
            db = FAISS.from_documents(docs, self.embedding_model)

            # 3. Perform the search and get the most relevant chunks
            results = db.similarity_search(query, k=3) # Get top 3 chunks

            if not results:
                return "No relevant information found for that query in the text."

            # 4. Return the combined content of the relevant chunks
            return "\n\n---\n\n".join([doc.page_content for doc in results])
        except Exception as e:
            return f"An error occurred during semantic search: {e}"


# --- NEW MATH TOOLS ---

class MultiplyTool(Tool):
    name = "multiply"
    description = "Multiply two numbers."
    inputs = {
        "a": {"type": "number", "description": "First number"},
        "b": {"type": "number", "description": "Second number"}
    }
    output_type = "number"

    def forward(self, a: int, b: int) -> int:
        return a * b

class AddTool(Tool):
    name = "add"
    description = "Add two numbers."
    inputs = {
        "a": {"type": "number", "description": "First number"},
        "b": {"type": "number", "description": "Second number"}
    }
    output_type = "number"

    def forward(self, a: int, b: int) -> int:
        return a + b

class SubtractTool(Tool):
    name = "subtract"
    description = "Subtract two numbers."
    inputs = {
        "a": {"type": "number", "description": "First number"},
        "b": {"type": "number", "description": "Second number"}
    }
    output_type = "number"

    def forward(self, a: int, b: int) -> int:
        return a - b

class DivideTool(Tool):
    name = "divide"
    description = "Divide two numbers."
    inputs = {
        "a": {"type": "number", "description": "Numerator"},
        "b": {"type": "number", "description": "Denominator"}
    }
    output_type = "number"

    def forward(self, a: int, b: int) -> float:
        if b == 0:
            return float('inf') # Return infinity for division by zero
        return a / b

class ModulusTool(Tool):
    name = "modulus"
    description = "Get the modulus of two numbers."
    inputs = {
        "a": {"type": "number", "description": "First number"},
        "b": {"type": "number", "description": "Second number"}
    }
    output_type = "number"

    def forward(self, a: int, b: int) -> int:
        return a % b
