import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import google.generativeai as genai
import faiss
import numpy as np
import os
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Web Crawler Component ---

class WebCrawler:
    """
    Crawls a website starting from a specific URL, staying on the same domain.
    """
    def __init__(self, start_url, max_pages=30):
        self.start_url = start_url
        self.base_netloc = urlparse(start_url).netloc
        self.visited_urls = set()
        self.max_pages = max_pages

    def crawl(self):
        """
        A generator that crawls a website and yields cleaned text and URL for each page.
        """
        urls_to_visit = [self.start_url]
        pages_crawled = 0
        
        while urls_to_visit and pages_crawled < self.max_pages:
            url = urls_to_visit.pop(0)
            if url in self.visited_urls:
                continue
            
            print(f"Crawling page {pages_crawled + 1}/{self.max_pages}: {url}")
            self.visited_urls.add(url)
            pages_crawled += 1
            
            try:
                response = requests.get(url, timeout=5)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract clean text
                for script_or_style in soup(["script", "style"]):
                    script_or_style.decompose()
                
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                clean_text = '\n'.join(line for line in lines if line)
                
                if clean_text:
                    yield {"url": url, "text": clean_text}

                # Find new links on the same domain
                for link in soup.find_all('a', href=True):
                    abs_url = urljoin(url, link['href'])
                    parsed_abs = urlparse(abs_url)

                    if parsed_abs.netloc == self.base_netloc and parsed_abs.scheme in ['http', 'https']:
                        if abs_url not in self.visited_urls:
                            urls_to_visit.append(abs_url)

            except requests.RequestException as e:
                print(f"Error crawling {url}: {e}")

# --- Vector Store Component ---

class VectorStore:
    """
    Handles text chunking, embedding generation, and FAISS indexing.
    """
    def __init__(self, chunk_size=1000, chunk_overlap=100):
        # Configure the Gemini API key
        api_key = "******"
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        genai.configure(api_key=api_key)
        
        self.embedding_model = 'models/text-embedding-004'
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.index = None
        self.documents = []  # Stores metadata {url, snippet}

    def build_index(self, pages_generator):
        """
        Builds the FAISS index from the text content provided by the crawler.
        """
        for page in pages_generator:
            chunks = self.text_splitter.create_documents([page['text']])
            for chunk in chunks:
                self.documents.append({
                    "url": page['url'],
                    "snippet": chunk.page_content
                })

        all_snippets = [doc['snippet'] for doc in self.documents]
        if not all_snippets:
            print("No text content found to index.")
            return

        print(f"Generating embeddings for {len(all_snippets)} chunks...")
        result = genai.embed_content(model=self.embedding_model, content=all_snippets, task_type="retrieval_document")
        embeddings = np.array(result['embedding'], dtype='float32')
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        print(f"Successfully indexed {self.index.ntotal} chunks into FAISS.")

    def search(self, query, k=5):
        """
        Searches the FAISS index for the top k most relevant document chunks.
        """
        if self.index is None:
            return [], 0
        
        start_time = time.time()
        query_embedding_result = genai.embed_content(
            model=self.embedding_model,
            content=query,
            task_type="retrieval_query"
        )
        query_embedding = np.array([query_embedding_result['embedding']], dtype='float32')
        
        distances, indices = self.index.search(query_embedding, k)
        end_time = time.time()
        
        retrieval_time = end_time - start_time
        results = [self.documents[i] for i in indices[0]]
        return results, retrieval_time

# --- QA Engine Component ---

class QAEngine:
    """
    Orchestrates the retrieval and generation process to answer questions.
    """
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.qa_model = genai.GenerativeModel('gemini-2.0-flash')

    def answer_question(self, query):
        """
        Retrieves context, constructs a prompt, and generates a grounded answer.
        """
        context_docs, retrieval_time = self.vector_store.search(query)
        
        if not context_docs:
            return {
                "answer": "I could not find any information related to your question on the crawled website.",
                "sources": [],
                "retrieval_time": retrieval_time
            }

        context_str = "\n---\n".join([f"Source URL: {doc['url']}\nContent: {doc['snippet']}" for doc in context_docs])
        
        prompt = f"""
        You are a helpful AI assistant. Your task is to answer the user's question based *strictly* on the provided context below.

        **Context:**
        {context_str}

        **Instructions:**
        1.  Analyze the context carefully.
        2.  Answer the user's question using only information found in the context.
        3.  Do not use any external knowledge or make assumptions.
        4.  If the answer cannot be found in the context, you MUST respond with: "I cannot answer this question based on the provided information."
        5.  Cite the source URL(s) you used to formulate the answer.

        **Question:** {query}

        **Answer:**
        """

        try:
            response = self.qa_model.generate_content(prompt)
            return {
                "answer": response.text,
                "sources": context_docs,
                "retrieval_time": retrieval_time
            }
        except Exception as e:
            print(f"Error during generation: {e}")
            return {
                "answer": f"An error occurred: {e}",
                "sources": [],
                "retrieval_time": retrieval_time
            }


