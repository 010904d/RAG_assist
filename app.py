from flask import Flask, render_template, request, jsonify
from backend_app import WebCrawler, VectorStore, QAEngine
import threading

app = Flask(__name__)

# --- In-memory Application State ---
# In a production app, use a database or a more robust solution
CRAWL_STATE = {
    "crawled_pages": [], 
    "is_crawling": False,
    "error": None
}
QA_SYSTEM = None

@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/start-crawl', methods=['POST'])
def start_crawl():
    """Starts the web crawling and indexing process in a background thread."""
    global QA_SYSTEM, CRAWL_STATE
    
    start_url = request.json.get('url')
    if not start_url:
        return jsonify({"error": "URL is required"}), 400

    if CRAWL_STATE["is_crawling"]:
        return jsonify({"error": "A crawl is already in progress"}), 400

    # Reset state for a new crawl
    CRAWL_STATE = {"crawled_pages": [], "is_crawling": True, "error": None}
    QA_SYSTEM = None

    def crawl_and_index_task():
        """The target function for the background thread."""
        global QA_SYSTEM, CRAWL_STATE
        try:
            # The WebCrawler is now initialized with a max_pages limit.
            crawler = WebCrawler(start_url, max_pages=30)
            vector_store = VectorStore()
            
            # This generator allows us to update the state as pages are crawled
            def page_generator():
                for page in crawler.crawl():
                    CRAWL_STATE["crawled_pages"].append(page['url'])
                    yield page

            vector_store.build_index(page_generator())
            QA_SYSTEM = QAEngine(vector_store)
        except Exception as e:
            CRAWL_STATE["error"] = str(e)
            print(f"An error occurred during crawl/index: {e}")
        finally:
            CRAWL_STATE["is_crawling"] = False

    # Run the task in a separate thread to keep the web server responsive
    thread = threading.Thread(target=crawl_and_index_task)
    thread.start()

    return jsonify({"message": "Crawl initiated."})

@app.route('/crawl-status')
def crawl_status():
    """Allows the frontend to poll for the current status of the crawl."""
    return jsonify(CRAWL_STATE)

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handles a user's question by querying the QA system."""
    if CRAWL_STATE["is_crawling"]:
        return jsonify({"error": "Please wait for crawling and indexing to complete."}), 400
    if not QA_SYSTEM:
        return jsonify({"error": "System not ready. Please start a crawl first."}), 400
        
    question = request.json.get('question')
    if not question:
        return jsonify({"error": "A question is required."}), 400

    result = QA_SYSTEM.answer_question(question)
    
    # For observability, log the sources used for this query
    print(f"Query: '{question}' used sources: {[s['url'] for s in result.get('sources', [])]}")
    
    return jsonify(result)

if __name__ == '__main__':
    # Make sure to set your GOOGLE_API_KEY as an environment variable
    # before running this script.
    app.run(debug=True, port=5001)

