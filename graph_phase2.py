import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Optional, Literal
from langchain.prompts import PromptTemplate
import pickle  # For simple caching
import asyncio

# Import fetch function (assume oauth_fetch.py is in the same directory)
from oauth_fetch import fetch_unread_emails

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0)

# Simple cache: dict stored in file (load/save by thread_id)
CACHE_FILE = "summary_cache.pkl"
def load_cache():
    try:
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}

def save_cache(cache):
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)

# State: What gets passed between nodes (expandable later)
class EmailState(TypedDict):
    thread_id: str
    subject: str
    body: str
    summary: Optional[str]
    label: Optional[str]
    confidence: Optional[float]

# Summarize node: 1-2 sentence summary + key entities; check cache first
async def summarize_email(state: dict) -> dict:
    cache = load_cache()
    if state['thread_id'] in cache:
        state['summary'] = cache[state['thread_id']]
        print(f"Cache hit for thread {state['thread_id']}")
        return state
    
    prompt = PromptTemplate.from_template(
        "Summarize this email in 1-2 sentences, extracting key entities (names, dates, topics): {body}"
    )
    response = await llm.ainvoke(prompt.invoke({"body": state['body']}))
    state['summary'] = response.content.strip()
    
    cache[state['thread_id']] = state['summary']
    save_cache(cache)
    return state

# Classify node: Rules + LLM for label/confidence
# Classify node: Rules + LLM for label/confidence (improved prompt for better parsing/confidence)
async def classify_intent(state: dict) -> dict:
    # Simple rules first (e.g., keyword-based) - expanded for your emails
    subject_lower = state['subject'].lower()
    body_lower = state['body'].lower()
    if any(word in subject_lower or word in body_lower for word in ["urgent", "emergency", "immediate"]):
        state['label'] = "urgent"
        state['confidence'] = 0.95
        return state
    if any(word in subject_lower or word in body_lower for word in ["billing", "payment", "invoice", "transaction"]):
        state['label'] = "billing"
        state['confidence'] = 0.90
        return state
    if any(word in subject_lower or word in body_lower for word in ["support", "help", "issue", "problem"]):
        state['label'] = "support"
        state['confidence'] = 0.85
        return state
    if any(word in subject_lower or word in body_lower for word in ["sales", "offer", "promotion", "course"]):
        state['label'] = "sales"
        state['confidence'] = 0.80
        return state
    
    # LLM fallback with tuned prompt (examples for better calibration)
    prompt = PromptTemplate.from_template(
        """Classify this email into EXACTLY ONE category from: {categories}.
        Be confident if it matches well (0.8-1.0), medium if unclear (0.5-0.8), low if poor fit (<0.5).
        Output STRICT JSON: {{"label": "category", "confidence": float (0-1)}}.
        
        Examples:
        - Subject: "Urgent Payment Due"; Summary: "Overdue bill reminder" → {{"label": "billing", "confidence": 0.95}}
        - Subject: "Course Update"; Summary: "New AI class registration" → {{"label": "sales", "confidence": 0.85}}
        - Subject: "General News"; Summary: "No action needed" → {{"label": "FYI", "confidence": 0.6}}
        
        Subject: {subject}
        Summary: {summary}"""
    )
    response = await llm.ainvoke(prompt.invoke({
        "categories": "{urgent, support, billing, sales, FYI}",
        "subject": state['subject'],
        "summary": state['summary'] or state['body'][:200]
    }))
    try:
        result = eval(response.content.strip())  # Parse JSON
        state['label'] = result.get("label", "FYI")
        state['confidence'] = result.get("confidence", 0.5)
    except Exception as e:
        print(f"Classification parse error: {e} - Defaulting")
        state['label'] = "FYI"
        state['confidence'] = 0.5
    return state


# Conditional router: If confidence < 0.85, route to Needs-review (re-summarize or manual)
def classify_condition(state: dict) -> Literal["needs_review", "end"]:
    return "needs_review" if state['confidence'] < 0.85 else "end"

# Needs-review node (placeholder: could log or notify; for now, just re-summarize and end)
async def needs_review(state: dict) -> dict:
    print(f"Low confidence ({state['confidence']}) for {state['label']} - Needs review")
    # For demo, re-run summarize as a placeholder action (but now ends without looping)
    return await summarize_email(state)

# Build graph
graph = StateGraph(EmailState)
graph.add_node("summarize_email", summarize_email)
graph.add_node("classify_intent", classify_intent)
graph.add_node("needs_review", needs_review)

# Edges
graph.add_edge(START, "summarize_email")  # Start with summarize
graph.add_edge("summarize_email", "classify_intent")  # Then classify

# Conditional after classify
graph.add_conditional_edges(
    "classify_intent",
    classify_condition,
    {
        "needs_review": "needs_review",
        "end": END
    }
)

# End after needs_review (no loop to prevent infinite runs)
graph.add_edge("needs_review", END)

# Compile
app = graph.compile()

# Test runner: Fetch emails and run graph for each
async def test_graph():
    emails = fetch_unread_emails()  # From oauth_fetch.py
    for email in emails:
        initial_state = {
            "thread_id": email["thread_id"],
            "subject": email["subject"],
            "body": email["body"],
            "summary": None,
            "label": None,
            "confidence": None
        }
        result = await app.ainvoke(initial_state)
        print(f"Thread {result['thread_id']}: Label={result['label']} (Conf={result['confidence']}), Summary={result['summary']}")

if __name__ == "__main__":
    asyncio.run(test_graph())
