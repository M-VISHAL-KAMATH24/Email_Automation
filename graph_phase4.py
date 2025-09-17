import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Optional, Literal, List
from langchain.prompts import PromptTemplate
import pickle  # For simple caching
import asyncio

# For Gmail API
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import base64
from email.mime.text import MIMEText

# For RAG
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# For checkpoints (human-in-the-loop)
from langgraph.checkpoint.memory import MemorySaver

# Import fetch function (assume oauth_fetch.py is in the same directory)
from oauth_fetch import fetch_unread_emails

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0)

# Updated scopes for Phase 4
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.send"
]

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
    snippets: Optional[List[dict]]
    retrieval_weak: Optional[bool]
    draft: Optional[str]
    draft_id: Optional[str]
    approved: Optional[bool]
    sender_email: Optional[str]  # Add this line for the original sender

# Load and split FAQs (ensure docs/faqs.txt exists)
loader = TextLoader("docs/faqs.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Embeddings model (local, fast)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Build/load vector store
vector_store = FAISS.from_documents(docs, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 chunks

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
async def classify_intent(state: dict) -> dict:
    # Simple rules first (e.g., keyword-based) - expanded for your emails
    sender_email = state.get("sender_email", "").lower()
    personal_contacts = ["kamathvishal26@gmail.com", "vishalkamath69@gmail.com"]
    if any(contact in sender_email for contact in personal_contacts):
        state['label'] = "personal"
        state['confidence'] = 1.0  # Be 100% confident it's personal
        return state
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

# Conditional router: If confidence < 0.85, route to Needs-review
def classify_condition(state: dict) -> Literal["needs_review", "end"]:
    return "needs_review" if state['confidence'] < 0.85 else "end"

# Needs-review node (placeholder: could log or notify; for now, just re-summarize and end)
async def needs_review(state: dict) -> dict:
    print(f"Low confidence ({state['confidence']}) for {state['label']} - Needs review")
    # For demo, re-run summarize as a placeholder action (but now ends without looping)
    return await summarize_email(state)

# Retrieve snippets node
async def retrieve_snippets(state: dict) -> dict:
    query = f"{state['subject']} {state['summary'] or state['body'][:200]}"
    retrieved_docs = retriever.invoke(query)
    state['snippets'] = [
        {"content": doc.page_content, "source": doc.metadata.get("source", "Unknown")}
        for doc in retrieved_docs
    ]
    # Check retrieval quality (simple: if low similarity, set flag for no-answer)
    if retrieved_docs and len(retrieved_docs) < 3:  # Adjust threshold
        state['retrieval_weak'] = True
    else:
        state['retrieval_weak'] = False
    return state

# Draft response node with grounded prompt
async def draft_response(state: dict) -> dict:
    if state['retrieval_weak']:
        state['draft'] = "I'm sorry, I couldn't find reliable information to answer this. Escalating to human support."
        return state
    
    snippets_text = "\n".join([f"Snippet {i+1} (Source: {s['source']}): {s['content']}" for i, s in enumerate(state['snippets'])])
    
    prompt = PromptTemplate.from_template(
        """Based ONLY on these retrieved snippets, draft a polite email response. Cite sources inline [Source: X]. If snippets don't fully answer, say "I need more info—escalating" and DO NOT hallucinate.
        Categories: For support/billing, provide helpful cited info; for others, keep brief.
        
        Subject: {subject}
        Summary: {summary}
        Snippets: {snippets_text}
        
        Draft:"""
    )
    response = await llm.ainvoke(prompt.invoke({
        "subject": state['subject'],
        "summary": state['summary'],
        "snippets_text": snippets_text
    }))
    state['draft'] = response.content.strip()
    return state

# Phase 4: Draft reply node
async def draft_reply(state: dict) -> dict:
    if state['retrieval_weak']:
        state['draft'] = "Escalating to human due to insufficient info."
        return state
    
    snippets_text = "\n".join([f"Snippet {i+1} [Source: {s['source']}]: {s['content']}" for i, s in enumerate(state['snippets'])])
    
    prompt = PromptTemplate.from_template(
        """Draft a polite, professional reply to this email. Personalize with sender's name if available. Use friendly tone. Cite sources inline [Source: X]. Guardrails: Mask any PII (e.g., replace emails with [REDACTED]). No external links unless from FAQs. If unsure, suggest escalation.
        
        Original Subject: {subject}
        Summary: {summary}
        Snippets: {snippets_text}
        
        Reply Draft (include subject and body):"""
    )
    response = await llm.ainvoke(prompt.invoke({
        "subject": state['subject'],
        "summary": state['summary'],
        "snippets_text": snippets_text
    }))
    state['draft'] = response.content.strip()  # e.g., "Subject: Re: Your Query\nBody: Dear [Name], ..."
    return state

# Helper to build Gmail service
def build_gmail_service(creds):
    return build("gmail", "v1", credentials=creds)

# Create Gmail draft node
async def create_gmail_draft(state: dict) -> dict:
    """
    Create a Gmail draft that replies to the original sender.
    """
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    service = build_gmail_service(creds)
    
    draft_content = state.get("draft", "")
    subject_line = f"Re: {state['subject']}"
    body_text = draft_content

    # Handle cases where the AI provides a full "Subject/Body" structure
    if "\nBody: " in draft_content:
        parts = draft_content.split("\nBody: ", 1)
        subject_line = parts[0].replace("Subject: ", "")
        body_text = parts[1] if len(parts) > 1 else ""
    
    # Build the MIME message
    message = MIMEText(body_text)
    message["to"] = state.get("sender_email", "default@sender.com")
    message["subject"] = subject_line
    
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    
    try:
        draft_obj = (
            service.users()
            .drafts()
            .create(userId="me", body={"message": {"raw": raw}})
            .execute()
        )
        state["draft_id"] = draft_obj["id"]
        print(f"Draft created with content: ID {state['draft_id']}")
    except HttpError as e:
        state["draft_id"] = None
        print(f"Error creating draft: {e}")
        
    return state

# Approval gate node (human-in-the-loop)
async def approval_gate(state: dict) -> dict:
    print(f"\nDraft for approval: {state['draft']}\nApprove? (y/n): ")
    approval = input().lower()
    state['approved'] = approval == 'y'
    if not state['approved']:
        print("Draft rejected—edit manually in Gmail.")
    return state

# Send email node
async def send_email(state: dict) -> dict:
    if not state.get('draft_id'):
        return state
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    service = build_gmail_service(creds)
    try:
        sent = service.users().drafts().send(userId="me", body={'id': state['draft_id']}).execute()
        print(f"Email sent: ID {sent['id']}")
    except HttpError as e:
        print(f"Error sending: {e}")
    return state

# Checkpoint for persistence
checkpointer = MemorySaver()

# Build graph
graph = StateGraph(EmailState)
graph.add_node("summarize_email", summarize_email)
graph.add_node("classify_intent", classify_intent)
graph.add_node("needs_review", needs_review)
graph.add_node("retrieve_snippets", retrieve_snippets)
graph.add_node("draft_response", draft_response)
graph.add_node("draft_reply", draft_reply)
graph.add_node("create_gmail_draft", create_gmail_draft)
graph.add_node("approval_gate", approval_gate)
graph.add_node("send_email", send_email)

# Edges
graph.add_edge(START, "summarize_email")
graph.add_edge("summarize_email", "classify_intent")

# Conditional after classify
graph.add_conditional_edges(
    "classify_intent",
    classify_condition,
    {
        "needs_review": "needs_review",
        "end": END
    }
)
graph.add_edge("needs_review", END)

# Phase 3/4 conditional for retrieve/draft
# In the post_classify_condition function

def post_classify_condition(state: dict) -> Literal["retrieve_snippets", "end"]:
    # Add "personal" to this list
    if state['label'] in ["support", "billing", "personal"] and state['confidence'] >= 0.85:
        return "retrieve_snippets"
    return "end"


graph.add_conditional_edges(
    "classify_intent",
    post_classify_condition,
    {"retrieve_snippets": "retrieve_snippets", "end": END}
)
graph.add_edge("retrieve_snippets", "draft_response")
graph.add_edge("draft_response", "draft_reply")
graph.add_edge("draft_reply", "create_gmail_draft")
graph.add_edge("create_gmail_draft", "approval_gate")

# Conditional after approval
def approval_condition(state: dict) -> Literal["send_email", "end"]:
    return "send_email" if state.get('approved') else "end"

graph.add_conditional_edges(
    "approval_gate",
    approval_condition,
    {"send_email": "send_email", "end": END}
)
graph.add_edge("send_email", END)

# Compile with checkpointer
app = graph.compile(checkpointer=checkpointer)

# Test runner: Fetch emails and run graph for each
async def test_graph():
    emails = fetch_unread_emails()  # From oauth_fetch.py
    for email in emails:
        # Use get with fallback to prevent KeyError if 'from' is missing
        sender = email.get("from", "default@sender.com")

        initial_state = {
            "thread_id": email["thread_id"],
            "subject": email["subject"],
            "body": email["body"],
            "sender_email": sender,
            "summary": None,
            "label": None,
            "confidence": None,
            "snippets": None,
            "retrieval_weak": None,
            "draft": None,
            "draft_id": None,
            "approved": None
        }
        config = {"configurable": {"thread_id": email["thread_id"]}}  # For checkpoints
        result = await app.ainvoke(initial_state, config)
        print(f"Thread {result['thread_id']}: Label={result['label']} (Conf={result['confidence']}), Summary={result['summary']}, Draft={result.get('draft', 'N/A')}, Sent={'Yes' if result.get('approved') else 'No'}")

if __name__ == "__main__":
    asyncio.run(test_graph())
