import os
import asyncio
import pickle
import base64
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from typing import TypedDict, Optional, List, Literal
from dotenv import load_dotenv
from colorama import init, Fore, Style

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

from oauth_fetch import fetch_unread_emails

init(autoreset=True)  # Initialize colorama for colors

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0)

FOLLOW_UP_DAYS = 5

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.send"
]

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
    sender_email: Optional[str]
    sent_time: Optional[str]
    followed_up: Optional[bool]


loader = TextLoader("docs/faqs.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(docs, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})


def build_gmail_service(creds):
    return build("gmail", "v1", credentials=creds)


async def summarize_email(state: dict) -> dict:
    cache = load_cache()
    if state["thread_id"] in cache:
        state["summary"] = cache[state["thread_id"]]
        print(f"{Fore.YELLOW}Cache hit for thread {state['thread_id']}")
        return state

    prompt = PromptTemplate.from_template(
        "Summarize this email in 1-2 sentences, extracting key entities (names, dates, topics): {body}"
    )
    response = await llm.ainvoke(prompt.invoke({"body": state["body"]}))
    state["summary"] = response.content.strip()

    cache[state["thread_id"]] = state["summary"]
    save_cache(cache)
    return state


async def classify_intent(state: dict) -> dict:
    sender_email = state.get("sender_email", "").lower()
    personal_contacts = ["kamathvishal26@gmail.com", "vishalkamath69@gmail.com"]
    if any(contact in sender_email for contact in personal_contacts):
        state["label"] = "personal"
        state["confidence"] = 1.0
        return state

    subject_lower = state["subject"].lower()
    body_lower = state["body"].lower()

    keywords = {
        "urgent": ["urgent", "emergency", "immediate"],
        "billing": ["billing", "payment", "invoice", "transaction"],
        "support": ["support", "help", "issue", "problem"],
        "sales": ["sales", "offer", "promotion", "course"],
    }

    for label, words in keywords.items():
        if any(word in subject_lower or word in body_lower for word in words):
            state["label"] = label
            state["confidence"] = 0.9
            return state

    # LLM fallback
    prompt = PromptTemplate.from_template(
        """Classify this email into EXACTLY ONE category from: {categories}.
        Output STRICT JSON: {{"label": "category", "confidence": float (0-1)}}.

        Subject: {subject}
        Summary: {summary}"""
    )
    response = await llm.ainvoke(
        prompt.invoke(
            {
                "categories": "{urgent, support, billing, sales, FYI}",
                "subject": state["subject"],
                "summary": state["summary"] or state["body"][:200],
            }
        )
    )
    try:
        result = eval(response.content.strip())
        state["label"] = result.get("label", "FYI")
        state["confidence"] = result.get("confidence", 0.5)
    except Exception as e:
        print(f"{Fore.RED}Classification parse error: {e} - Defaulting")
        state["label"] = "FYI"
        state["confidence"] = 0.5
    return state


def classify_condition(state: dict) -> Literal["needs_review", "end"]:
    return "needs_review" if state["confidence"] < 0.85 else "end"


async def needs_review(state: dict) -> dict:
    print(f"{Fore.RED}Low confidence ({state['confidence']}) for {state['label']} - Needs review")
    return await summarize_email(state)


async def retrieve_snippets(state: dict) -> dict:
    query = f"{state['subject']} {state['summary'] or state['body'][:200]}"
    retrieved_docs = retriever.invoke(query)
    state["snippets"] = [{"content": doc.page_content, "source": doc.metadata.get("source", "Unknown")} for doc in retrieved_docs]
    state["retrieval_weak"] = len(retrieved_docs) < 3 if retrieved_docs else True
    return state


async def draft_response(state: dict) -> dict:
    if state.get("retrieval_weak"):
        state["draft"] = "I'm sorry, I couldn't find reliable information to answer this. Escalating to human support."
        return state
    snippets_text = "\n".join(
        [f"Snippet {i+1} (Source: {s['source']}): {s['content']}" for i, s in enumerate(state["snippets"])]
    )
    prompt = PromptTemplate.from_template(
        """Based ONLY on these retrieved snippets, draft a polite email response. Cite sources inline [Source: X]. If snippets don't fully answer, say "I need more info—escalating" and DO NOT hallucinate.

        Subject: {subject}
        Summary: {summary}
        Snippets: {snippets_text}

        Draft:"""
    )
    response = await llm.ainvoke(
        prompt.invoke({"subject": state["subject"], "summary": state["summary"], "snippets_text": snippets_text})
    )
    state["draft"] = response.content.strip()
    return state


async def draft_reply(state: dict) -> dict:
    if state.get("retrieval_weak"):
        state["draft"] = "Escalating to human due to insufficient info."
        return state
    snippets_text = "\n".join(
        [f"Snippet {i+1} [Source: {s['source']}]: {s['content']}" for i, s in enumerate(state["snippets"])]
    )
    prompt = PromptTemplate.from_template(
        """Draft a polite, professional reply to this email. Personalize with sender's name if available. Use friendly tone. Cite sources inline [Source: X]. Guardrails: Mask any PII (e.g., replace emails with [REDACTED]). No external links unless from FAQs. If unsure, suggest escalation.
        
        Original Subject: {subject}
        Summary: {summary}
        Snippets: {snippets_text}
        
        Reply Draft (include subject and body):"""
    )
    response = await llm.ainvoke(
        prompt.invoke({"subject": state["subject"], "summary": state["summary"], "snippets_text": snippets_text})
    )
    state["draft"] = response.content.strip()
    return state


async def draft_followup(state: dict) -> dict:
    prompt = PromptTemplate.from_template(
        """You are an AI assistant drafting a polite, friendly follow-up email as a reminder.
        The original message was sent {days} days ago and no reply has been received yet.
        Write a brief email to nudge the recipient for a response.

        Follow-up email draft:"""
    )
    response = await llm.ainvoke(prompt.invoke({"days": FOLLOW_UP_DAYS}))
    state["draft"] = response.content.strip()
    state["is_followup"] = True
    return state


async def create_gmail_draft(state: dict) -> dict:
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    service = build_gmail_service(creds)
    draft_content = state.get("draft", "")
    subject_line = f"Re: {state['subject']}"
    body_text = draft_content

    if "\nBody: " in draft_content:
        parts = draft_content.split("\nBody: ", 1)
        subject_line = parts[0].replace("Subject: ", "")
        body_text = parts[1] if len(parts) > 1 else ""

    message = MIMEText(body_text)
    message["to"] = state.get("sender_email", "default@sender.com")
    message["subject"] = subject_line

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

    try:
        draft_obj = service.users().drafts().create(userId="me", body={"message": {"raw": raw}}).execute()
        state["draft_id"] = draft_obj["id"]
        print(f"{Fore.GREEN}Draft created with ID: {state['draft_id']}")
    except HttpError as e:
        state["draft_id"] = None
        print(f"{Fore.RED}Error creating draft: {e}")

    return state


async def approval_gate(state: dict) -> dict:
    print(f"\n{Fore.RED}EMAIL FROM INBOX:\n{state['body']}\n")
    print(f"{Fore.GREEN}AI GENERATED DRAFT:\n{state.get('draft', 'No draft.')}\n")
    approval = input(Fore.YELLOW + "Approve draft? (y/n): ").strip().lower()
    state["approved"] = approval == "y"
    if not state["approved"]:
        print("Draft rejected - please edit manually if needed.")
    return state


async def send_email(state: dict) -> dict:
    if not state.get("draft_id") or not state.get("approved"):
        print(f"{Fore.YELLOW}No draft found or not approved. Skipping send.")
        return state

    creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    service = build_gmail_service(creds)
    try:
        sent = service.users().drafts().send(userId="me", body={"id": state["draft_id"]}).execute()
        print(f"{Fore.GREEN}Email sent: ID {sent['id']}")
        if state.get("is_followup"):
            state["followed_up"] = True
            print(f"Marked thread {state['thread_id']} as followed up.")
        state["sent_time"] = datetime.now().isoformat()
    except HttpError as e:
        print(f"{Fore.RED}Error sending email: {e}")

    return state


async def check_unreplied_threads(state: dict) -> dict:
    if state.get("followed_up"):
        state["followup_needed"] = False
        return state
    if not state.get("sent_time"):
        state["followup_needed"] = False
        return state

    sent_dt = datetime.fromisoformat(state["sent_time"])
    now = datetime.now()

    if now - sent_dt < timedelta(days=FOLLOW_UP_DAYS):
        state["followup_needed"] = False
        return state

    creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    service = build_gmail_service(creds)

    try:
        thread = service.users().threads().get(userId="me", id=state["thread_id"]).execute()
        messages = thread.get("messages", [])
        state["followup_needed"] = len(messages) <= 1
        print(f"{Fore.CYAN}Checked thread {state['thread_id']} - Follow-up needed: {state['followup_needed']}")
    except Exception as e:
        print(f"{Fore.RED}Failed to check thread replies: {e}")
        state["followup_needed"] = False

    return state


async def label_and_archive(state: dict) -> dict:
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    service = build_gmail_service(creds)
    thread_id = state.get("thread_id")
    if not thread_id:
        print(f"{Fore.YELLOW}No thread_id, skipping labeling and archiving.")
        return state

    # Replace with your Gmail label IDs
    label_handled = "Label_Handled_Id"
    label_waiting = "Label_Waiting_Id"
    label_escalated = "Label_Escalated_Id"

    label_to_add = label_handled
    remove_label = ["INBOX"]  # Archive thread from inbox

    try:
        service.users().threads().modify(
            userId="me",
            id=thread_id,
            body={"addLabelIds": [label_to_add], "removeLabelIds": remove_label}
        ).execute()
        print(f"{Fore.MAGENTA}Thread {thread_id} labeled and archived.")
    except Exception as e:
        print(f"{Fore.RED}Failed to label/archive thread {thread_id}: {e}")

    return state


# Checkpoint setup
checkpointer = MemorySaver()

# Build state graph
graph = StateGraph(EmailState)

graph.add_node("summarize_email", summarize_email)
graph.add_node("classify_intent", classify_intent)
graph.add_node("needs_review", needs_review)
graph.add_node("retrieve_snippets", retrieve_snippets)
graph.add_node("draft_response", draft_response)
graph.add_node("draft_reply", draft_reply)
graph.add_node("draft_followup", draft_followup)
graph.add_node("create_gmail_draft", create_gmail_draft)
graph.add_node("approval_gate", approval_gate)
graph.add_node("send_email", send_email)
graph.add_node("check_unreplied_threads", check_unreplied_threads)
graph.add_node("label_and_archive", label_and_archive)

# Define edges
graph.add_edge(START, "summarize_email")
graph.add_edge("summarize_email", "classify_intent")

graph.add_conditional_edges(
    "classify_intent",
    classify_condition,
    {"needs_review": "needs_review", "end": END},
)
graph.add_edge("needs_review", END)

def post_classify_condition(state: dict) -> Literal["retrieve_snippets", "end"]:
    if state["label"] in ["support", "billing", "personal"] and state["confidence"] >= 0.85:
        return "retrieve_snippets"
    return "end"

graph.add_conditional_edges(
    "classify_intent",
    post_classify_condition,
    {"retrieve_snippets": "retrieve_snippets", "end": END},
)
graph.add_edge("retrieve_snippets", "draft_response")
graph.add_edge("draft_response", "draft_reply")
graph.add_edge("draft_reply", "create_gmail_draft")
graph.add_edge("create_gmail_draft", "approval_gate")

def approval_condition(state: dict) -> Literal["send_email", "end"]:
    return "send_email" if state.get("approved") else "end"

graph.add_conditional_edges(
    "approval_gate", approval_condition, {"send_email": "send_email", "end": END}
)
graph.add_edge("send_email", "check_unreplied_threads")

graph.add_conditional_edges(
    "check_unreplied_threads",
    lambda s: "draft_followup" if s.get("followup_needed") else "label_and_archive",
    {"draft_followup": "draft_followup", "label_and_archive": "label_and_archive"},
)
graph.add_edge("draft_followup", "create_gmail_draft")
graph.add_edge("label_and_archive", END)

app = graph.compile(checkpointer=checkpointer)


# Runner to process unread emails
async def test_graph():
    emails = fetch_unread_emails()
    for email in emails:
        sender = email.get("from", "default@sender.com")
        print(f"\n{Fore.RED}{'='*30}\nReading Email from: {sender}\nSubject: {email['subject']}\n{'='*30}")

        initial_state = {
            "thread_id": email["thread_id"],
            "subject": email["subject"],
            "body": email["body"],
            "sender_email": sender,
        }
        config = {"configurable": {"thread_id": email["thread_id"]}}
        result = await app.ainvoke(initial_state, config)
        print(
            f"Thread {result['thread_id']}: Label={result['label']} (Conf={result['confidence']}), "
            f"Summary={result['summary']}, Draft={result.get('draft', 'N/A')}, Sent={'Yes' if result.get('approved') else 'No'}"
        )


if __name__ == "__main__":
    asyncio.run(test_graph())
