import os
from dotenv import load_dotenv

# 1) Load env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise SystemExit("Set GEMINI_API_KEY in .env (Google AI Studio)")

# 2) LLM: Gemini via LangChain Google integration
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # fast + cheap; swap to gemini-2.5-pro for higher quality
    google_api_key=api_key,
    temperature=0
)

# 3) Minimal agent: LangGraph prebuilt ReAct agent (no tools for now)
from langgraph.prebuilt import create_react_agent
agent = create_react_agent(
    model=llm,
    tools=[],  # add tools later (Gmail, retrieval)
    prompt="You are a concise assistant."
)

# 4) Single invoke to verify end-to-end loop
res = agent.invoke({
    "messages": [{"role": "user", "content": "Hello! Reply in one short friendly line."}]
})
print(res)
