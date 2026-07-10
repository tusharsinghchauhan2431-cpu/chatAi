import streamlit as st
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from ddgs import DDGS

load_dotenv()

st.set_page_config(page_title="Web Search Chatbot", page_icon="🌐")
st.title("🌐 Web Search Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_resource
def get_llm():
    return ChatOllama(model="llama3.2")


llm = get_llm()


def web_search(query, max_results=4):
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    f"{r['title']}: {r['body']}\nSource: {r['href']}"
                )
        return "\n\n".join(results) if results else "No relevant web results found."
    except Exception as e:
        print(f"Web search error: {e}")
        return "Web search unavailable right now."


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.spinner("Thinking.."):
        web_context = web_search(question)

    history_text = "\n".join(
        f"{m['role']}: {m['content']}"
        for m in st.session_state.messages[-6:-1]
    )

    prompt = f"""
You are a friendly, helpful AI assistant with access to your own broad
knowledge AND some live web search results below.

Guidelines:
- If the message is a greeting, casual chat, or looks like a typo of a
  common word or greeting (e.g. "hlw", "helo", "thx", "sup"), just respond
  naturally as if you understood it. Do NOT ask the user to clarify unless
  the message is genuinely ambiguous or ONLY makes sense with more context.
- Use the web search results if they are relevant and helpful.
- If the web results are irrelevant or empty, ignore them and answer using
  your own knowledge instead.
- Never refuse to answer just because the web results don't cover it — only
  say you don't know if you genuinely have no idea.
- Keep answers clear, direct, and conversational.


Recent conversation:
{history_text}

Web search results:
{web_context}

Question:
{question}

Answer:
"""

    try:
        with st.chat_message("assistant"):
            answer = st.write_stream(
                chunk.content for chunk in llm.stream(prompt)
            )

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        with st.expander("🔍 Web Search Results"):
            st.markdown(web_context)

    except Exception as e:
        st.error(f"Error: {e}")
