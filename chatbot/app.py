import streamlit as st
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from ddgs import DDGS

load_dotenv()

st.set_page_config(page_title="Web Search Chatbot", page_icon="🌐")
st.title("🌐 Web Search Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

llm = ChatOllama(model="llama3.2")


def web_search(query, max_results=4):
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    f"{r['title']}: {r['body']}\nSource: {r['href']}"
                )
        return "\n\n".join(results) if results else "No relevant web results found."
    except Exception:
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

    with st.spinner("Thinking..."):
        try:
            web_context = web_search(question)

            history_text = "\n".join(
                f"{m['role']}: {m['content']}"
                for m in st.session_state.messages[-6:-1]
            )

            prompt = f"""
You are a helpful, knowledgeable AI assistant with access to your own broad
knowledge AND some live web search results below.

Use the web search results if they are relevant and helpful. If they are not
relevant, irrelevant, or empty, ignore them and answer using your own
knowledge instead. Never refuse to answer just because the web results don't
cover it — only say you don't know if you genuinely have no idea.

Recent conversation:
{history_text}

Web search results:
{web_context}

Question:
{question}

Answer clearly and directly:
"""

            response = llm.invoke(prompt)
            answer = response.content

            with st.chat_message("assistant"):
                st.markdown(answer)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

            with st.expander(" Web Search Results"):
                st.markdown(web_context)

        except Exception as e:
            st.error(f"Error: {e}")