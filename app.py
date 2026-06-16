import streamlit as st
import os
import tempfile
import time

# --- AI PROVIDERS ---
from langchain_groq import ChatGroq

# --- RAG TOOLS ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


st.set_page_config(page_title="Study Buddy Pro", page_icon="🎓", layout="wide")

# Custom CSS for a better look
st.markdown("""
<style>
    .stChatMessage {font-size: 1.05rem;}
</style>
""", unsafe_allow_html=True)

st.title("Study Buddy: Ultimate Edition")
st.caption("Upload your coursework and ask questions!")


if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "current_file" not in st.session_state:
    st.session_state.current_file = ""


def get_api_key(provider):
    """
    Checks Streamlit Secrets first. If found, returns it.
    This allows you to share the app without friends typing keys.
    """
    key_map = {
        "Groq": "GROQ_API_KEY"
    }
    try:
        return st.secrets[key_map[provider]]
    except:
        return None


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

def get_llm(api_key, model_name):
    """
    Returns the correct AI model based on user selection.
    """
    return ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        max_tokens=None, # Allow long answers
        temperature=0.5
    )

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


with st.sidebar:
    st.header("Brain Settings")
    
    # A. Provider Selection
    provider = "Groq"
    st.success("Using Groq")
    
    # B. Model Selection (Dynamic)

    model_name = st.selectbox(
        "Model",
        [
            "llama-3.3-70b-versatile",
            "deepseek-r1-distill-llama-70b",
            "qwen/qwen3-32b"
        ]
    )

    # C. API Key Logic
    secret_key = get_api_key(provider)
    
    if secret_key:
        st.success(f"{provider} Key loaded from system!")
        api_key = secret_key
    else:
        api_key = st.text_input(f"Enter {provider} API Key:", type="password")
        if not api_key:
             st.warning(f"Please enter a key.")

    st.divider()
    
    # D. File Uploader
    uploaded_file = st.file_uploader("📂 Upload PDF", type="pdf")

    if uploaded_file:
        if uploaded_file.name != st.session_state.current_file:
            st.toast("New file detected. Resetting memory...", icon="🧠")
            st.session_state.vectorstore = None
            st.session_state.messages = []
            st.session_state.current_file = uploaded_file.name
            st.rerun()

    # E. Tools
    if len(st.session_state.messages) > 0:
        st.divider()
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()


if not api_key:
    st.stop()

if uploaded_file and st.session_state.vectorstore is None:
    with st.spinner("Analyzing PDF (Deep Chunking)..."):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            # Large chunks for deep context
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000, chunk_overlap=300
            )
            splits = text_splitter.split_documents(docs)

            embeddings = get_embeddings()
            persist_dir = f"chroma_db/{uploaded_file.name}"

            st.session_state.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=persist_dir
            )
            
            os.remove(tmp_path)
            st.success(f"Processed {len(splits)} knowledge blocks.")
            time.sleep(1)
            st.rerun()
            
        except Exception as e:
            st.error(f"Error processing PDF: {e}")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a detailed question..."):
    
    if not st.session_state.vectorstore:
        st.error("Upload a PDF first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"Consulting {model_name}..."):
            try:
                llm = get_llm(api_key, model_name)
                
                # Fetch pages of context
                retriever = st.session_state.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 8, "fetch_k": 30}
                )
                
                template = """
                You are an expert university professor helping a student learn.

                Use the provided course material as your primary source.

                Guidelines:
                - Explain concepts clearly and accurately.
                - Adapt depth to the student's question.
                - Use examples and analogies when helpful.
                - Show step-by-step reasoning for technical topics.
                - If the user asks for quizzes, generate questions and answers from the material.
                - If information is missing from the material, say so instead of inventing facts.

                COURSE MATERIAL:
                {context}

                CHAT HISTORY:
                {chat_history}

                QUESTION:
                {question}

                ANSWER:
                """

                custom_prompt = PromptTemplate.from_template(template)
                history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-5:]])

                rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough(), "chat_history": lambda x: history_str}
                    | custom_prompt
                    | llm
                    | StrOutputParser()
                )
                docs = retriever.invoke(prompt)

                response = rag_chain.invoke(prompt)

                st.markdown(response)

                with st.expander("Sources Used"):
                    for i, doc in enumerate(docs, 1):
                        page = doc.metadata.get("page", "Unknown")

                        st.markdown(f"### Source {i} (Page {page + 1})")

                        st.write(
                            doc.page_content[:500] +
                            ("..." if len(doc.page_content) > 500 else "")
                        )

                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )


            except Exception as e:
                st.error(f"Error: {e}")
                if "429" in str(e):
                    st.warning("Rate Limit Hit! Try switching the Provider in the sidebar.")
                if "404" in str(e):
                    st.warning("Model not found. Try switching to a different model in the dropdown.")


