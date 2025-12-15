import streamlit as st
import os
import tempfile

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


st.set_page_config(page_title="Study Buddy Pro", page_icon="🎓", layout="wide")

st.title("Study Buddy")


# Store Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Store the Vector Database
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Store the Current File Name (To detect if you changed files)
if "current_file" not in st.session_state:
    st.session_state.current_file = ""

# Store API Key so it doesn't vanish on interactions
if "api_key" not in st.session_state:
    st.session_state.api_key = ""


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_llm(api_key):
    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )


with st.sidebar:
    st.header("⚙️ Settings")
    
    # A. API Key Input (With Session State Persistence)
    # We use a placeholder to keep the key if it exists
    input_key = st.text_input("Groq API Key:", type="password", value=st.session_state.api_key)
    if input_key:
        st.session_state.api_key = input_key
    
    st.markdown("[Get Free Key](https://console.groq.com/keys)")
    st.divider()
    
    # B. File Uploader
    uploaded_file = st.file_uploader("📂 Upload Coursework (PDF)", type="pdf")

    # C. RESET LOGIC (Fixing the PDF Switching Issue)
    if uploaded_file:
        # Check if the uploaded file is DIFFERENT from the one in memory
        if uploaded_file.name != st.session_state.current_file:
            st.toast(f"New file detected: {uploaded_file.name}")
            
            # CLEAR EVERYTHING
            st.session_state.vectorstore = None
            st.session_state.messages = []
            st.session_state.current_file = uploaded_file.name # Update current file name
            
            # Force a rerun to refresh the UI immediately
            st.rerun()

    # D. Session History (Visual list of questions asked)
    if len(st.session_state.messages) > 0:
        st.divider()
        st.subheader("🕑 Session History")
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                # Truncate long questions
                display_text = (msg["content"][:40] + '..') if len(msg["content"]) > 40 else msg["content"]
                st.caption(f"🔹 {display_text}")
        
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()


# Ensure API Key is present
if not st.session_state.api_key:
    st.warning("👈 Please enter your Groq API Key in the sidebar to start.")
    st.stop()

# Ensure File is processed
if uploaded_file and st.session_state.vectorstore is None:
    with st.spinner("🧠 Analyzing new PDF..."):
        try:
            embeddings = get_embeddings()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_file_path = tmp_file.name

            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)

            st.session_state.vectorstore = Chroma.from_documents(
                documents=chunks, 
                embedding=embeddings
            )
            
            os.remove(tmp_file_path)
            st.success("Analysis Complete! Ask me anything.")
        except Exception as e:
            st.error(f"Error: {e}")



# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle Input
if prompt := st.chat_input("Ask a question based on the PDF..."):
    
    if st.session_state.vectorstore is None:
        st.error("Please upload a PDF first!")
        st.stop()

    # User Message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # AI Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                llm = get_llm(st.session_state.api_key)
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})

                # Prepare context
                chat_history_str = "\n".join(
                    [f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-5:]]
                )

                template = """
                You are an expert university tutor.
                
                Instructions:
                1. Answer based ONLY on the Context below.
                2. If the answer isn't there, say you don't know.
                3. Be detailed and helpful.
                
                Chat History:
                {chat_history}

                Context:
                {context}

                Question:
                {question}

                Answer:
                """
                
                custom_prompt = PromptTemplate.from_template(template)
                
                chain = (
                    {
                        "context": retriever, 
                        "question": lambda x: x,
                        "chat_history": lambda x: chat_history_str
                    }
                    | custom_prompt
                    | llm
                    | StrOutputParser()
                )

                response = chain.invoke(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Rerun to update sidebar history immediately
                st.rerun()

            except Exception as e:
                st.error(f"Error: {e}")