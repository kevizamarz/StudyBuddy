import streamlit as st
import os
import tempfile
import time

# NEW LIBRARY FOR GOOGLE
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ----------------------------------------------------------------------
# 1. APP CONFIGURATION
# ----------------------------------------------------------------------
st.set_page_config(page_title="Study Buddy (Gemini)", page_icon="♊", layout="wide")

st.markdown("""
<style>
    .stChatMessage {font-size: 1.05rem;}
</style>
""", unsafe_allow_html=True)

st.title("Professor AI: Study Buddy")
st.caption("High limits, huge context window, and deep answers.")

# ----------------------------------------------------------------------
# 2. SESSION STATE
# ----------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "current_file" not in st.session_state:
    st.session_state.current_file = ""

# ----------------------------------------------------------------------
# 3. AI FUNCTIONS
# ----------------------------------------------------------------------

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_llm(api_key):
    return ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model="gemini-pro", # The stable, standard version
        temperature=0.6,
        convert_system_message_to_human=True 
    )
    
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ----------------------------------------------------------------------
# 4. SIDEBAR SETTINGS
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    
    # 1. Google API Key
    input_key = st.text_input("Google API Key:", type="password", value=st.session_state.api_key)
    if input_key:
        st.session_state.api_key = input_key.strip()
    
    
    st.divider()
    
    # 2. File Uploader
    uploaded_file = st.file_uploader("📂 Upload PDF", type="pdf")

    if uploaded_file:
        if uploaded_file.name != st.session_state.current_file:
            st.toast("New file detected. Resetting memory...", icon="🧠")
            st.session_state.vectorstore = None
            st.session_state.messages = []
            st.session_state.current_file = uploaded_file.name
            st.rerun()

    # 3. Download/Clear
    if len(st.session_state.messages) > 0:
        st.divider()
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

# ----------------------------------------------------------------------
# 5. PDF PROCESSING
# ----------------------------------------------------------------------
if not st.session_state.api_key:
    st.warning("👈 Enter Google API Key to start.")
    st.stop()

if uploaded_file and st.session_state.vectorstore is None:
    with st.spinner("🧠 Analyzing PDF..."):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            # We can use slightly larger chunks with Gemini because it has a huge window
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000, 
                chunk_overlap=300,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            splits = text_splitter.split_documents(docs)

            embeddings = get_embeddings()
            st.session_state.vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            
            os.remove(tmp_path)
            st.success(f"Ready! Processed {len(splits)} chunks.")
            time.sleep(1)
            st.rerun()
            
        except Exception as e:
            st.error(f"Error processing PDF: {e}")

# ----------------------------------------------------------------------
# 6. CHAT LOGIC
# ----------------------------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    
    if not st.session_state.vectorstore:
        st.error("Upload a PDF first.")
        st.stop()

    # User Msg
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI Msg
    with st.chat_message("assistant"):
        with st.spinner("Gemini is thinking..."):
            try:
                llm = get_llm(st.session_state.api_key)
                
                # k=7 (Giving Gemini lots of context because it can handle it)
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 7})
                
                template = """
                You are an expert University Professor.
                Answer strictly based on the context provided.
                
                INSTRUCTIONS:
                1. If grammar is wrong, fix it mentally and answer the intended question.
                2. Be detailed and academic. Do not be brief.
                3. Use this structure:
                   - **Concept Definition**
                   - **Detailed Explanation**
                   - **Examples/Code**
                   - **Summary**

                CONTEXT:
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

                response = rag_chain.invoke(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

                # Sources
                related_docs = retriever.invoke(prompt)
                with st.expander("📚 Sources"):
                    for i, doc in enumerate(related_docs):
                        st.markdown(f"**Page {doc.metadata.get('page', '?')}**")
                        st.caption(doc.page_content[:200] + "...")

            except Exception as e:
                st.error(f"Error: {e}")

