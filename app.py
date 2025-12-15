import streamlit as st
import os
import tempfile
import time

# Libraries for RAG
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


st.set_page_config(
    page_title="Study Buddy Pro", 
    page_icon="🎓", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom CSS: Makes the text area bigger and cleaner
st.markdown("""
<style>
    .stChatMessage {font-size: 1.05rem;}
    div.stButton > button {width: 100%;}
</style>
""", unsafe_allow_html=True)

st.title("Study Buddy")
st.caption("I provide exhaustive, lecture-quality answers from your coursework.")


if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "api_key" not in st.session_state:
    st.session_state.api_key = ""

if "current_file" not in st.session_state:
    st.session_state.current_file = ""



@st.cache_resource
def get_embeddings():
    # Downloads the embedding model locally (Runs on CPU)
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_llm(api_key):
    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        
        max_tokens=None, 
        temperature=0.6 
    )

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


with st.sidebar:
    st.header("⚙️ Configuration")
    
    # 1. API Key
    input_key = st.text_input("Groq API Key:", type="password", value=st.session_state.api_key)
    if input_key:
        st.session_state.api_key = input_key.strip()
    
    st.divider()
    
    # 2. File Uploader
    uploaded_file = st.file_uploader("📂 Upload PDF (Coursework)", type="pdf")

    # 3. Intelligent Reset Logic
    if uploaded_file:
        if uploaded_file.name != st.session_state.current_file:
            st.toast("New file detected. Processing...", icon="🧠")
            st.session_state.vectorstore = None
            st.session_state.messages = []
            st.session_state.current_file = uploaded_file.name
            st.rerun()

    # 4. Download Chat History
    if len(st.session_state.messages) > 0:
        st.divider()
        chat_text = ""
        for msg in st.session_state.messages:
            role = "STUDENT" if msg["role"] == "user" else "PROFESSOR"
            chat_text += f"{role}: {msg['content']}\n\n"
            
        st.download_button(
            label="💾 Download Conversation",
            data=chat_text,
            file_name="study_session.txt",
            mime="text/plain"
        )
        
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()


# Check API Key first
if not st.session_state.api_key:
    st.warning("👈 Please enter your Groq API Key in the sidebar to start.")
    st.stop()

if uploaded_file and st.session_state.vectorstore is None:
    with st.spinner("🧠 Reading PDF... Organizing data for deep understanding..."):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            # MEGA CHUNKS: 2500 characters allows for very long context retention
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2500, 
                chunk_overlap=400,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            splits = text_splitter.split_documents(docs)

            embeddings = get_embeddings()
            st.session_state.vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            
            os.remove(tmp_path)
            st.success(f"Processed {len(splits)} knowledge blocks from the PDF.")
            time.sleep(1)
            st.rerun()
            
        except Exception as e:
            st.error(f"Error processing PDF: {e}")



for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a detailed question..."):
    
    if not st.session_state.vectorstore:
        st.error("Please upload a PDF first.")
        st.stop()

    # User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Consulting the lecture notes..."):
            try:
                llm = get_llm(st.session_state.api_key)
                
                # Fetch 6 very large chunks (This is about 15,000 characters of context)
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 6})
                
                # --- THE "LONG FORM" PROMPT ---
                template = """
                You are a distinguished University Professor in IT. 
                Your goal is to provide EXHAUSTIVE, DETAILED explanations.
                
                INSTRUCTIONS FOR THE PROFESSOR:
                1. **Grammar Check**: If the student makes spelling errors, interpret their intent and answer the correct question.
                2. **Length**: Do NOT summarize. Write as much as necessary to fully explain the topic.
                3. **Structure**: You MUST use the following format exactly:

                ### 1. Introduction
                - Define the concept clearly.
                
                ### 2. Deep Dive Analysis
                - Explain the underlying mechanics, logic, or theory.
                - Connect this concept to other parts of the coursework if possible.
                
                ### 3. Practical Examples
                - Provide code snippets, real-world analogies, or diagrams (using text) to illustrate.
                
                ### 4. Summary & Key Takeaways
                - Bullet points of the most important facts.

                CONTEXT FROM PDF:
                {context}

                CHAT HISTORY:
                {chat_history}

                STUDENT QUESTION:
                {question}

                PROFESSOR'S DETAILED ANSWER:
                """

                custom_prompt = PromptTemplate.from_template(template)

                history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-5:]])

                # The Chain
                rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough(), "chat_history": lambda x: history_str}
                    | custom_prompt
                    | llm
                    | StrOutputParser()
                )

                response = rag_chain.invoke(prompt)
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

                # Source Accordion
                related_docs = retriever.invoke(prompt)
                with st.expander("📚 View Source Snippets (Evidence)"):
                    for i, doc in enumerate(related_docs):
                        st.markdown(f"**Page {doc.metadata.get('page', '?')}**")
                        st.caption(doc.page_content[:300] + "...")

            except Exception as e:
                st.error(f"Error generation: {e}")
