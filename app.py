import streamlit as st
import os
import tempfile
import time

# --- AI PROVIDERS ---
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import ChatCohere

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

st.title("Professor AI: Ultimate Edition")
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
        "Groq": "GROQ_API_KEY",
        "Google": "GOOGLE_API_KEY",
        "Cohere": "COHERE_API_KEY"
    }
    try:
        return st.secrets[key_map[provider]]
    except:
        return None


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_llm(provider, api_key, model_name):
    """
    Returns the correct AI model based on user selection.
    """
    if provider == "Groq":
        return ChatGroq(
            groq_api_key=api_key,
            model_name=model_name,
            max_tokens=None, # Allow long answers
            temperature=0.5
        )
    
    elif provider == "Google":
        return ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model_name,
            temperature=0.5,
            convert_system_message_to_human=True
        )
    
    elif provider == "Cohere":
        return ChatCohere(
            cohere_api_key=api_key,
            model=model_name,
            temperature=0.5
        )

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


with st.sidebar:
    st.header("⚙️ Brain Settings")
    
    # A. Provider Selection
    provider = st.radio("Select AI Provider:", ["Groq", "Google", "Cohere"], index=1)
    
    # B. Model Selection (Dynamic)
    if provider == "Groq":
        model_name = st.selectbox("Model:", [
            "llama-3.3-70b-versatile", 
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768"
        ])
    elif provider == "Google":
        model_name = st.selectbox("Model:", [
            "gemini-1.5-flash", # Try this first (Fast & Free)
            "gemini-1.5-pro",   # Smarter but slower
            "gemini-pro" 
        ])
    elif provider == "Cohere":
        model_name = st.selectbox("Model:", [
            "command-r-plus-08-2024", # The new Smartest
            "command-r-08-2024",
        ])

    # C. API Key Logic
    secret_key = get_api_key(provider)
    
    if secret_key:
        st.success(f"✅ {provider} Key loaded from system!")
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
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()


if not api_key:
    st.stop()

if uploaded_file and st.session_state.vectorstore is None:
    with st.spinner("🧠 Analyzing PDF (Deep Chunking)..."):
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
            st.session_state.vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            
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
                llm = get_llm(provider, api_key, model_name)
                
                # Fetch 6 pages of context
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 6})
                
                template = """
                You are an experienced university-level educator whose primary goal is to help a student deeply understand their coursework.

                Your role is adaptive:
                - If the subject is technical, explain it with precision and logic.
                - If the subject is theoretical, explain it with intuition and clarity.
                - If the subject is mathematical or scientific, reason step by step.
                - If the subject is descriptive or conceptual, explain ideas, relationships, and implications clearly.

                GENERAL TEACHING GUIDELINES:

                1. Intent Awareness & Clarification
                - If the student’s question contains spelling or grammar mistakes, infer the intended meaning and answer the correct question.
                - If the question is ambiguous or incomplete, first ask a brief clarifying question **only if it is necessary for correctness**.
                - If clarification is not strictly required, proceed with the most reasonable interpretation and explain your assumptions.

                2. Teaching-Oriented Explanation
                - Assume the student is learning this topic for the first time.
                - Teach as a professor would during a lecture or consultation.
                - Prioritize understanding over short or exam-style answers.
                - Always aim to clarify the student’s doubts, either through detailed explanation or by connecting concepts logically.

                3. Adaptive Structure
                - Do NOT follow a fixed or mandatory format.
                - Organize explanations naturally based on the topic and question.
                - Use headings, subheadings, bullet points, step-by-step reasoning, comparisons, or worked examples **only when they improve comprehension**.

                4. Depth, Enrichment & Beyond the Question
                - Fully answer the student’s question first.
                - Then, where appropriate:
                    - introduce prerequisite or background concepts,
                    - explain *why* the topic matters,
                    - connect it to other related areas in the coursework,
                    - highlight common student misconceptions or mistakes.

                5. Real-World Examples & Intuition
                - Include real-world examples, analogies, or practical scenarios wherever possible to make abstract ideas easier to understand.
                - Examples should relate to university-level contexts and daily life to build intuition.

                6. Difficulty Adaptation
                - Adjust the depth, pace, and complexity of the explanation based on the apparent difficulty of the question.
                - For simple questions, explain clearly but concisely.
                - For complex questions, explain progressively, as a professor would on a whiteboard.

                7. Q&A Mode
                - If the student explicitly requests questions and answers, generate them in a **question-paper style** based on the PDF content.
                - Provide correct, clear answers immediately after each question.
                - Ensure questions cover key concepts, examples, and practical applications found in the PDF.

                8. Use of Provided Material
                - Base explanations primarily on the provided PDF context.
                - Do not fabricate facts outside the coursework.
                - If something is implied rather than explicitly stated, explain it carefully as a logical or contextual inference.

                9. Tone
                - Calm, patient, and instructional.
                - Encourage understanding, not memorization.
                - Sound like a professor guiding a student to *learn*, not just to get an answer.

                COURSE MATERIAL CONTEXT:
                {context}

                RECENT CHAT HISTORY:
                {chat_history}

                STUDENT QUESTION:
                {question}

                DETAILED TEACHING RESPONSE:
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

            except Exception as e:
                st.error(f"Error: {e}")
                if "429" in str(e):
                    st.warning("Rate Limit Hit! Try switching the Provider in the sidebar.")
                if "404" in str(e):
                    st.warning("Model not found. Try switching to a different model in the dropdown.")



