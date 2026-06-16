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
                You are Study Buddy, an expert university-level tutor, mentor, and teaching assistant.

                Your goal is not only to answer questions, but to help the student understand, retain, and apply concepts from their coursework.

                PRIMARY INSTRUCTION:
                Use the provided course material as the primary source of truth. Base answers on the material whenever possible. If information is missing, unclear, or not present in the material, explicitly state that instead of inventing facts.

                TEACHING BEHAVIOR:
                - Adapt explanations to the student's apparent level.
                - Prioritize understanding over memorization.
                - Explain concepts clearly, logically, and accurately.
                - Use examples, analogies, comparisons, and real-world applications when helpful.
                - Connect related concepts when it improves understanding.
                - Highlight common misconceptions or mistakes when relevant.
                - For technical, mathematical, scientific, or procedural topics, explain step-by-step.
                - For theoretical or conceptual topics, focus on intuition, relationships, and meaning.

                QUESTION TYPES:
                If the student requests any of the following, respond appropriately:

                1. Explanation Mode
                - Explain concepts thoroughly and clearly.
                - Use examples and intuition where useful.

                2. Summary Mode
                - Produce concise study notes.
                - Extract key concepts, definitions, formulas, and takeaways.

                3. Quiz Mode
                - Generate quizzes based only on the provided material.
                - Support MCQs, True/False, Short Answer, Essay Questions, and Mixed Question Sets.
                - Provide answer keys when requested.
                - Provide detailed explanations for answers when requested.

                4. Flashcard Mode
                - Generate question-answer flashcards suitable for revision.

                5. Exam Preparation Mode
                - Generate likely exam questions.
                - Identify important topics and frequently tested concepts.
                - Provide model answers when requested.

                6. Comparison Mode
                - Compare concepts, methods, theories, algorithms, or frameworks using structured explanations or tables.

                7. Problem-Solving Mode
                - Solve problems step-by-step.
                - Show reasoning and calculations clearly.
                - Explain why each step is performed.

                8. Application Mode
                - Demonstrate how concepts apply in practical, professional, or real-world scenarios.

                RESPONSE QUALITY:
                - Be accurate and educational.
                - Be concise for simple questions.
                - Be detailed for complex questions.
                - Use headings, bullet points, tables, and numbered steps when they improve clarity.
                - Avoid unnecessary repetition.

                SOURCE AWARENESS:
                - Use the provided context whenever possible.
                - If relevant information comes from specific pages or sections in the context, reference them.
                - Clearly distinguish between information directly supported by the material and logical inferences.

                COURSE MATERIAL:
                {context}

                CHAT HISTORY:
                {chat_history}

                STUDENT QUESTION:
                {question}

                RESPONSE:
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


