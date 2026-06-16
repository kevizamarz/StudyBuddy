# Study Buddy

A Streamlit-based study assistant that loads PDF course material, creates embeddings with Hugging Face, and answers questions using Groq LLMs.

## Features

- Upload PDF documents for knowledge ingestion
- Split PDF content into searchable chunks
- Create and persist a Chroma vector store
- Query the document using a Groq model
- Clear chat history and upload new files

## Go check this out

Visit: https://studybuddypro.streamlit.app

## Notes

- The app stores embeddings in `chroma_db/` by default.
- Add any API keys to Streamlit secrets or avoid committing them to version control.
