# create_vector_db_from_pdfs.py

import os
import pickle
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def load_pdfs(pdf_paths):
    documents = []
    for file_path in pdf_paths:
        try:
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
    return documents

def create_vector_db_from_pdfs(pdf_paths, output_path="faiss_store_pdfs.pkl"):
    print("📄 Loading PDF documents...")
    pdf_docs = load_pdfs(pdf_paths)

    if not pdf_docs:
        raise ValueError("No documents were loaded from the provided PDFs.")

    print("✂️ Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(pdf_docs)

    print("📐 Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    print(f"💾 Saving vector DB to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(vectorstore, f)

    print("✅ Vector DB created and saved successfully!")

# Example usage
if __name__ == "__main__":
    # Replace with your local PDF file paths
    pdf_files = ["./sample.pdf"]
    create_vector_db_from_pdfs(pdf_files)
