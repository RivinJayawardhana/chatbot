# app.py

import os
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from langchain.chains import RetrievalQAWithSourcesChain
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

app = Flask(__name__)
CORS(app)

VECTOR_DB_PATH = "faiss_store_combined.pkl"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or "your-fallback-key"

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-002",
    google_api_key=GOOGLE_API_KEY
)

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.json
        query = data.get("question")

        if not query:
            return jsonify({"error": "Missing question"}), 400

        if not os.path.exists(VECTOR_DB_PATH):
            return jsonify({"error": "Vector DB not found"}), 500

        with open(VECTOR_DB_PATH, "rb") as f:
            vectorstore = pickle.load(f)

        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )

        result = chain({"question": query}, return_only_outputs=True)
        return jsonify({
            "answer": result.get("answer", "No answer generated."),
            "sources": result.get("sources", "")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def index():
    return "🔍 Chatbot API is running."

if __name__ == "__main__":
    app.run(debug=True)
