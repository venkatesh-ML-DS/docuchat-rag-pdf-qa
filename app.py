import os
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_KEY)

def get_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    return "".join([page.extract_text() for page in reader.pages])

def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    pdf = request.files["pdf"]
    text = get_pdf_text(pdf)
    chunks = get_chunks(text)
    vs = FAISS.from_texts(chunks, embedding=get_embeddings())
    vs.save_local("faiss_index")
    return jsonify({"message": "PDF processed successfully!"})

@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("question")
    vs = FAISS.load_local("faiss_index", get_embeddings(), allow_dangerous_deserialization=True)
    docs = vs.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {question}"
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return jsonify({"answer": response.text})

if __name__ == "__main__":
    app.run(debug=True)