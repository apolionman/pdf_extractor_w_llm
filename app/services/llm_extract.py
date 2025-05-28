import uuid, json, re, json, os, shutil, io, csv
# LangChain for local usage
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM  # For local Mistral model calls
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from app.util.tools import *
from typing import List, Dict
from dotenv import load_dotenv
from fastapi import HTTPException

OLLAMA_BASE_URL=os.getenv("OLLAMA_BASE_URL")
LLM_MODEL=os.getenv("LLM_MODEL")
EMBEDDING_MODEL=os.getenv("EMBEDDING_MODEL")

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

class SentenceTransformersEmbeddings(Embeddings):
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Encode a list of texts into vectors."""
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        """Encode a single text (query) into a vector."""
        return self.model.encode(text, show_progress_bar=False).tolist()

embeddings = SentenceTransformersEmbeddings(EMBEDDING_MODEL)

llm = OllamaLLM(
    model=LLM_MODEL,
    temperature=0.0,
    num_ctx=32768,
    base_url=OLLAMA_BASE_URL
    # max_tokens, top_p, etc. can also be set here
)

pdf_system_prompt = """
You are an AI data extractor for clinical and wearable health technology documents. Your task is to extract **only explicitly stated** information from the text related to the following **targeted keyword fields**, and output the result in a **flat JSON object** format (field-value pairs). Do not make assumptions or add inferred content. Use "not specified" if the information is missing or not clearly mentioned.

Target fields (JSON keys):
{{
    "wearable_biosensor": "string",
    "healthcare_monitoring": "string",
    "biomarkers": "[...]",
    "bioligical_fluids": "[...]",
    "physiological_conditions": "[...]",
    "associated_conditions": "[...]",
    "monitoring_methods": "[...]",
    "wearable_sensors": "[...]",
    "device_type": "[...]",
    "device_brand": "[...]",
    "device_model": "string",
    "monitoring_features": "[...]",
    "accuracy": "string",
    "physiological_parameters": "[...]"
}}

RULES:
- Only use exact terms or values directly from the document.
- For "FDA Status/Year/AP", extract all 3 components if present; otherwise return "not specified".
- Always return a value for every keyword field, using "not specified" if missing.
- Output a single flat JSON object per document, with each key corresponding to a field name and its value as a string.
- Do not return nested objects, arrays, or any content outside the JSON object.
"""

user_prompt = "Extract the listed keyword-based medical device information from the document in flat JSON format. Use 'not specified' where information is missing. Do not infer data."

pdf_prompt_template = ChatPromptTemplate.from_messages([
    ("system", pdf_system_prompt),
    ("human", "{context}")
])


def create_vector_store_from_txt(text_path):
    """
    Loads a text file, splits it into manageable chunks, and creates a FAISS-based vector store
    from these chunks using Hugging Face Embeddings.
    """
    vector_store_path = "vector_files"
    os.makedirs(vector_store_path, exist_ok=True)

    file_name = os.path.splitext(os.path.basename(text_path))[0]
    file_extn = os.path.splitext(os.path.basename(text_path))[1]

    # 1. Load the raw text as Document objects
    loader = TextLoader(text_path, encoding='utf-8')
    pages = loader.load()
    if not pages:
        raise ValueError("No valid text data to create vector store.")

    # 2. Split the documents into smaller chunks
    #    Adjust `chunk_size` and `chunk_overlap` to suit your use case
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=10000, chunk_overlap=500
        )
    chunked_docs = text_splitter.split_documents(pages)
    if not chunked_docs:
        raise ValueError("No valid chunks to create vector store.")

    # 3. Create embeddings for each chunk and store them in a FAISS index
    vector_store = FAISS.from_documents(chunked_docs, embeddings)
    vector_store.save_local(vector_store_path)
    print(f"Vector Store Created for {file_name}{file_extn}")

    return vector_store_path


def retrieve_info(vector_db_path, user_prompt, prompt_template):
    """
    Given a path to a local FAISS index, retrieve relevant chunks and build a
    StuffDocumentsChain with a local LLM (Ollama).
    """
    vector_db = FAISS.load_local(
        vector_db_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vector_db.as_retriever()
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, stuff_documents_chain)

    response = rag_chain.invoke({"context": user_prompt})["answer"]
    return response

def clean_trailing_commas(json_like_str: str) -> str:
    # Remove trailing commas before } or ]
    cleaned = re.sub(r",(\s*[}\]])", r"\1", json_like_str)
    return cleaned

async def summarization(file_path):
    """
    Main summary function.
    Extracts structured health technology information from a PDF and returns it in JSON format.
    """
    if not isinstance(file_path, str):
        raise TypeError(f"Invalid file_path type: expected str, got {type(file_path)} — value: {file_path}")

    summary_json = {}

    if file_path.endswith(".pdf"):
        doc_text, doc_text_path = get_text_from_pdf_paddle(file_path)

        if doc_text_path is not None and doc_text.strip():
            doc_vector_db = create_vector_store_from_txt(doc_text_path)

            summary_raw = retrieve_info(doc_vector_db, user_prompt, pdf_prompt_template)

            if not summary_raw.strip():
                raise ValueError("[ERROR] ❌ Empty response from LLM.")

            try:
                summary_parsed = json.loads(summary_raw)

                if not isinstance(summary_parsed, dict):
                    raise ValueError("[ERROR] Expected a flat dictionary as JSON output.")

                summary_json = summary_parsed

            except json.JSONDecodeError as err:
                print(f"[WARNING] JSON parsing failed: {err}")
                cleaned_raw = clean_trailing_commas(summary_raw)
                try:
                    summary_parsed = json.loads(cleaned_raw)

                    if not isinstance(summary_parsed, dict):
                        raise ValueError("[ERROR] Cleaned JSON still not a dictionary.")

                    summary_json = summary_parsed

                    print("[INFO] ✅ Successfully parsed after cleaning trailing commas.")
                except Exception as fallback_err:
                    error_log_path = os.path.join("/tmp", f"llm_output_error_{uuid.uuid4()}.txt")
                    with open(error_log_path, "w", encoding="utf-8") as f:
                        f.write(summary_raw)
                    print(f"[ERROR] Both original and cleaned JSON failed: {fallback_err}")
                    raise ValueError(f"Invalid JSON from LLM. Original error: {err}")

            shutil.rmtree(doc_vector_db)

    return summary_json
