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
EMBEDDING_MODEL='/embedded/models--sentence-transformers--all-MiniLM-L12-v2/snapshots/c004d8e3e901237d8fa7e9fff12774962e391ce5'

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

param = {
    "wearable_biosensor": {"type": str, "description": "The name or type of the wearable biosensor device."},
    "healthcare_monitoring": {"type": str, "description": "The healthcare context or purpose for which the device is used."},
    "biomarkers": {"type": list, "description": "Biological markers that the device monitors, such as glucose, lactate, etc."},
    "bioligical_fluids": {"type": list, "description": "Types of biological fluids analyzed, e.g., sweat, blood, saliva."},
    "physiological_conditions": {"type": list, "description": "Physiological conditions assessed or measured by the device."},
    "associated_conditions": {"type": list, "description": "Medical conditions associated with the use of this device (e.g., diabetes, cardiovascular disease)."},
    "monitoring_methods": {"type": list, "description": "Methods used for monitoring, such as optical, electrochemical, etc."},
    "wearable_sensors": {"type": list, "description": "Types of sensors integrated into the device (e.g., ECG sensor, accelerometer)."},
    "device_type": {"type": list, "description": "Form factor or category of the device, such as wristband, patch, smart ring."},
    "device_brand": {"type": list, "description": "Brand(s) of the device if mentioned."},
    "device_model": {"type": str, "description": "Specific model name or number of the device."},
    "monitoring_features": {"type": list, "description": "Key features or functionalities the device offers for monitoring health."},
    "accuracy": {"type": str, "description": "Reported accuracy of the device or any validation metrics."},
    "physiological_parameters": {"type": list, "description": "Specific physiological parameters being measured (e.g., heart rate, SpO2)."}
}


def prepare_prompt(param):
    # Generate a nicely formatted string for each field with its description
    fields_with_description = "\n".join([
        f'- "{key}" ({val["type"]}): {val["description"]}'
        for key, val in param.items()
    ])
    
    pdf_system_prompt = f"""
        You are an AI data extractor for clinical and wearable health technology documents. Your task is to extract **only explicitly stated** information from the text related to the following **targeted keyword fields**, and output the result in a **flat JSON object** format (field-value pairs). Do not make assumptions or add inferred content. Use "not specified" if the information is missing or not clearly mentioned.

        Target fields with descriptions:
        {fields_with_description}

        RULES:
        - Only use exact terms or values directly from the document.
        - For "FDA Status/Year/AP", extract all 3 components if present; otherwise return "not specified".
        - Always return a value for every keyword field, using "not specified" if missing.
        - Output must be a SINGLE FLAT JSON OBJECT with no nesting.
        - Example format: {{
            "wearable_biosensor": "value or not specified",
            "healthcare_monitoring": "value or not specified",
            "biomarkers": ["list", "of", "values"] or "not specified"
          }}
        - Do not return nested objects, arrays, or any content outside the JSON object.
        """
    return pdf_system_prompt



user_prompt = "Extract the listed keyword-based medical device information from the document in flat JSON format. Use 'not specified' where information is missing. Do not infer data."

pdf_prompt_template = ChatPromptTemplate.from_messages([
    ("system", prepare_prompt(param)),
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

    response = rag_chain.invoke({"input": user_prompt})["answer"]
    return response

def clean_trailing_commas(json_like_str: str) -> str:
    # Remove trailing commas before } or ]
    cleaned = re.sub(r",(\s*[}\]])", r"\1", json_like_str)
    # Fix missing quotes around keys
    cleaned = re.sub(r'(\{|\,)\s*([a-zA-Z0-9_]+)\s*:', r'\1"\2":', cleaned)
    # Fix missing quotes around string values
    cleaned = re.sub(r':\s*([a-zA-Z][a-zA-Z0-9_]*)\s*([,\}])', r': "\1"\2', cleaned)
    return cleaned

async def summarization(file_path, user_prompt_data=None):
    if not isinstance(file_path, str):
        raise TypeError(f"Invalid file_path type: expected str, got {type(file_path)} — value: {file_path}")

    summary_json = {}

    if file_path.endswith(".pdf"):
        doc_text, doc_text_path = get_text_from_pdf_paddle(file_path)

        if doc_text_path is not None and doc_text.strip():
            doc_vector_db = create_vector_store_from_txt(doc_text_path)

            # Dynamically generate prompt template
            dynamic_param = user_prompt_data if user_prompt_data else param
            dynamic_template = ChatPromptTemplate.from_messages([
                ("system", prepare_prompt(dynamic_param)),
                ("human", "{context}")
            ])

            summary_raw = retrieve_info(doc_vector_db, user_prompt, dynamic_template)

            if not summary_raw.strip():
                raise ValueError("[ERROR] ❌ Empty response from LLM.")

            # Enhanced JSON parsing
            try:
                # First try to parse directly
                summary_parsed = json.loads(summary_raw)
                
                # If we get here, parsing succeeded but might be wrong format
                if isinstance(summary_parsed, dict):
                    # Check if the response is nested like in the error
                    if any(isinstance(v, dict) for v in summary_parsed.values()):
                        # If nested, try to flatten it
                        flattened = {}
                        for key, value in summary_parsed.items():
                            if isinstance(value, dict):
                                # Just take the first simple value we find
                                for subkey, subvalue in value.items():
                                    if isinstance(subvalue, (str, int, float, bool)) or subvalue is None:
                                        flattened[key] = subvalue
                                        break
                                else:
                                    flattened[key] = "not specified"
                            else:
                                flattened[key] = value
                        summary_json = flattened
                    else:
                        summary_json = summary_parsed
                else:
                    raise ValueError("[ERROR] Expected a dictionary as JSON output.")
                    
            except json.JSONDecodeError as err:
                # Try to fix common JSON issues
                cleaned_raw = clean_trailing_commas(summary_raw)
                
                # Try to extract just the JSON part if there's extra text
                json_match = re.search(r'\{.*\}', cleaned_raw, re.DOTALL)
                if json_match:
                    cleaned_raw = json_match.group(0)
                
                try:
                    summary_parsed = json.loads(cleaned_raw)
                    if not isinstance(summary_parsed, dict):
                        raise ValueError("[ERROR] Cleaned JSON still not a dictionary.")
                    summary_json = summary_parsed
                except Exception as fallback_err:
                    error_log_path = os.path.join("/tmp", f"llm_output_error_{uuid.uuid4()}.txt")
                    with open(error_log_path, "w", encoding="utf-8") as f:
                        f.write(f"Original error: {err}\n\nAttempted to clean:\n{cleaned_raw}\n\nOriginal output:\n{summary_raw}")
                    raise ValueError(f"Invalid JSON from LLM. Original error: {err}")

            shutil.rmtree(doc_vector_db)

    return summary_json

