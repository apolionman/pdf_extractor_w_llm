from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_ollama import OllamaLLM
from langchain_core.prompts.prompt import PromptTemplate
from fastapi.responses import StreamingResponse
from app.util.query_handler2 import Neo4jQueryMaster
from typing import Generator
import os, json
from dotenv import load_dotenv
load_dotenv('.env')
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
llm = OllamaLLM(
    model=LLM_MODEL,
    temperature=0.0,
    num_ctx=32768,
    base_url=OLLAMA_BASE_URL
)
KG_DICT = {
    "prime_kg":{
        "url": "bolt://94.202.21.171:8687",
        "username": "neo4j",
        "password": "prime007!"
    },
    "supplement_kg":{
        "url": "bolt://94.202.21.171:8887",
        "username": "neo4j",
        "password": "supplement007!"
    },
    "wear_kg":{
        "url": "bolt://94.202.21.171:8787",
        "username": "neo4j",
        "password": "wear007!"
    }
}
config = KG_DICT['prime_kg']
graph = Neo4jGraph(
    url=config["url"],
    username=config["username"],
    password=config["password"]
)


handler = Neo4jQueryMaster(graph=graph, llm=llm)
handler.query('can you give me random drug')
