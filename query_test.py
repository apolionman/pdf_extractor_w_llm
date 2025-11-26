from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_ollama import OllamaLLM
from langchain_core.prompts.prompt import PromptTemplate
from fastapi.responses import StreamingResponse
from langchain.memory import ConversationBufferMemory
from collections import defaultdict
from app.util.query_handler2 import Neo4jQueryMaster
from typing import Generator
import os, json
from dotenv import load_dotenv
load_dotenv('.env')

session_memories = defaultdict(lambda: ConversationBufferMemory(memory_key="chat_history", return_messages=True))

def get_session_memory(session_id: str) -> ConversationBufferMemory:
    return session_memories[session_id]

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
LLM_MODEL = "gemma3:27b"
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
config = KG_DICT['supplement_kg']
graph = Neo4jGraph(
    url=config["url"],
    username=config["username"],
    password=config["password"]
)
memory = get_session_memory("default")

handler = Neo4jQueryMaster(graph=graph, llm=llm, memory=memory)
handler.query('What supplement can help increase stress and degrade cognitive function for an adult female with depression and fatigue without conflicting with imipramine?')
