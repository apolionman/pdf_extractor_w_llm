def transcribe(audio_input: Union[str, bytes]) -> str:
    """
    Transcribes audio from a file path or audio bytes using Whisper.
    Args:
        audio_input (str or bytes): Path to an audio file or raw audio bytes.
    Returns:
        str: The transcribed text.
    """
    # If input is bytes, write to a temporary file
    if isinstance(audio_input, bytes):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
            temp_audio.write(audio_input)
            temp_audio_path = temp_audio.name
    elif isinstance(audio_input, str) and os.path.exists(audio_input):
        temp_audio_path = audio_input
    else:
        raise ValueError("audio_input must be a valid file path or audio bytes")
    # Load and transcribe the audio file
    result = stt.transcribe(temp_audio_path, fp16=False)
    text = result["text"].strip()
    # Clean up if we created a temp file
    if isinstance(audio_input, bytes):
        os.remove(temp_audio_path)
    return text




###
# def format_schema_for_prompt(raw_schema_data):
#         node_props = {}
#         relationships = set()

#         for entry in raw_schema_data:
#             if entry["type"] == "node":
#                 label = entry["label"]
#                 prop = entry["property"]
#                 if label not in node_props:
#                     node_props[label] = set()
#                 node_props[label].add(prop)
#             elif entry["type"] == "relationship":
#                 rel_type = entry["property"]
#                 start = entry["start"]
#                 end = entry["end"]
#                 relationships.add((rel_type, start, end))

#         nodes_formatted = "\n".join(
#             f"- {label} (properties: {', '.join(sorted(props))})"
#             for label, props in node_props.items()
#         )

#         rels_formatted = "\n".join(
#             f"- {rel_type} (from {start} to {end})"
#             for rel_type, start, end in sorted(relationships)
#         )

#         return f"Nodes:\n{nodes_formatted}\n\nRelationships:\n{rels_formatted}"

#     # Get full schema including relationships
#     raw_schema = graph.query("""
#     CALL apoc.meta.schema()
#     YIELD value as schema
#     UNWIND keys(schema) as label
#     WITH label, schema[label] as data
#     OPTIONAL MATCH (n:label)
#     RETURN {
#         label: label,
#         type: data.type,
#         properties: keys(data.properties),
#         relationships: [rel in data.relationships | {
#             type: rel.type,
#             direction: rel.direction,
#             label: rel.label
#         }]
#     } as nodeInfo
#     """)

#     formatted_schema = format_schema_for_prompt(raw_schema)

#     CYPHER_GENERATION_TEMPLATE = """
#     Task: Generate Cypher statements to query a Neo4j graph database.

#     Instructions:
#     - Generate ONLY the Cypher query - no explanations or additional text.
#     - Use only the provided schema information.
#     - Use proper Cypher syntax with MATCH clauses.
#     - Always use RETURN at the end to return specific properties.
#     - For random sampling, use ORDER BY rand() LIMIT.
#     - Do not use array expansion operators like [*].
#     - Return specific node properties rather than entire nodes.

#     Schema:
#     {schema}

#     Question: {question}

#     Cypher Query:
#     """

#     CYPHER_GENERATION_PROMPT = PromptTemplate(
#         input_variables=["schema", "question"], 
#         template=CYPHER_GENERATION_TEMPLATE
#     )

#     # Create the QA chain
#     chain = GraphCypherQAChain.from_llm(
#         llm=llm,
#         graph=graph,
#         cypher_prompt=CYPHER_GENERATION_PROMPT,
#         verbose=True,
#         validate_cypher=True,  # Validate Cypher before execution
#         allow_dangerous_requests=True,
#         stream=True,
#         top_k=10  # Limit number of results
#     )