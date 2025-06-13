from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_ollama import OllamaLLM
from langchain_core.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory
import re, json, os
from neo4j.exceptions import CypherSyntaxError, ClientError
from typing import Dict, Any, List, Optional, Tuple

class Neo4jQueryMaster:
    def __init__(self, graph, llm, memory=None):
        self._init_graph_connection(graph)
        self._init_llm(llm)
        self._cache_schema()
        self.memory = memory or ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self._setup_query_chain()

    def _init_graph_connection(self, graph):
        """Initialize and verify Neo4j connection"""
        self.graph = graph
        # Verify connection
        try:
            self.graph.query("RETURN 1 AS test")
        except Exception as e:
            raise ConnectionError(f"Neo4j connection failed: {str(e)}")

    def _init_llm(self, llm):
        """Initialize LLM with error handling"""
        try:
            self.llm = llm
        except Exception as e:
            raise RuntimeError(f"LLM initialization failed: {str(e)}")

    def _cache_schema(self):
        """Cache schema information with retries"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.node_labels = self._fetch_node_labels()
                self.relationship_types = self._fetch_relationship_types()
                self.schema_info = self._generate_schema_info()
                print(self.schema_info)
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to cache schema after {max_retries} attempts: {str(e)}")
                continue
            
    def _extract_schema(self):
        # Query Neo4j to get the full schema
        result = self.graph.query("CALL db.schema.visualization()")[0]
        nodes = list({n['name'] for n in result['nodes'] if 'name' in n})
        relationships = [
            {
                "from": r[0]["labels"][0] if isinstance(r[0], dict) and "labels" in r[0] else r[0],
                "type": r[1]["type"] if isinstance(r[1], dict) and "type" in r[1] else r[1],
                "to": r[2]["labels"][0] if isinstance(r[2], dict) and "labels" in r[2] else r[2]
            }
            for r in result['relationships']
        ]
        return {"nodes": nodes, "relationships": relationships}

    def _fetch_node_labels(self) -> List[str]:
        result = self.graph.query("CALL db.labels() YIELD label RETURN label")
        return [r["label"] for r in result if "label" in r]

    def _fetch_relationship_types(self) -> List[Tuple[List[str], str, List[str]]]:
        query = """
        MATCH (a)-[r]->(b)
        RETURN DISTINCT labels(a) AS fromLabels, type(r) AS relationshipType, labels(b) AS toLabels
        """
        result = self.graph.query(query)
        return [
            (r["fromLabels"], r["relationshipType"], r["toLabels"])
            for r in result
            if "relationshipType" in r
        ]

    def _generate_schema_info(self) -> str:
        # Format relationships as "fromLabel->relationshipType->toLabel"
        formatted_relationships = [
            f"({', '.join(from_labels)})-[:{rel_type}]->({', '.join(to_labels)})"
            for from_labels, rel_type, to_labels in self.relationship_types
        ]
        
        return (
            f"Available Node Labels: {', '.join(sorted(self.node_labels))}\n"
            f"Available Relationships: {', '.join(sorted(formatted_relationships))}"
        )

    def _setup_query_chain(self):
        """Configure the query chain with proper input variables"""
        self.cypher_prompt = PromptTemplate(
            input_variables=["question", 
                             "schema_info", 
                             "node_labels", 
                             "relationship_types",
                             "chat_history"
                             ],

            template=self._get_prompt_template()
        )
        
        self.chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            temperature=0.0,
            graph=self.graph,
            memory=self.memory, 
            cypher_prompt=self.cypher_prompt,
            verbose=True,
            validate_cypher=True,
            top_k=10,
            max_retries=1,
            cypher_query_timeout=15,
            return_direct=True,
            return_intermediate_steps=False,
            allow_dangerous_requests=True
        )

    def _get_prompt_template(self) -> str:
        return """
        You are a Neo4j Cypher expert. Generate ONLY the Cypher query to answer the question.

        Rules:
            1. Use only these node labels: {node_labels}
            2. Use only these relationship types: {relationship_types}
            3. Always include LIMIT unless asking for counts
            4. Never use variable-length paths ([*])
            5. Return specific properties, not whole nodes
            6. Use `labels(n)` instead of `n.label`
            7. Use `type(r)` instead of `r.type`
            8. Avoid using `type()` on nodes
            9. Any variable used in RETURN must be defined or carried in WITH
            10. If you use `n.name` in RETURN, you must either:
                - Keep `n` in WITH, or
                - Use `n.name AS node_name` in WITH and reference `node_name` in RETURN
            11. Do NOT use `AS` inside `collect()` or other functions. Apply aliasing outside the function.
            12. Use `collect(n.property) AS alias`, not `collect(n.property AS alias)`
            13. In `WITH` clauses, all expressions (e.g., type(r)) must use `AS` to create an alias. You cannot use an unaliased expression in `WITH`.
            14. Do NOT use SQL-style subqueries like `WITH alias AS (...)`. Instead, write the query directly or use `CALL` if subqueries are needed.
            15. Strictly! always wrap property names that contain spaces, hyphens, or special characters in plain backticks. For example:
                - Use rrh.`Age group_Min` instead of rrh.Age group_Min
                - Use rrh.`Ref_range-Female_Max` instead of rrh.Ref_range-Female_Max
                - Do NOT escape the backticks — use `rrh.`Property Name`` (not rrh.\`Property Name\`)
            16. Do NOT use `OR` inside `MATCH` or pattern expressions. Use separate `OPTIONAL MATCH` or `MATCH` statements instead.
            17. Do NOT write natural-language expressions like `WITH any <label> nodes`. Always use `MATCH (n:Label)` to select nodes.
            18. All node retrievals must begin with a `MATCH` clause, never a `WITH`.
            19. Every variable (e.g., n, rrh) used in RETURN must be introduced with a MATCH or WITH clause.
            20. Always assign a variable to each relationship in MATCH clauses (e.g., `[r:RELATION_TYPE]`, not `[:RELATION_TYPE]`) — this is required when using `type(r)` in RETURN or WITH.
            21. Never use patterns like `type(n)-[r:REL]->(m)` in RETURN or WITH. The `type()` function only takes a relationship variable, not a full pattern. Instead, assign the relationship to a variable in MATCH and use `type(r)` directly.
            22. Only respond with a pure Cypher query and strictly do not return any explanation.
            23. Do not use `*` or `~` for wildcards. Use `WHERE node.property CONTAINS ""` for substring matching instead.
            24. Use `WHERE node.property =~ '.*(?i)value.*'` only if you want case-insensitive matching. Otherwise, use `CONTAINS`.
            25. Never use `~` inside `curly brackets` property maps. Always use `WHERE` clauses for pattern or substring matching.

        Schema Info:
        {schema_info}

        Question: {question}

        Cypher Query:

        Example:
            Question: What supplements are useful for a disease like Menkes?
            Schema: Disease-[:HAS_SYMPTOM]->Symptom<-[:MANAGED_BY]-Supplement
        Example Cypher Query:
            MATCH (d:Disease)-[r1:HAS_SYMPTOM]->(s:Symptom)<-[r2:MANAGED_BY]-(supplement:Supplement)
            WHERE d.name CONTAINS 'Menkes'
            RETURN DISTINCT supplement.name AS supplement
            LIMIT 10

        Example:
            Question: What are the health conditions negatively affected by a high heart rate, and what lifestyle changes are recommended for those conditions?
            Schema: Parameter-[:ASSOCIATED_WITH]->Parameter_Value-[:NEGATIVELY_AFFECT]->Health_Condition_Medical-[:RECOMMENDED_LIFESTYLE_CHANGES]->Lifestyle
        Example Cypher Query:
            MATCH (p:Parameter)-[:ASSOCIATED_WITH]->(v:Parameter_Value)
            WHERE p.name CONTAINS 'Heart Rate' AND v.name CONTAINS 'High'
            MATCH (v)-[:NEGATIVELY_AFFECT]->(m:Health_Condition_Medical)
            MATCH (m)-[:RECOMMENDED_LIFESTYLE_CHANGES]->(l:Lifestyle)
            RETURN DISTINCT p.name AS parameter, v.name AS value, m.name AS condition, l.name AS recommended_lifestyle
            LIMIT 10

        Example:
            Question: What medical condition is negatively affected by low blood pressure, and what supplements are recommended for it?
            Schema: Parameter-[:ASSOCIATED_WITH]->Parameter_Value-[:NEGATIVELY_AFFECT]->Health_Condition_Medical-[:RECOMMENDED_SUPPLEMENTS]->Supplements
        Example Cypher Query:
            MATCH (p:Parameter)-[:ASSOCIATED_WITH]->(v:Parameter_Value)
            WHERE (p.name CONTAINS 'Blood Pressure' OR p.name CONTAINS 'Blood') AND v.name CONTAINS 'Low'
            MATCH (v)-[:NEGATIVELY_AFFECT]->(m:Health_Condition_Medical)
            MATCH (m)-[:RECOMMENDED_SUPPLEMENTS]->(s:Supplements)
            RETURN DISTINCT p.name AS parameter, v.name AS value, m.name AS condition, s.name AS recommended_supplement
            LIMIT 10
        
        Example:
            Question: What health condition is negatively affected by a short menstrual cycle, and what blood tests are recommended for it?
            Schema: Parameter-[:ASSOCIATED_WITH]->Parameter_Value-[:NEGATIVELY_AFFECT]->Health_Condition_Medical-[:RECOMMENDED_BLOOD_TESTS]->Blood_Tests
        Example Cypher Query:
            MATCH (p:Parameter)-[:ASSOCIATED_WITH]->(v:Parameter_Value)
            WHERE (p.name CONTAINS 'Menstrual Tracking' OR p.name CONTAINS 'Menstrual') AND v.name CONTAINS 'Short'
            MATCH (v)-[:NEGATIVELY_AFFECT]->(m:Health_Condition_Medical)
            MATCH (m)-[:RECOMMENDED_BLOOD_TESTS]->(b:Blood_Tests)
            RETURN DISTINCT p.name AS parameter, v.name AS value, m.name AS condition, b.name AS recommended_blood_test
            LIMIT 10

        Example:
            Question: What health conditions are negatively affected by low sleep tracking parameters, and what lifestyle changes are recommended for those conditions?
            Schema: Parameter-[:ASSOCIATED_WITH]->Parameter_Value-[:NEGATIVELY_AFFECT]->Health_Condition_Medical-[:RECOMMENDED_LIFESTYLE_CHANGES]->Lifestyle
        Example Cypher Query:
            MATCH (p:Parameter)-[:ASSOCIATED_WITH]->(v:Parameter_Value)
            WHERE (p.name CONTAINS 'Sleep Tracking' OR p.name CONTAINS 'Sleep') AND v.name CONTAINS 'Low'
            MATCH (v)-[:NEGATIVELY_AFFECT]->(m:Health_Condition_Medical)
            MATCH (m)-[:RECOMMENDED_LIFESTYLE_CHANGES]->(l:Lifestyle)
            RETURN DISTINCT p.name AS parameter, v.name AS value, m.name AS condition, l.name AS recommended_lifestyle
            LIMIT 10

        """

    def query(self, question: str) -> Dict[str, Any]:
        inputs = {
            "query": question,
            "schema_info": self.schema_info,
            "node_labels": self.node_labels,
            "relationship_types": self.relationship_types
        }

        result = self.chain.invoke(inputs)

        structured_result = {
            "question": question,
            "cypher_query": result.get("intermediate_steps", [{}])[0].get("cypher_query", "N/A"),
            "data": result.get("result", "No result returned")
        }

        return structured_result


    def _try_llm_query(self, question: str) -> Optional[Dict[str, Any]]:
        """Attempt to use LLM-generated query"""
        try:
            # Prepare the exact input keys expected by the prompt
            inputs = {
                "question": question,
                "schema_info": self.schema_info,
                "node_labels": self.node_labels,
                "relationship_types": self.relationship_types
            }
            
            # The chain only needs question and schema_info
            chain_inputs = {
                "question": question,
                "schema_info": self.schema_info
            }
            
            result = self.chain.invoke(chain_inputs)
            
            if not result or "result" not in result:
                raise ValueError("Empty or invalid result from LLM chain")
                
            return {
                "results": result["result"],
                "query_used": result.get("intermediate_steps", {}).get("query", "unknown"),
                "source": "llm_generated"
            }
        except Exception as e:
            print(f"LLM query attempt failed: {str(e)}")
            return None

    def _try_pattern_query(self, question: str) -> Optional[Dict[str, Any]]:
        """Generate query based on question patterns"""
        try:
            limit = self._extract_limit(question) or 5
            target_label = self._detect_target_label(question)
            
            query = f"""
            MATCH (n:{target_label})
            RETURN n AS result
            LIMIT {limit}
            """
            
            results = self.graph.query(query)
            return {
                "results": results,
                "query_used": query,
                "source": "pattern_match"
            }
        except Exception as e:
            print(f"Pattern query failed: {str(e)}")
            return None

    def _execute_fallback_query(self, question: str) -> Dict[str, Any]:
        """Ultra-reliable fallback query"""
        try:
            limit = self._extract_limit(question) or 5
            query = f"""
            MATCH (n)
            RETURN labels(n) AS labels, properties(n) AS properties
            LIMIT {limit}
            """
            
            results = self.graph.query(query)
            return {
                "results": results,
                "query_used": query,
                "source": "ultimate_fallback"
            }
        except Exception as e:
            return {
                "error": f"All query attempts failed: {str(e)}",
                "question": question
            }

    def _extract_limit(self, question: str) -> Optional[int]:
        match = re.search(r'(?:top|first|limit)?\s*(\d+)', question.lower())
        return int(match.group(1)) if match else None

    def _detect_target_label(self, question: str) -> str:
        """Find the most relevant node label in the question"""
        question_lower = question.lower()
        for label in self.node_labels:
            if label.lower() in question_lower:
                return label
        return self.node_labels[0] if self.node_labels else "Node"