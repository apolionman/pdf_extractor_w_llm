from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_ollama import OllamaLLM
from langchain_core.prompts.prompt import PromptTemplate
import os
import re
from neo4j.exceptions import CypherSyntaxError, ClientError
from typing import Dict, Any, List, Optional

class Neo4jQueryMaster:
    def __init__(self, graph, llm):
        self._init_graph_connection(graph)
        self._init_llm(llm)
        self._cache_schema()
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
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to cache schema after {max_retries} attempts: {str(e)}")
                continue

    def _fetch_node_labels(self) -> List[str]:
        result = self.graph.query("CALL db.labels() YIELD label RETURN label")
        return [r["label"] for r in result if "label" in r]

    def _fetch_relationship_types(self) -> List[str]:
        result = self.graph.query("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
        return [r["relationshipType"] for r in result if "relationshipType" in r]

    def _generate_schema_info(self) -> str:
        return (
            f"Available Node Labels: {', '.join(sorted(self.node_labels))}\n"
            f"Available Relationships: {', '.join(sorted(self.relationship_types))}"
        )

    def _setup_query_chain(self):
        """Configure the query chain with proper input variables"""
        self.cypher_prompt = PromptTemplate(
            input_variables=["question", "schema_info", "node_labels", "relationship_types"],

            template=self._get_prompt_template()
        )
        
        self.chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            temperature=0.0,
            graph=self.graph,
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
            15. Stictly! always wrap property names that contain spaces, hyphens, or special characters in plain backticks. For example:
                - Use rrh.`Age group_Min` instead of rrh.Age group_Min
                - Use rrh.`Ref_range-Female_Max` instead of rrh.Ref_range-Female_Max
                - Do NOT escape the backticks — use `rrh.`Property Name`` (not rrh.\`Property Name\`)
            16. Do NOT use `OR` inside `MATCH` or pattern expressions. Use separate `OPTIONAL MATCH` or `MATCH` statements instead.
            17. Do NOT write natural-language expressions like `WITH any <label> nodes`. Always use `MATCH (n:Label)` to select nodes.
            18. All node retrievals must begin with a `MATCH` clause, never a `WITH`.
            19. Every variable (e.g., n, rrh) used in RETURN must be introduced with a MATCH or WITH clause.
            20. Always assign a variable to each relationship in MATCH clauses (e.g., `[r:RELATION_TYPE]`, not `[:RELATION_TYPE]`) — this is required when using `type(r)` in RETURN or WITH.
            21. Never use patterns like `type(n)-[r:REL]->(m)` in RETURN or WITH. The `type()` function only takes a relationship variable, not a full pattern. Instead, assign the relationship to a variable in MATCH and use `type(r)` directly.

            Schema Info:
            {schema_info}

            Question: {question}

            Cypher Query:
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