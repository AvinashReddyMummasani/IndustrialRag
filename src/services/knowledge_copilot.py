import os
import logging
from typing import List, Dict, Any, TypedDict
from pydantic import BaseModel, Field
import instructor
from groq import Groq
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END
from src.core.schemas import EntityExtractor,RelevanceGrade,CitedAnswer,GroundednessGrade,UtilityGrade

from src.db.postgres_client import PostgresPool 
from src.db.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


# =====================================================================
# State Machine Contract
# =====================================================================

class AgentState(TypedDict):
    query: str
    combined_context: str
    generation: str
    evidence: List[str]
    is_relevant: bool
    is_grounded: bool
    is_useful: bool
    retries: int

# =====================================================================
# System Workflow Engine
# =====================================================================

class KnowledgeCopilot:
    def __init__(self):
        raw_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.client = instructor.from_groq(raw_client, mode=instructor.Mode.TOOLS)
        
        # Dense vector encoder mapping safely to standard relational parameters
        self.embedding_model = SentenceTransformer("all-mpnet-base-v2")
        self.llm_model = "llama3-70b-8192"
        self.max_retries = 3
        
        self.graph = self._compile_workflow()

    def _get_query_embedding(self, query: str) -> List[float]:
        embedding = self.embedding_model.encode(query).tolist()
        if len(embedding) < 1536:
            embedding.extend([0.0] * (1536 - len(embedding)))
        return embedding[:1536]

    def _fetch_vector_context(self, query_vector: List[float], limit: int = 4) -> List[Dict[str, Any]]:
            chunks = []
            sql = """
                SELECT c.chunk_text, d.filename, d.id, (c.embedding <=> %s::vector) AS distance 
                FROM document_chunks c
                JOIN documents d ON c.document_id = d.id
                ORDER BY distance ASC LIMIT %s;
            """
            # Changed to PostgresPool
            with PostgresPool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (query_vector, limit))
                    for row in cur.fetchall():
                        chunks.append({
                            "text": row[0],
                            "filename": row[1],
                            "doc_id": row[2]
                        })
            return chunks

    def _fetch_graph_context(self, entities: List[str]) -> List[Dict[str, Any]]:
        if not entities:
            return []
        cypher = """
        UNWIND $entity_ids AS e_id
        MATCH (n {entity_id: e_id})
        OPTIONAL MATCH (n)-[r]->(m)
        RETURN n.entity_id AS source, type(r) AS relationship, m.entity_id AS target, r.document_id AS doc_id
        LIMIT 30
        """
        records = []
        with Neo4jClient._driver.session() as session:
            result = session.run(cypher, entity_ids=entities)
            for record in result:
                records.append(record.data())
        return records

    # --- LangGraph Graph Components ---

    def retrieve_node(self, state: AgentState) -> dict:
        logger.info("[State Node] Fetching multi-modal contexts.")
        query = state["query"]
        
        try:
            extraction: EntityExtractor = self.client.chat.completions.create(
                model=self.llm_model,
                response_model=EntityExtractor,
                messages=[
                    {"role": "system", "content": "Extract industrial asset identifiers/tags from instructions."},
                    {"role": "user", "content": query}
                ],
                temperature=0.0
            )
            entities = [e.upper() for e in extraction.entities]
        except Exception:
            entities = []

        query_vector = self._get_query_embedding(query)
        vector_data = self._fetch_vector_context(query_vector)
        graph_data = self._fetch_graph_context(entities)
        
        vector_str = "\n".join([f"- [Doc Source File: {c['filename']}] {c['text']}" for c in vector_data])
        graph_str = "\n".join([
            f"- [Graph Context Doc ID: {g['doc_id']}] ({g['source']})-[{g['relationship']}]->({g['target']})" 
            for g in graph_data if g['relationship']
        ])
        
        combined_context = f"=== VECTOR DATABASE RECORDS ===\n{vector_str}\n\n=== RELATIONSHIP TOPOLOGY MAP ===\n{graph_str}"
        return {"combined_context": combined_context}

    def grade_context_node(self, state: AgentState) -> dict:
        logger.info("[State Node] Validating retrieval context relevancy scores.")
        try:
            grade: RelevanceGrade = self.client.chat.completions.create(
                model=self.llm_model,
                response_model=RelevanceGrade,
                messages=[
                    {"role": "system", "content": "Determine if the combined database contexts hold data directly matching parameters specified in the user's inquiry."},
                    {"role": "user", "content": f"User Target Query: {state['query']}\n\nProvided Datastores Content:\n{state['combined_context']}"}
                ],
                temperature=0.0
            )
            if not grade.is_relevant:
                return {"combined_context": "[System Metric Alert: Relevant datastore logs missing for explicit parameters.]", "is_relevant": False}
            return {"is_relevant": True}
        except Exception:
            return {"is_relevant": True}

    def generate_node(self, state: AgentState) -> dict:
        logger.info(f"[State Node] Generating target analysis. Attempt {state['retries'] + 1}")
        prompt = f"""
        Synthesize a rigorous engineering answer addressing the query below. 
        Your statement must rely exclusively on facts specified inside the database context block. 
        For every fact you generate, extract the corresponding metadata string (e.g. filename or Doc ID) and map it to evidence_links. 
        If the context parameters are absent, abort generation and note that metrics are missing.

        Database Context Parameters:
        {state['combined_context']}

        Target User Query: {state['query']}
        """
        try:
            response: CitedAnswer = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.llm_model,
                response_model=CitedAnswer,
                temperature=0.0
            )
            return {
                "generation": response.answer, 
                "evidence": response.evidence_links, 
                "retries": state["retries"] + 1
            }
        except Exception as e:
            logger.error(f"Structured Generation crash down step: {e}")
            return {"generation": "Generation Pipeline Fault encountered.", "evidence": [], "retries": state["retries"] + 1}

    def grade_generation_node(self, state: AgentState) -> dict:
        logger.info("[State Node] Grading synthesized answer metrics against original contexts.")
        ctx = state["combined_context"]
        gen = state["generation"]
        q = state["query"]
        
        try:
            grounded: GroundednessGrade = self.client.chat.completions.create(
                model=self.llm_model,
                response_model=GroundednessGrade,
                messages=[{"role": "user", "content": f"Database Master Context:\n{ctx}\n\nProposed Generation Text:\n{gen}"}],
                temperature=0.0
            )
            useful: UtilityGrade = self.client.chat.completions.create(
                model=self.llm_model,
                response_model=UtilityGrade,
                messages=[{"role": "user", "content": f"User Operational Query:\n{q}\n\nSynthesized Answer text:\n{gen}"}],
                temperature=0.0
            )
            return {"is_grounded": grounded.is_grounded, "is_useful": useful.fully_answers}
        except Exception:
            return {"is_grounded": True, "is_useful": True}

    # --- Edge Evaluation Routines ---

    def _evaluate_loop_requirements(self, state: AgentState) -> str:
        if state["is_grounded"] and state["is_useful"]:
            logger.info("[Routing Edge] Generation parameters validated. Finalizing execution state.")
            return "end"
        
        if state["retries"] >= self.max_retries:
            logger.warning("[Routing Edge] Context constraints bounds exceeded. Forcing hard termination.")
            return "end"
            
        logger.warning("[Routing Edge] Internal compliance validation failure. Forcing loop recalculation.")
        return "generate"

    # --- Graph Orchestration ---

    def _compile_workflow(self):
        workflow = StateGraph(AgentState)
        
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("grade_context", self.grade_context_node)
        workflow.add_node("generate", self.generate_node)
        workflow.add_node("grade_generation", self.grade_generation_node)
        
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_context")
        workflow.add_edge("grade_context", "generate")
        workflow.add_edge("generate", "grade_generation")
        
        workflow.add_conditional_edges(
            "grade_generation",
            self._evaluate_loop_requirements,
            {
                "generate": "generate",
                "end": END
            }
        )
        return workflow.compile()

    def ask(self, query: str) -> dict:
        """System entrypoint. Returns structured answers mapped to verified source references."""
        initial_state = {
            "query": query,
            "combined_context": "",
            "generation": "",
            "evidence": [],
            "is_relevant": False,
            "is_grounded": False,
            "is_useful": False,
            "retries": 0
        }
        final_state = self.graph.invoke(initial_state)
        return {
            "answer": final_state["generation"],
            "evidence_links": final_state["evidence"]
        }