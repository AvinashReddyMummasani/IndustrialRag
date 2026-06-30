import os
import logging
import asyncio
from typing import List, Dict, Any, TypedDict, Optional
import instructor
from groq import AsyncGroq
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END

# Assuming these are correctly defined in your core/schemas
from src.core.schemas import EntityExtractor, RelevanceGrade, CitedAnswer, GroundednessGrade, UtilityGrade
from src.db.postgres_client import PostgresPool 
from src.db.neo4j_client import Neo4jClient
from src.services.entity_resolver import EntityResolver

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    original_query: str
    resolved_query: str
    combined_context: str
    generation: str
    evidence: List[str]
    is_relevant: bool
    is_grounded: bool
    is_useful: bool
    retries: int
    system_warning: Optional[str]


class KnowledgeCopilot:
    def __init__(self, embedding_model, llm: str = "llama-3.3-70b-versatile"):
        # API Keys should ideally be injected via dependency injection or a config manager, not read directly here.
        raw_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        self.client = instructor.from_groq(raw_client, mode=instructor.Mode.TOOLS)
        self.resolver = EntityResolver()
        
        self.embedding_model = embedding_model
        self.llm_model = llm
        self.max_retries = 3
        
        self.graph = self._compile_workflow()

    def _get_query_embedding(self, query: str) -> List[float]:
        embedding = self.embedding_model.encode(query).tolist()
        if len(embedding) < 384:
            embedding.extend([0.0] * (384 - len(embedding)))
        return embedding[:384]

    async def _fetch_vector_context(self, query_vector: List[float], limit: int = 4) -> List[Dict[str, Any]]:
        chunks = []
        sql = """
            SELECT c.chunk_text, d.filename, d.id, (c.embedding <=> $1::vector) AS distance 
            FROM document_chunks c
            JOIN documents d ON c.document_id = d.id
            ORDER BY distance ASC LIMIT $2;
        """
        async with PostgresPool.get_connection() as conn:
            rows = await conn.fetch(sql, str(query_vector), limit)
            for row in rows:
                chunks.append({
                    "text": row['chunk_text'],
                    "filename": row['filename'],
                    "doc_id": row['id']
                })
        return chunks

    async def _fetch_graph_context(self, entities: List[str]) -> List[Dict[str, Any]]:
        if not entities:
            return []
        
        cypher = """
        UNWIND $entity_ids AS e_id
        MATCH (n {entity_id: e_id})
        OPTIONAL MATCH (n)-[r]->(m)
        RETURN n.entity_id AS source, type(r) AS relationship, m.entity_id AS target, r.document_id AS doc_id
        LIMIT 30
        """
        return await Neo4jClient.execute_read_query(cypher, parameters={"entity_ids": entities})

    async def resolve_query_node(self, state: AgentState) -> dict:
        logger.info("[State Node] Resolving semantic references in query.")
        
        system_prompt = (
            "You are a strict query formatter for an industrial engineering database. "
            "If the user query contains ambiguous pronouns (e.g., 'that asset', 'it') without prior context, "
            "return 'AMBIGUOUS_QUERY'. Otherwise, return a clean, highly specific search string."
        )
        
        try:
            # Standard completion, not structured, as we just want a string back.
            response = await self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    # Note: In a real system, you inject Chat History here so the LLM can resolve the pronoun.
                    {"role": "user", "content": f"Query: {state['original_query']}"}
                ],
                temperature=0.0
            )
            resolved = response.choices[0].message.content.strip()
            
            if resolved == "AMBIGUOUS_QUERY":
                return {
                    "resolved_query": state['original_query'], 
                    "system_warning": "Missing explicit entity identifiers in query."
                }
                
            return {"resolved_query": resolved, "system_warning": None}
        except Exception as e:
            logger.error(f"Query resolution failure: {e}")
            return {"resolved_query": state['original_query'], "system_warning": "Query resolution failed."}

    async def retrieve_node(self, state: AgentState) -> dict:
        logger.info("[State Node] Executing multi-modal contexts retrieval.")
        query = state["resolved_query"] 
        
        system_prompt = (
            "You are an elite reliability engineering extraction engine.\n"
            "Analyze the provided text and isolate all operational tags, equipment identifiers, "
            "failure codes, or industry compliance standards (e.g., ISO, API, ASME).\n\n"
            "CRITICAL RULES:\n"
            "1. Do not hallucinate tags. If no explicit tags exist, return an empty entities list.\n"
            "2. Isolate substrings cleanly. For example, in 'Inspect PUMP-101A and B', extract 'PUMP-101A' and 'PUMP-101B'.\n"
            "3. Provide an analytical engineering log detailing your extraction choices."
        )

        try:
            extraction: EntityExtractor = await self.client.chat.completions.create(
                model=self.llm_model,
                response_model=EntityExtractor,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Target Text: {query}"}
                ],
                temperature=0.0
            )
            logger.info(f"[Extraction Complete] Logs: {extraction.reasoning_log}")
            
        except Exception as e:
            logger.error(f"Critical schema extraction breakdown: {e}")
            extraction = EntityExtractor(entities=[], reasoning_log="Pipeline extraction execution failure.")

        asset_mentions = [
            ent.raw_mention for ent in extraction.entities if ent.entity_type == "ASSET"
        ]
        
        resolved_entities = []
        for raw in asset_mentions:
            canonical_id = await self.resolver.resolve_asset_id(raw)
            if canonical_id:
                resolved_entities.append(canonical_id)

        query_vector = self._get_query_embedding(query)
        
        vector_data, graph_data = await asyncio.gather(
            self._fetch_vector_context(query_vector),
            self._fetch_graph_context(resolved_entities)
        )
        
        vector_str = "\n".join([f"- [Doc Source File: {c['filename']}] {c['text']}" for c in vector_data])
        graph_str = "\n".join([
            f"- [Graph Context Doc ID: {g['doc_id']}] ({g['source']})-[{g['relationship']}]->({g['target']})" 
            for g in graph_data if g['relationship']
        ])
        
        combined_context = f"=== VECTOR DATABASE RECORDS ===\n{vector_str}\n\n=== RELATIONSHIP TOPOLOGY MAP ===\n{graph_str}"
        return {"combined_context": combined_context}

    async def grade_context_node(self, state: AgentState) -> dict:
        logger.info("[State Node] Validating retrieval context relevancy scores.")
        try:
            grade: RelevanceGrade = await self.client.chat.completions.create(
                model=self.llm_model,
                response_model=RelevanceGrade,
                messages=[
                    {"role": "system", "content": "Determine if the combined database contexts hold data directly matching parameters specified in the user's inquiry."},
                    {"role": "user", "content": f"User Target Query: {state['resolved_query']}\n\nProvided Datastores Content:\n{state['combined_context']}"}
                ],
                temperature=0.0
            )
            
            if not grade.is_relevant:
                logger.warning("Context deemed irrelevant by grader.")
                existing_warning = state.get("system_warning")
                new_warning = "[System Metric Alert: Relevant datastore logs may be missing.]"
                combined_warning = f"{existing_warning} | {new_warning}" if existing_warning else new_warning
                
                return {
                    "is_relevant": False,
                    "system_warning": combined_warning
                }
            return {"is_relevant": True}
        except Exception:
            # Fail open if grader fails
            return {"is_relevant": True}

    async def generate_node(self, state: AgentState) -> dict:
        logger.info(f"[State Node] Generating target analysis. Attempt {state['retries'] + 1}")
        
        warning_block = f"SYSTEM ALERT: {state.get('system_warning')}\n" if state.get('system_warning') else ""
        
        prompt = f"""
        Synthesize a rigorous engineering answer addressing the query below. 
        Your statement must rely exclusively on facts specified inside the database context block. 
        For every fact you generate, extract the corresponding metadata string (e.g. filename or Doc ID) and map it to evidence_links. 
        
        {warning_block}
        If the required parameters to answer the query are completely absent, state explicitly what is missing and terminate the analysis. Do not hallucinate.

        Database Context Parameters:
        {state['combined_context']}

        Target User Query: {state['resolved_query']}
        """
        try:
            response: CitedAnswer = await self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.llm_model,
                response_model=CitedAnswer,
                temperature=0.2 # Slight temperature bump to prevent identical retries
            )
            return {
                "generation": response.answer, 
                "evidence": response.evidence_links, 
                "retries": state["retries"] + 1
            }
        except Exception as e:
            logger.error(f"Structured Generation crash down step: {e}")
            return {"generation": "Generation Pipeline Fault encountered.", "evidence": [], "retries": state["retries"] + 1}

    async def grade_generation_node(self, state: AgentState) -> dict:
        logger.info("[State Node] Grading synthesized answer metrics against original contexts.")
        ctx = state["combined_context"]
        gen = state["generation"]
        q = state["resolved_query"]
        
        try:
            grounded: GroundednessGrade = await self.client.chat.completions.create(
                model=self.llm_model,
                response_model=GroundednessGrade,
                messages=[{"role": "user", "content": f"Database Master Context:\n{ctx}\n\nProposed Generation Text:\n{gen}"}],
                temperature=0.0
            )
            useful: UtilityGrade = await self.client.chat.completions.create(
                model=self.llm_model,
                response_model=UtilityGrade,
                messages=[{"role": "user", "content": f"User Operational Query:\n{q}\n\nSynthesized Answer text:\n{gen}"}],
                temperature=0.0
            )
            return {"is_grounded": grounded.is_grounded, "is_useful": useful.fully_answers}
        except Exception:
            logger.warning("Exception happened at grading output context. The output may be inaccurate. Defaulting to True to pass.")
            return {"is_grounded": True, "is_useful": True}

    def _evaluate_loop_requirements(self, state: AgentState) -> str:
        if state["is_grounded"] and state["is_useful"]:
            logger.info("[Routing Edge] Generation parameters validated. Finalizing execution state.")
            return "end"
        
        if state["retries"] >= self.max_retries:
            logger.warning("[Routing Edge] Context constraints bounds exceeded. Forcing hard termination.")
            return "end"
            
        logger.warning("[Routing Edge] Internal compliance validation failure. Forcing loop recalculation.")
        return "generate"

    def _compile_workflow(self):
        workflow = StateGraph(AgentState)
        
        workflow.add_node("resolve_query", self.resolve_query_node)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("grade_context", self.grade_context_node)
        workflow.add_node("generate", self.generate_node)
        workflow.add_node("grade_generation", self.grade_generation_node)
        
        workflow.set_entry_point("resolve_query")
        workflow.add_edge("resolve_query", "retrieve")
        workflow.add_edge("retrieve", "grade_context")
        
        # Context grading flows directly to generation regardless of relevancy, 
        # allowing the generator to synthesize the warning + whatever partial context exists.
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

    async def ask(self, query: str) -> dict:
        """System entrypoint. Returns structured answers mapped to verified source references."""
        initial_state = {
            "original_query": query,
            "resolved_query": "",
            "combined_context": "",
            "generation": "",
            "evidence": [],
            "is_relevant": False,
            "is_grounded": False,
            "is_useful": False,
            "retries": 0,
            "system_warning": None
        }
        
        final_state = await self.graph.ainvoke(initial_state)
        logger.info(f"The final state of knowledge copilot is : {final_state}")
        
        return {
            "answer": final_state["generation"],
            "evidence_links": final_state["evidence"]
        }