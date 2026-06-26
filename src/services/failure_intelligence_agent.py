import json
import logging
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from src.db.postgres_client import PostgresPool

logger = logging.getLogger(__name__)

class AnomalyEvent(BaseModel):
    asset_category: str
    current_symptoms: str
    severity: str = "MEDIUM"

class FailureIntelligenceEngine:
    """Phase 5: Proactive Hybrid Search & Pattern Recognition Engine."""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        
        # High reasoning capacity model for complex causal pattern matching
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile", 
            temperature=0.1, 
            max_retries=2
        )
        
        # Dynamically bind the tool so it can access the memory-resident embedding model
        self.tools = [self._build_hybrid_search_tool()]
        
        self.system_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an Industrial Failure Intelligence AI.
            A new anomaly has been detected in the plant. Your job is to prevent a catastrophic failure.
            
            1. Use 'search_historical_incidents' to find similar past failures for this specific equipment category.
            2. Analyze the historical root causes that evolved from these exact symptoms.
            3. Generate a PROACTIVE ALERT containing:
               - **HISTORICAL PRECEDENT**: What happened last time these symptoms occurred?
               - **PREDICTED FAILURE MODE**: What component is likely about to break?
               - **IMMEDIATE CONTAINMENT ACTION**: What should the operators do right now to prevent it?
               
            If no highly relevant history exists, state that this is a novel anomaly profile."""),
            ("human", "New Anomaly Logged -> Category: {asset_category} | Symptoms: {current_symptoms} | Severity: {severity}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        agent = create_tool_calling_agent(self.llm, self.tools, self.system_prompt)
        self.executor = AgentExecutor(
            agent=agent, 
            tools=self.tools, 
            verbose=True, 
            max_iterations=4,
            handle_parsing_errors=True
        )

    def _build_hybrid_search_tool(self):
        """Creates a tool bound to the singleton embedding model for hybrid search."""
        
        @tool("search_historical_incidents")
        def hybrid_search(asset_category: str, symptom_description: str, limit: int = 3) -> str:
            """
            Searches the historical incident database for similar past failures.
            Always use this to cross-reference current symptoms with past disasters.
            """
            # 1. Generate the dense vector embedding for the symptom
            try:
                query_vector = self.embedding_model.encode(symptom_description).tolist()
            except Exception as e:
                logger.error(f"Embedding generation failed: {e}")
                return "System Error: Could not vectorize search query."

            # 2. Execute Pre-filtered HNSW Vector Search in PostgreSQL
            # The WHERE clause filters by category first, then <=> calculates cosine distance
            query = """
                SELECT incident_id, severity, root_cause_category, incident_narrative, 
                       1 - (narrative_embedding <=> %s::vector) AS similarity_score
                FROM historical_incidents
                WHERE asset_category = %s
                ORDER BY narrative_embedding <=> %s::vector
                LIMIT %s;
            """
            try:
                with PostgresPool.get_connection() as conn:
                    with conn.cursor() as cur:
                        # Pass vector twice: once for select scoring, once for ordering
                        cur.execute(query, (query_vector, asset_category, query_vector, limit))
                        rows = cur.fetchall()
                
                if not rows:
                    return f"No similar historical incidents found for {asset_category}."

                payload = [
                    {
                        "incident_id": r[0],
                        "severity": r[1],
                        "root_cause": r[2],
                        "narrative": r[3],
                        "relevance_score": round(float(r[4]), 3)
                    }
                    for r in rows if float(r[4]) > 0.65  # Hard threshold to prevent hallucinated matches
                ]
                
                return json.dumps(payload) if payload else "No incidents crossed the relevance threshold."
                
            except Exception as e:
                logger.error(f"Hybrid search database failure: {e}")
                return f"Database Error: {str(e)}"
                
        return hybrid_search

    def evaluate_anomaly(self, event: AnomalyEvent) -> Dict[str, Any]:
        """Orchestrates the predictive analysis loop."""
        logger.info(f"Triggering Failure Intelligence for {event.asset_category}")
        try:
            result = self.executor.invoke({
                "asset_category": event.asset_category,
                "current_symptoms": event.current_symptoms,
                "severity": event.severity
            })
            return {"status": "SUCCESS", "alert": result["output"]}
        except Exception as e:
            logger.critical(f"Intelligence Engine Panicked: {e}")
            return {"status": "FATAL", "alert": f"Execution failure: {str(e)}"}