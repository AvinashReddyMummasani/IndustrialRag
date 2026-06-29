import json
import logging
import asyncio
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from src.db.postgres_client import PostgresPool

logger = logging.getLogger(__name__)


class AnomalyEvent(BaseModel):
    asset_category: str = Field(..., description="The category of the equipment (e.g., 'Centrifugal Pump').")
    current_symptoms: str = Field(..., description="The observed telemetry or physical anomaly.")
    severity: str = Field(default="MEDIUM", description="System-assigned severity level.")

class ProactiveAlertSchema(BaseModel):
    historical_precedent: str = Field(description="Summary of past incidents that matched these symptoms. State if novel.")
    predicted_failure_mode: str = Field(description="The specific component or system likely to fail next.")
    immediate_containment_action: List[str] = Field(description="Step-by-step actions operators must take immediately.")
    confidence_score: float = Field(description="Diagnostic confidence between 0.0 and 1.0 based on historical match quality.")
    referenced_incident_ids: List[str] = Field(description="List of 'incident_id' primary keys used to make this prediction.")



class FailureIntelligenceEngine:
    """
     Based on current symtoms and previous patterns it suggest what to do.
    """
    
    def __init__(self, embedding_model,llm : str="llama-3.3-70b-versatile"):
        self.embedding_model = embedding_model
        
        self.llm = ChatGroq(
            model= llm, 
            temperature=0.1, 
            max_retries=3
        )
        self.structured_llm = self.llm.with_structured_output(ProactiveAlertSchema)
        
        self.system_prompt = """You are an Industrial Failure Intelligence AI.
A new anomaly has been detected in the plant. Your job is to prevent a catastrophic failure.

Mandatory Execution Flow:
1. Analyze the provided Historical Incident Data that matched the current symptoms.
2. Determine the historical root causes that evolved from these exact symptoms.
3. Compare the semantic similarity scores to determine threat validity.

Focus purely on analytical extraction. Output strictly to the requested JSON schema."""

    async def _generate_embedding_async(self, text: str) -> list[float]:
        """Safely offloads CPU-bound tensor operations to a background thread."""
        def _sync_encode():
            return self.embedding_model.encode(text).tolist()
        return await asyncio.to_thread(_sync_encode)

    async def _fetch_similar_incidents(self, asset_category: str, query_vector: list[float], limit: int = 3) -> str:
        """Executes native async pgvector similarity search."""
        query = """
            SELECT incident_id, severity, root_cause_category, incident_narrative, 
                   1 - (narrative_embedding <=> $1::vector) AS similarity_score
            FROM historical_incidents
            WHERE asset_category = $2
            ORDER BY narrative_embedding <=> $1::vector
            LIMIT $3;
        """
        try:
            async with PostgresPool.get_connection() as conn:
                vector_str = str(query_vector)
                rows = await conn.fetch(query, vector_str, asset_category, limit)
            
            if not rows:
                return f"No historical incidents found for {asset_category}."

            payload = [
                {
                    "incident_id": r['incident_id'],
                    "severity": r['severity'],
                    "root_cause": r['root_cause_category'],
                    "narrative": r['incident_narrative'],
                    "relevance_score": round(float(r['similarity_score']), 3)
                }
                for r in rows if float(r['similarity_score']) > 0.65 
            ]
            
            return json.dumps(payload) if payload else "No incidents crossed the 0.65 relevance threshold."
            
        except Exception as e:
            logger.error(f"Hybrid search DB failure for {asset_category}: {e}", exc_info=True)
            return f"Database Error: {str(e)}"

    async def evaluate_anomaly(self, event: AnomalyEvent) -> Dict[str, Any]:
        """Async entry point for webhook and API handlers."""
        logger.info(f"Triggering Failure Intelligence for {event.asset_category}")
        
        try:
            # 1. Generate vector representation of the symptom (CPU-bound, threaded)
            query_vector = await self._generate_embedding_async(event.current_symptoms)
            
            # 2. Fetch highly correlated historical incidents (I/O-bound, async)
            historical_context = await self._fetch_similar_incidents(
                event.asset_category, 
                query_vector, 
                limit=3
            )
            
            # 3. Construct deterministic prompt payload
            synthesis_payload = (
                f"New Anomaly Logged:\n"
                f"Category: {event.asset_category}\n"
                f"Symptoms: {event.current_symptoms}\n"
                f"Severity: {event.severity}\n\n"
                f"--- HISTORICAL MATCHES (Similarity > 0.65) ---\n"
                f"{historical_context}\n\n"
                f"Extract the predictive intelligence strictly into the required JSON schema."
            )

            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=synthesis_payload)
            ]
            
            # 4. Single pass LLM synthesis
            structured_payload: ProactiveAlertSchema = await self.structured_llm.ainvoke(messages)
            
            return {"status": "SUCCESS", "alert": structured_payload.model_dump()}
            
        except Exception as e:
            logger.critical(f"Intelligence Engine Panicked: {e}", exc_info=True)
            return {
                "status": "FATAL", 
                "alert": {"error": f"Execution failure: {str(e)}"}
            }