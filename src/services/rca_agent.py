import logging
import asyncio
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from src.services.sql_tools import fetch_postgres_incidents_async, fetch_neo4j_procedures_async
from src.services.entity_resolver import EntityResolver

logger = logging.getLogger(__name__)


class RCADiagnosticInput(BaseModel):
    asset_id: str = Field(..., description="Fuzzy or explicit asset tag from the field.")
    asset_type: str = Field(..., description="Class of the asset (e.g., Centrifugal Pump, Conveyor).")
    symptom: str = Field(..., description="Primary reported anomaly.")

class RCAReportSchema(BaseModel):
    executive_anomaly_discovery: str = Field(description="High-level diagnostic assessment of the failure.")
    chronic_pattern_determination: str = Field(description="Explicit correlation between historical incidents and current symptom.")
    oem_stratification_alignment: str = Field(description="Specific technical guidance extracted from OEM manuals.")
    actionable_remediation_route: List[str] = Field(description="Step-by-step array of instructions for field technicians.")
    missing_data_flags: List[str] = Field(description="List of unavailable operational baselines or missing history. Empty if complete.")
    system_confidence_score: float = Field(description="Float between 0.0 and 1.0 indicating diagnostic confidence based on available data.")
    evidence_citations: List[str] = Field(description="List of specific work_order_ids and OEM source_file names used to make this determination.")


class IndustrialRCAEngine:
    """
    Highly concurrent, deterministic RAG orchestrator for Root Cause Analysis.
    Optimized for strict schema adherence and minimized Time-To-First-Token (TTFT).
    """
    
    def __init__(self,embedding_model):
        self.resolver = EntityResolver()
        
        # Bind the LLM directly to the JSON schema. No tool-calling loops required.
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile", 
            temperature=0.0, 
            max_retries=3
        )
        self.structured_llm = self.llm.with_structured_output(RCAReportSchema)
        
        self.system_prompt = """You are a Principal Reliability and Systems Safety Engineer in heavy industry. 
            Your role is to compile a granular Root Cause Analysis (RCA) profile for operational anomalies.

            Mandatory Diagnostic Directives:
            1. Review the provided Historical Incident Data for localized operational context.
            2. Review the OEM Procedures for structural schematic rules and resolution logic.
            3. Synthesize the overlapping intersections of explicit relational values and semantic knowledge networks.

            Analyze the data step-by-step. Focus purely on accurate data extraction and correlation. 
            If historical data is missing, explicitly note it in your reasoning. Do not invent failure codes.
            Strictly adhere to the provided JSON schema constraints."""
        
    async def _generate_vector_async(self, text: str) -> list[float]:
            """
            Executes CPU-bound tensor inference on a background thread to prevent 
            event loop blocking. Returns a standard Python list for Neo4j compatibility.
            """

            def _sync_encode():
                return self.embedding_model.encode(text).tolist()

            return await asyncio.to_thread(_sync_encode)




    async def execute_rca_evaluation(self, config: RCADiagnosticInput) -> Dict[str, Any]:
        """
        Intercepts fuzzy inputs, resolves canonical hardware IDs, fetches data concurrently 
        via native async drivers, and synthesizes the final RCA report in a single pass.
        """
        logger.info(f"Attempting to resolve fuzzy alias: '{config.asset_id}'")
        canonical_asset_id = await self.resolver.resolve_asset_id(config.asset_id)
        
        if not canonical_asset_id:
            logger.warning(f"RCA execution aborted. Asset alias '{config.asset_id}' unmapped.")
            return {
                "status": "FAILED", 
                "report": {
                    "error": f"Entity Resolution Failure: System could not map alias '{config.asset_id}' to a canonical ID."
                }
            }
        
        logger.info("Generating semantic embedding for symptom...")
        symptom_vector = await self._generate_vector_async(config.symptom)
            
        logger.info(f"Alias resolved: {canonical_asset_id}. Executing concurrent I/O fetches.")

        try:
            # Execute database I/O concurrently on the async event loop
            historical_data, oem_data = await asyncio.gather(
                fetch_postgres_incidents_async(canonical_asset_id, scan_limit=5),
                fetch_neo4j_procedures_async(config.asset_type, symptom_vector)
            )

            # Construct the deterministic prompt payload
            synthesis_payload = (
                f"Asset Target: {canonical_asset_id}\n"
                f"Class: {config.asset_type}\n"
                f"Observed Symptom: {config.symptom}\n\n"
                f"--- HISTORICAL INCIDENT DATA ---\n{historical_data}\n\n"
                f"--- OEM PROCEDURES ---\n{oem_data}\n\n"
                "Extract the final findings and map them strictly into the requested JSON schema."
            )

            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=synthesis_payload)
            ]
            
            logger.info(f"Context assembled for {canonical_asset_id}. Invoking structured LLM synthesis.")
            
            structured_payload: RCAReportSchema = await self.structured_llm.ainvoke(messages)

            return {"status": "SUCCESS", "report": structured_payload.model_dump()}
            
        except Exception as e:
            logger.critical(f"RCA System Execution Panicked: {e}", exc_info=True)
            return {
                "status": "FATAL", 
                "report": {"error": f"System execution terminated unexpectedly. Context: {str(e)}"}
            }