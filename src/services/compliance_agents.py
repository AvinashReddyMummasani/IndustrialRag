import json
import logging
import asyncio
from typing import Dict, Any

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from src.services.compliance_tools import fetch_regulatory_clauses, fetch_recent_inspections
from src.services.entity_resolver import EntityResolver
from src.core.schemas import AuditRequest, AuditReportSchema

logger = logging.getLogger(__name__)

class RegulatoryComplianceEngine:
    """Engine for compliance auditing."""
    
    def __init__(self,llm=None):
        self.resolver = EntityResolver()
        
        self.llm = ChatGroq(
            model=llm, 
            temperature=0.0,
            max_retries=3
        )
        
        self.structured_llm = self.llm.with_structured_output(AuditReportSchema)
        
        self.system_prompt = """You are a Lead Quality & Regulatory Auditor for heavy industrial operations.
Your task is to conduct a zero-hallucination compliance gap analysis.

You will be provided with:
1. Legal Thresholds (extracted directly from the regulatory graph).
2. Operational Ground-Truth (recent physical inspection telemetry).

Your mandate:
Cross-reference the operational parameters against the legal thresholds. 
Extract the final findings and map them strictly into the requested JSON schema. Do not invent data."""

    async def execute_audit(self, config: AuditRequest) -> Dict[str, Any]:
        """
        Executes parallel database pre-fetching and a single-shot LLM synthesis.
        """
        logger.info(f"Initiating fast-path compliance audit against {config.target_standard} for {config.asset_id}")
        
        # 1. RESOLVE ENTITY (I/O Bound)
        canonical_asset_id = await self.resolver.resolve_asset_id(config.asset_id)
        
        if not canonical_asset_id:
            logger.warning(f"Audit aborted. Unmapped alias: '{config.asset_id}'")
            return {
                "status": "FAILED", 
                "report": {"error": f"Audit aborted. Asset alias '{config.asset_id}' cannot be verified in the master registry."}
            }

        try:
            # 2. PARALLEL DATA PRE-FETCHING (I/O Bound)

            reg_task = fetch_regulatory_clauses.ainvoke({
                "standard_name": config.target_standard, 
                "equipment_type": config.asset_type
            })
            
            insp_task = fetch_recent_inspections.ainvoke({
                "asset_id": canonical_asset_id
            })
            
            # Block the event loop here ONLY until both databases return their payloads
            reg_data, insp_data = await asyncio.gather(reg_task, insp_task)

            # 3. CONTEXT ASSEMBLY
            compiled_evidence = (
                f"--- TARGET ASSET: {canonical_asset_id} ({config.asset_type}) ---\n\n"
                f"=== LEGAL & REGULATORY THRESHOLDS ===\n"
                f"{reg_data}\n\n"
                f"=== RECENT OPERATIONAL INSPECTIONS ===\n"
                f"{insp_data}"
            )

            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=compiled_evidence)
            ]

            # 4. SINGLE-SHOT LLM SYNTHESIS
            logger.info(f"Database payloads compiled for {canonical_asset_id}. Executing schema synthesis...")
            structured_payload: AuditReportSchema = await self.structured_llm.ainvoke(messages)
            
            return {"status": "SUCCESS", "report": structured_payload.model_dump()}

        except Exception as e:
            logger.critical(f"Compliance Engine Panicked: {e}", exc_info=True)
            return {
                "status": "FATAL", 
                "report": {"error": f"Audit execution failure: {str(e)}"}
            }