import logging
from typing import Dict, Any
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from src.services.compliance_tools import fetch_regulatory_clauses, fetch_recent_inspections
from src.services.entity_resolver import EntityResolver

logger = logging.getLogger(__name__)

class AuditRequest(BaseModel):
    asset_id: str  
    asset_type: str
    target_standard: str # e.g., "OISD-144"

class RegulatoryComplianceEngine:
    """Autonomous auditor mapping regulatory text to relational equipment states."""
    
    def __init__(self):
        self.resolver = EntityResolver()
        
        # Strict low-temperature inference for legal/compliance mapping
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile", 
            temperature=0.0,
            max_retries=2
        )
        
        self.tools = [fetch_regulatory_clauses, fetch_recent_inspections]
        
        self.system_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Lead Quality & Regulatory Auditor for heavy industrial operations.
            Your task is to conduct a zero-hallucination compliance gap analysis.
            
            Execution Flow:
            1. Use 'fetch_regulatory_clauses' to define the exact legal thresholds for the asset type under the requested standard.
            2. Use 'fetch_recent_inspections' to pull the ground-truth operational parameters and latest deviations.
            3. Cross-reference the operational parameters against the legal thresholds.
            
            Output Structure:
            - **REGULATORY BASELINE**: State the exact clauses and limits found.
            - **OPERATIONAL STATE**: Summarize the latest inspection findings.
            - **COMPLIANCE GAPS**: Explicitly list any deviations where the operational state violates the regulatory baseline. If none, state "FULLY COMPLIANT".
            - **AUDIT EVIDENCE**: Specify which database records prove this status."""),
            ("human", "Conduct gap analysis for Asset: {canonical_asset_id} | Type: {asset_type} | Standard: {target_standard}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        agent = create_tool_calling_agent(self.llm, self.tools, self.system_prompt)
        self.executor = AgentExecutor(
            agent=agent, 
            tools=self.tools, 
            verbose=True, 
            max_iterations=5,
            handle_parsing_errors=True
        )

    def execute_audit(self, config: AuditRequest) -> Dict[str, Any]:
        """Validates canonical IDs and orchestrates the audit ReAct loop."""
        logger.info(f"Initiating compliance audit against {config.target_standard} for {config.asset_id}")
        
        canonical_asset_id = self.resolver.resolve_asset_id(config.asset_id)
        
        if not canonical_asset_id:
            return {
                "status": "FAILED", 
                "report": f"Audit aborted. Asset alias '{config.asset_id}' cannot be verified in the master registry."
            }

        try:
            result = self.executor.invoke({
                "canonical_asset_id": canonical_asset_id,
                "asset_type": config.asset_type,
                "target_standard": config.target_standard
            })
            return {"status": "SUCCESS", "report": result["output"]}
        except Exception as e:
            logger.critical(f"Compliance Engine Panicked: {e}")
            return {"status": "FATAL", "report": f"Audit execution failure: {str(e)}"}