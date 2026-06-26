import logging
from typing import Dict, Any
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from src.services.sql_tools import get_historical_asset_incidents, extract_knowledge_graph_oem_procedures
from src.services.entity_resolver import EntityResolver

logger = logging.getLogger(__name__)

class RCADiagnosticInput(BaseModel):
    asset_id: str  
    asset_type: str
    symptom: str

class IndustrialRCAEngine:
    """Orchestrates ReAct agent workflows for Phase 3 Root Cause Analysis using Groq LPUs."""
    
    def __init__(self):
        # Initialize the resolution layer for Postgres primary key mapping
        self.resolver = EntityResolver()
        
        # Enforce deterministic analysis via temperature=0.0
        # utilizing Llama 3.3 70B which natively supports tool calling
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile", 
            temperature=0.0,
            max_retries=2
        )
        
        # Bind the isolated deterministic tools
        self.tools = [get_historical_asset_incidents, extract_knowledge_graph_oem_procedures]
        
        self.system_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Principal Reliability and Systems Safety Engineer in heavy industry. 
            Your role is to compile a granular Root Cause Analysis (RCA) profile for operational anomalies.
            
            Mandatory Diagnostic Directives:
            1. Pull recent relational failures using 'get_historical_asset_incidents' to extract localized operational context.
            2. Pull structural schematic rules and resolution logic via 'extract_knowledge_graph_oem_procedures'.
            3. Synthesize the overlapping intersections of explicit relational values and semantic knowledge networks.
            
            Structure your output into these explicit components:
            - **EXECUTIVE ANOMALY DISCOVERY**: High-level failure assessment.
            - **CHRONIC PATTERN DETERMINATION**: Correlation between historical issues and current behavior.
            - **OEM STRATIFICATION ALIGNMENT**: Guidance extracted from manuals.
            - **ACTIONABLE REMEDIATION ROUTE**: Explicit recovery path.
            
            If historical data is missing, explicitly state the lack of operational baseline. Do not invent failure codes."""),
            ("human", "Execute deep diagnostic loop for Asset Target: {canonical_asset_id} | Class: {asset_type} | Observed Symptom: {symptom}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        # Langchain automatically maps the tools to the Groq payload specification
        agent_layer = create_tool_calling_agent(self.llm, self.tools, self.system_prompt)
        
        self.executor = AgentExecutor(
            agent=agent_layer, 
            tools=self.tools, 
            verbose=True, 
            max_iterations=6,
            handle_parsing_errors=True
        )

    def execute_rca_evaluation(self, config: RCADiagnosticInput) -> Dict[str, Any]:
        """
        Intercepts fuzzy inputs, resolves canonical hardware IDs, and executes Groq-powered tool loops.
        """
        logger.info(f"Attempting to resolve fuzzy alias: '{config.asset_id}'")
        canonical_asset_id = self.resolver.resolve_asset_id(config.asset_id)
        
        if not canonical_asset_id:
            logger.warning(f"RCA execution aborted. Asset alias '{config.asset_id}' unmapped.")
            return {
                "status": "FAILED", 
                "report": (
                    f"Entity Resolution Failure: The system could not map the provided asset alias "
                    f"'{config.asset_id}' to a known operational canonical ID in the primary registry. "
                    f"Please verify the equipment tag."
                )
            }
            
        logger.info(f"Alias resolved. Executing RCA payload against canonical DB token: {canonical_asset_id}")

        try:
            runtime_payload = self.executor.invoke({
                "canonical_asset_id": canonical_asset_id,
                "asset_type": config.asset_type,
                "symptom": config.symptom
            })
            return {"status": "SUCCESS", "report": runtime_payload["output"]}
            
        except Exception as e:
            logger.critical(f"RCA System Execution Panicked: {e}")
            return {
                "status": "FATAL", 
                "report": f"System execution terminated unexpectedly during cross-functional analytical trace. Context: {str(e)}"
            }