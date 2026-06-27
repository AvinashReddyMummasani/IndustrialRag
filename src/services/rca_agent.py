import logging
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.errors import GraphRecursionError

from src.services.sql_tools import get_historical_asset_incidents, extract_knowledge_graph_oem_procedures
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
    """Asynchronous LangGraph orchestrator for deterministic, structured Root Cause Analysis."""
    
    def __init__(self):
        self.resolver = EntityResolver()
        
        # Model 1: The Reasoning Agent (Bound to Tools)
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile", 
            temperature=0.0,
            max_retries=3
        )
        
        # Model 2: The Extraction Synthesizer (Bound to Schema)
        self.structured_llm = self.llm.with_structured_output(RCAReportSchema)
        
        self.tools = [get_historical_asset_incidents, extract_knowledge_graph_oem_procedures]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # The prompt is stripped of Markdown formatting requirements.
        # The LLM's only job here is to think and execute SQL/Graph queries.
        self.system_prompt = """You are a Principal Reliability and Systems Safety Engineer in heavy industry. 
Your role is to compile a granular Root Cause Analysis (RCA) profile for operational anomalies.

Mandatory Diagnostic Directives:
1. Pull recent relational failures using 'get_historical_asset_incidents' to extract localized operational context.
2. Pull structural schematic rules and resolution logic via 'extract_knowledge_graph_oem_procedures'.
3. Synthesize the overlapping intersections of explicit relational values and semantic knowledge networks.

Analyze the data step-by-step. Focus purely on accurate data extraction and correlation. 
If historical data is missing, explicitly note it in your reasoning. Do not invent failure codes."""

        self.graph = self._build_graph()

    def _build_graph(self):
        """Constructs the deterministic execution DAG."""
        workflow = StateGraph(MessagesState)
        
        # NODE 1: Autonomous Reasoning & Tool Calling
        async def call_model(state: MessagesState):
            messages = state["messages"]
            if not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=self.system_prompt)] + messages
            
            response = await self.llm_with_tools.ainvoke(messages)
            return {"messages": [response]}

        # NODE 2: JSON Extraction & Formatting
        async def format_final_report(state: MessagesState):
            messages = state["messages"]
            extraction_prompt = HumanMessage(
                content="Review the preceding diagnostic analysis. Extract the final findings and map them strictly into the requested JSON schema."
            )
            
            structured_payload: RCAReportSchema = await self.structured_llm.ainvoke(messages + [extraction_prompt])
            
            return {"messages": [AIMessage(content=structured_payload.model_dump_json())]}

        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("format_output", format_final_report)

        # EDGE ROUTING
        workflow.add_edge(START, "agent")
        
        workflow.add_conditional_edges(
            "agent", 
            tools_condition,
            {"tools": "tools", "__end__": "format_output"}
        )
        
        workflow.add_edge("tools", "agent")
        workflow.add_edge("format_output", END)

        return workflow.compile()

    async def execute_rca_evaluation(self, config: RCADiagnosticInput) -> Dict[str, Any]:
        """
        Intercepts fuzzy inputs, resolves canonical hardware IDs, and executes LangGraph loops.
        """
        logger.info(f"Attempting to resolve fuzzy alias: '{config.asset_id}'")
        canonical_asset_id = await self.resolver.resolve_asset_id(config.asset_id)
        
        if not canonical_asset_id:
            logger.warning(f"RCA execution aborted. Asset alias '{config.asset_id}' unmapped.")
            return {
                "status": "FAILED", 
                "report": (
                    f"Entity Resolution Failure: The system could not map the provided asset alias "
                    f"'{config.asset_id}' to a known operational canonical ID in the primary registry."
                )
            }
            
        logger.info(f"Alias resolved. Executing RCA payload against canonical DB token: {canonical_asset_id}")

        initial_payload = (
            f"Execute deep diagnostic loop for Asset Target: {canonical_asset_id} | "
            f"Class: {config.asset_type} | Observed Symptom: {config.symptom}"
        )

        try:
            # Prevents infinite loops if the LLM gets trapped repeating queries
            run_config = {"recursion_limit": 6}
            
            output_state = await self.graph.ainvoke(
                {"messages": [HumanMessage(content=initial_payload)]},
                config=run_config
            )
            
            # The payload is now guaranteed to be a JSON string mapping to RCAReportSchema
            final_report_json = output_state["messages"][-1].content
            return {"status": "SUCCESS", "report": final_report_json}
            
        except GraphRecursionError:
            logger.error("RCA Graph hit iteration limit before finalizing the report.")
            return {
                "status": "PARTIAL_SUCCESS", 
                "report": '{"error": "Agent hit safety iteration limits. Synthesis incomplete. Review manual logs."}'
            }
        except Exception as e:
            logger.critical(f"RCA System Execution Panicked: {e}", exc_info=True)
            return {
                "status": "FATAL", 
                "report": f'{{"error": "System execution terminated unexpectedly. Context: {str(e)}"}}'
            }