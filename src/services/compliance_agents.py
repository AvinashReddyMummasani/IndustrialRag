import logging
from typing import Dict, Any

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.errors import GraphRecursionError

from src.services.compliance_tools import fetch_regulatory_clauses, fetch_recent_inspections
from src.services.entity_resolver import EntityResolver
from src.core.schemas import AuditRequest, AuditReportSchema

logger = logging.getLogger(__name__)

class RegulatoryComplianceEngine:
    """Asynchronous, deterministic auditor mapping regulatory text to state using LangGraph."""
    
    def __init__(self):
        self.resolver = EntityResolver()
        
        # 1. Base LLM for reasoning and tool calling
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile", 
            temperature=0.0,
            max_retries=3
        )
        
        # 2. Schema-enforced LLM for the final extraction node
        self.structured_llm = self.llm.with_structured_output(AuditReportSchema)
        
        self.tools = [fetch_regulatory_clauses, fetch_recent_inspections]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        self.system_prompt = """You are a Lead Quality & Regulatory Auditor for heavy industrial operations.
Your task is to conduct a zero-hallucination compliance gap analysis.

Execution Flow:
1. Use 'fetch_regulatory_clauses' to define the exact legal thresholds for the asset type.
2. Use 'fetch_recent_inspections' to pull ground-truth parameters.
3. Cross-reference the operational parameters against the legal thresholds.

Analyze the data step-by-step. Do not attempt to format the final output yet; focus purely on data gathering and logical comparison."""

        # Pre-compile the execution graph
        self.graph = self._build_graph()

    def _build_graph(self):
        """Constructs the deterministic execution graph for the auditor."""
        workflow = StateGraph(MessagesState)
        
        # NODE 1: The Reasoning Loop
        async def call_model(state: MessagesState):
            messages = state["messages"]
            if not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=self.system_prompt)] + messages
            
            response = await self.llm_with_tools.ainvoke(messages)
            return {"messages": [response]}

        # NODE 2: The Formatter (Synthesis)
        async def format_final_report(state: MessagesState):
            messages = state["messages"]
            extraction_prompt = HumanMessage(
                content="Review the preceding audit analysis. Extract the final findings and map them strictly into the requested JSON schema."
            )
            
            # Note: For long tool histories to prevent token limits
            hist = messages
            if len(messages) > 10:
                hist = messages[-10:]

            structured_payload: AuditReportSchema = await self.structured_llm.ainvoke(hist + [extraction_prompt])
            
            # Convert Pydantic object back into an AIMessage string payload for state compatibility
            return {"messages": [AIMessage(content=structured_payload.model_dump_json())]}

        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("format_output", format_final_report)

        # EDGE ROUTING
        workflow.add_edge(START, "agent")
        
        # Intercept the end of the tool loop. 
        # If tools are called -> route to 'tools'. If agent is done -> route to 'format_output'.
        workflow.add_conditional_edges(
            "agent", 
            tools_condition,
            {"tools": "tools", "__end__": "format_output"}
        )
        
        workflow.add_edge("tools", "agent")
        workflow.add_edge("format_output", END)

        return workflow.compile()

    async def execute_audit(self, config: AuditRequest) -> Dict[str, Any]:
        """Validates canonical IDs and orchestrates the asynchronous audit loop."""
        logger.info(f"Initiating compliance audit against {config.target_standard} for {config.asset_id}")
        
        canonical_asset_id = await self.resolver.resolve_asset_id(config.asset_id)
        
        if not canonical_asset_id:
            logger.warning(f"Audit aborted. Unmapped alias: '{config.asset_id}'")
            return {
                "status": "FAILED", 
                "report": f"Audit aborted. Asset alias '{config.asset_id}' cannot be verified in the master registry."
            }

        initial_payload = (
            f"Conduct gap analysis for Asset: {canonical_asset_id} | "
            f"Type: {config.asset_type} | Standard: {config.target_standard}"
        )

        try:
            # Allows for 5 tool-calling cycles max
            run_config = {"recursion_limit": 5}
            
            output_state = await self.graph.ainvoke(
                {"messages": [HumanMessage(content=initial_payload)]},
                config=run_config
            )
            
            # The final output is now guaranteed to be the serialized JSON string from the Pydantic model
            final_report_json = output_state["messages"][-1].content
            return {"status": "SUCCESS", "report": final_report_json}
            
        except GraphRecursionError:
            logger.error(f"Audit graph hit recursion limits on {canonical_asset_id}.")
            return {
                "status": "PARTIAL_SUCCESS", 
                "report": "Agent hit safety iteration limits before schema extraction could complete. Audit mapping incomplete."
            }
        except Exception as e:
            logger.critical(f"Compliance Engine Panicked: {e}", exc_info=True)
            return {"status": "FATAL", "report": f"Audit execution failure: {str(e)}"}