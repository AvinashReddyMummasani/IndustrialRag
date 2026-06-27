import json
import logging
import asyncio
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.errors import GraphRecursionError

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
    """Learns from past patterns"""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        
        # Model 1: The Reasoner (Allowed slight temperature for pattern matching)
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile", 
            temperature=0.1, 
            max_retries=3
        )
        
        # Model 2: The Synthesizer (Zero temperature, forced Pydantic schema)
        self.structured_llm = self.llm.with_structured_output(ProactiveAlertSchema)
        
        # Bind the isolated async tool closure
        self.tools = [self._build_hybrid_search_tool()]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        self.system_prompt = """You are an Industrial Failure Intelligence AI.
A new anomaly has been detected in the plant. Your job is to prevent a catastrophic failure.

Mandatory Execution Flow:
1. Use 'search_historical_incidents' to find similar past failures.
2. Analyze the historical root causes that evolved from these exact symptoms.
3. Compare the semantic similarity scores to determine threat validity.

Focus purely on analytical extraction and database querying. Do not attempt to format the final alert layout."""

        # Compile the deterministic DAG
        self.graph = self._build_graph()

    def _build_hybrid_search_tool(self) -> StructuredTool:
        """Encapsulates blocking dependencies in a thread-safe asynchronous tool."""
        
        def _sync_search_logic(asset_category: str, symptom_description: str, limit: int) -> str:
            """Synchronous execution core for CPU and I/O blocking tasks."""
            # 1. CPU-Bound Embedding Generation
            try:
                query_vector = self.embedding_model.encode(symptom_description).tolist()
            except Exception as e:
                logger.error(f"Embedding generation failed: {e}")
                return "System Error: Could not vectorize search query."

            # 2. I/O-Bound Database Execution
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
                    for r in rows if float(r[4]) > 0.65  # Noise rejection threshold
                ]
                
                return json.dumps(payload) if payload else "No incidents crossed the 0.65 relevance threshold."
                
            except Exception as e:
                logger.error(f"Hybrid search DB failure: {e}")
                return f"Database Error: {str(e)}"

        async def _async_wrapper(asset_category: str, symptom_description: str, limit: int = 3) -> str:
            """Pushes the synchronous core to the asyncio thread pool."""
            return await asyncio.to_thread(_sync_search_logic, asset_category, symptom_description, limit)

        return StructuredTool.from_function(
            coroutine=_async_wrapper,
            name="search_historical_incidents",
            description="Searches the incident database for past failures using semantic vector similarity."
        )

    def _build_graph(self):
        """Constructs the LangGraph state machine."""
        workflow = StateGraph(MessagesState)
        
        # NODE 1: Autonomous Reasoning
        async def call_model(state: MessagesState):
            messages = state["messages"]
            if not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=self.system_prompt)] + messages
            
            response = await self.llm_with_tools.ainvoke(messages)
            return {"messages": [response]}

        # NODE 2: Strict Schema Extraction
        async def format_final_alert(state: MessagesState):
            messages = state["messages"][-1]
            extraction_prompt = HumanMessage(
                content="Review the threat analysis and extract the final predictive intelligence into the required JSON schema."
            )
            
            structured_payload: ProactiveAlertSchema = await self.structured_llm.ainvoke(messages + [extraction_prompt])
            return {"messages": [AIMessage(content=structured_payload.model_dump_json())]}

        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("format_output", format_final_alert)

        # Edges
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", tools_condition, {"tools": "tools", "__end__": "format_output"})
        workflow.add_edge("tools", "agent")
        workflow.add_edge("format_output", END)

        return workflow.compile()

    async def evaluate_anomaly(self, event: AnomalyEvent) -> Dict[str, Any]:
        """Async entry point for webhook and API handlers."""
        logger.info(f"Triggering Failure Intelligence for {event.asset_category}")
        
        initial_payload = (
            f"New Anomaly Logged -> Category: {event.asset_category} | "
            f"Symptoms: {event.current_symptoms} | Severity: {event.severity}"
        )

        try:
            # Hard limit protects against hallucination-induced infinite loops
            run_config = {"recursion_limit": 5}
            
            output_state = await self.graph.ainvoke(
                {"messages": [HumanMessage(content=initial_payload)]},
                config=run_config
            )
            
            final_json = output_state["messages"][-1].content
            return {"status": "SUCCESS", "alert": final_json}
            
        except GraphRecursionError:
            logger.error(f"Intelligence graph recursion limit hit for {event.asset_category}.")
            return {
                "status": "PARTIAL_SUCCESS", 
                "alert": '{"error": "Agent iteration limits breached. Review telemetry manually."}'
            }
        except Exception as e:
            logger.critical(f"Intelligence Engine Panicked: {e}", exc_info=True)
            return {
                "status": "FATAL", 
                "alert": f'{{"error": "Execution failure: {str(e)}"}}'
            }