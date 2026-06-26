import logging
from fastapi import APIRouter, HTTPException, Depends, Request
from src.core.schemas import QueryRequest, QueryResponse
from src.services.knowledge_copilot import KnowledgeCopilot

logger = logging.getLogger(__name__)

router = APIRouter()

def get_copilot_engine(request: Request) -> KnowledgeCopilot:
    """
    Dependency injection function. 
    Safely extracts the singleton Copilot instance from the FastAPI application state.
    """
    if not hasattr(request.app.state, "copilot") or request.app.state.copilot is None:
        logger.error("KnowledgeCopilot not found in application state.")
        raise HTTPException(
            status_code=503, 
            detail="The Knowledge Copilot engine is currently offline or failed to initialize."
        )
    return request.app.state.copilot

@router.post("/query", response_model=QueryResponse)
def query_knowledge(
    payload: QueryRequest, 
    copilot: KnowledgeCopilot = Depends(get_copilot_engine)
):
    """
    Executes the deterministic GraphRAG / Self-RAG loop.
    Defined synchronously due to underlying blocking database drivers.
    """
    logger.info(f"API Request received for query: {payload.query_text}")

    try:
        # The 'copilot' object here is the exact instance loaded in main.py
        response_payload = copilot.ask(payload.query_text)
        return response_payload
        
    except Exception as e:
        logger.error(f"Copilot execution failed at router boundary: {e}")
        # Mask internal stack traces from the client while returning a standard 500
        raise HTTPException(
            status_code=500, 
            detail="The Knowledge Copilot encountered a system error during execution."
        )