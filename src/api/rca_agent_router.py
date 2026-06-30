from fastapi import APIRouter, HTTPException, Depends, Request

from src.services.rca_agent import IndustrialRCAEngine, RCADiagnosticInput
from src.core.schemas import APIWebResponse

router = APIRouter(prefix="/api/v1/industrial-intelligence", tags=["Reliability Intelligence"])





def resolve_rca_engine(request: Request) -> IndustrialRCAEngine:
    if not hasattr(request.app.state, "rca_engine") or request.app.state.rca_engine is None:
        raise HTTPException(status_code=503, detail="RCA Engine is not initialized.")
    return request.app.state.rca_engine

@router.post("/root-cause-analysis", response_model=APIWebResponse)
async def compute_root_cause_matrix(
    payload: RCADiagnosticInput,
    engine: IndustrialRCAEngine = Depends(resolve_rca_engine)
):
    """
    Ingests live engineering system failure tokens, fetches cross-sectional records 
    from PostgreSQL and Neo4j, and returns a verified engineering safety directive.
    """
    execution_result = await engine.execute_rca_evaluation(payload)
    
    if execution_result["status"] == "FATAL":
        raise HTTPException(status_code=500, detail=execution_result["report"])
        
    return APIWebResponse(
        status=execution_result["status"],
        report=execution_result["report"]
    )