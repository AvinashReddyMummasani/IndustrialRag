from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from src.services.compliance_agents import RegulatoryComplianceEngine, AuditRequest

router = APIRouter(prefix="/api/v1/compliance", tags=["Phase 4 - Regulatory Intelligence"])

class AuditResponse(BaseModel):
    status: str
    report: str

def get_compliance_engine(request: Request) -> RegulatoryComplianceEngine:
    if not hasattr(request.app.state, "compliance_engine") or request.app.state.compliance_engine is None:
        raise HTTPException(status_code=503, detail="Compliance Engine offline.")
    return request.app.state.compliance_engine

@router.post("/run-audit", response_model=AuditResponse)
def execute_compliance_audit(
    payload: AuditRequest,
    engine: RegulatoryComplianceEngine = Depends(get_compliance_engine)
):
    """Generates an auto-audit report comparing equipment state to legal standards."""
    result = engine.execute_audit(payload)
    
    if result["status"] == "FATAL":
        raise HTTPException(status_code=500, detail=result["report"])
        
    return AuditResponse(status=result["status"], report=result["report"])