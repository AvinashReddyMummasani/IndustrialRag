from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks
import logging
from src.services.failure_intelligence_agent import FailureIntelligenceEngine, AnomalyEvent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/intelligence", tags=["Phase 5 - Proactive Intelligence"])

### Connect alerts to notifications of team emails and Frontend (store in db and fetch to frontend)

def get_intelligence_engine(request: Request) -> FailureIntelligenceEngine:
    if not hasattr(request.app.state, "intelligence_engine") or request.app.state.intelligence_engine is None:
        raise HTTPException(status_code=503, detail="Failure Intelligence Engine offline.")
    return request.app.state.intelligence_engine

# --- SYNCHRONOUS ENDPOINT (Manual Dashboard Lookup) ---
@router.post("/analyze-anomaly")
def manual_anomaly_analysis(
    payload: AnomalyEvent,
    engine: FailureIntelligenceEngine = Depends(get_intelligence_engine)
):
    """Deep dives into an anomaly and returns a predicted failure report immediately."""
    result = engine.evaluate_anomaly(payload)
    if result["status"] == "FATAL":
        raise HTTPException(status_code=500, detail=result["alert"])
    return result

# --- EVENT-DRIVEN ENDPOINT (Proactive Alerting) ---
def proactive_alert_worker(payload: AnomalyEvent, engine: FailureIntelligenceEngine):
    """Background worker that processes the anomaly and pushes alerts to external systems."""
    logger.info(f"Background worker analyzing anomaly for {payload.asset_category}...")
    result = engine.evaluate_anomaly(payload)
    
    if result["status"] == "SUCCESS":
        # In a real production system, you would push this alert to Slack, 
        # an MQTT topic, or an email gateway here.
        logger.warning(f"PROACTIVE ALERT GENERATED:\n{result['alert']}")
    else:
        logger.error(f"Background worker failed to generate alert: {result['alert']}")

@router.post("/trigger-webhook", status_code=202)
def log_work_order_webhook(
    payload: AnomalyEvent,
    background_tasks: BackgroundTasks,
    engine: FailureIntelligenceEngine = Depends(get_intelligence_engine)
):
    """
    Webhook target for CMMS systems. When a work order is logged, this accepts the payload 
    and immediately returns 202. The AI analysis runs in the background.
    """
    background_tasks.add_task(proactive_alert_worker, payload, engine)
    return {"status": "ACCEPTED", "message": "Anomaly queued for predictive analysis."}