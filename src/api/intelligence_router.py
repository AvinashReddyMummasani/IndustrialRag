import logging
import asyncio
from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks

from src.services.failure_intelligence_agent import FailureIntelligenceEngine, AnomalyEvent
from src.services.notification_service import NotificationService
from src.db.notification_repo import NotificationRepo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/Failure_intelligence", tags=["Proactive Intelligence"])

def get_intelligence_engine(request: Request) -> FailureIntelligenceEngine:
    if not hasattr(request.app.state, "intelligence_engine") or request.app.state.intelligence_engine is None:
        raise HTTPException(status_code=503, detail="Failure Intelligence Engine offline.")
    return request.app.state.intelligence_engine

# BACKGROUND WORKER
async def proactive_alert_worker(payload: AnomalyEvent, engine: FailureIntelligenceEngine):
    """
    Orchestrates LLM evaluation, DB persistence (via thread pool), and async email dispatch.
    """
    logger.info(f"Worker initiated analysis for {payload.asset_category}")
    
    result = await engine.evaluate_anomaly(payload)
    
    if result.get("status") == "SUCCESS":
        alert_text = result["alert"]
        
        try:

            users_to_notify = await asyncio.to_thread(
                NotificationRepo.save_alert_and_get_targets,
                asset_id=payload.asset_category,
                severity="CRITICAL",
                report=alert_text
            )
            
            # Non-blocking async network dispatch
            notifier = NotificationService()
            await notifier.broadcast_alert(users_to_notify, payload.asset_category, alert_text)
            
        except Exception as db_err:
            logger.error(f"Worker failed during DB persistence or broadcast: {db_err}")
    else:
        logger.error(f"Intelligence Engine returned non-success state: {result.get('alert')}")


@router.post("/trigger-webhook", status_code=202)
async def log_work_order_webhook(
    payload: AnomalyEvent,
    background_tasks: BackgroundTasks,
    engine: FailureIntelligenceEngine = Depends(get_intelligence_engine)
):
    """
    Webhook entry point. Queues the predictive analysis to avoid blocking the upstream CMMS.
    """
    background_tasks.add_task(proactive_alert_worker, payload, engine)
    return {"status": "ACCEPTED", "message": "Anomaly queued for background analysis and broadcast."}


@router.get("/alerts", tags=["Frontend Integration"])
async def fetch_recent_alerts(limit: int = 10):
    """
    Frontend endpoint to retrieve the latest alerts.
    Wrapped in to_thread to prevent psycopg2 from blocking the main loop.
    """
    def _fetch_sync():
        from src.db.postgres_client import PostgresPool
        query = """
            SELECT id, asset_id, severity, report_payload, created_at 
            FROM alert_records 
            ORDER BY created_at DESC LIMIT %s
        """
        with PostgresPool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (limit,))
                rows = cur.fetchall()
                return [
                    {"id": r[0], "asset_id": r[1], "severity": r[2], "report_payload": r[3], "timestamp": str(r[4])} 
                    for r in rows
                ]
    
    try:
        data = await asyncio.to_thread(_fetch_sync)
        return {"status": "SUCCESS", "data": data}
    except Exception as e:
        logger.error(f"Failed to fetch alerts: {e}")
        raise HTTPException(status_code=500, detail="Database fetch failure.")