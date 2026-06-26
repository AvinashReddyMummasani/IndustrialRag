import logging
from datetime import date
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError
from src.db.postgres_client import PostgresPool  # Assuming your file name aligns with the class

logger = logging.getLogger(__name__)

class OperationalAssetSchema(BaseModel):
    asset_id: str
    asset_name: str
    asset_type: str
    installation_date: Optional[date] = None
    criticality: str = "MEDIUM"
    current_status: str = "OPERATIONAL"

class MaintenanceRecordSchema(BaseModel):
    work_order_id: str
    asset_id: str
    failure_code: str
    downtime_hours: float = Field(..., ge=0.0)
    technician_notes: str

class CMMSDataPipeline:
    """Production ETL Pipeline for structured Asset registries and Work Order processing."""

    @staticmethod
    def ingest_assets(asset_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validates and processes batch installations/updates of industrial hardware targets."""
        valid_assets: List[OperationalAssetSchema] = []
        errors: List[Dict[str, Any]] = []

        for index, record in enumerate(asset_records):
            try:
                valid_assets.append(OperationalAssetSchema(**record))
            except ValidationError as e:
                errors.append({"index": index, "validation_error": e.errors()})

        if not valid_assets:
            return {"inserted": 0, "status": "SKIPPED", "errors": errors}

        query = """
            INSERT INTO industrial_assets (asset_id, asset_name, asset_type, installation_date, criticality, current_status)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (asset_id) DO UPDATE SET
                current_status = EXCLUDED.current_status,
                criticality = EXCLUDED.criticality;
        """

        try:
            with PostgresPool.get_connection() as conn:
                with conn.cursor() as cur:
                    batch_data = [
                        (a.asset_id, a.asset_name, a.asset_type, a.installation_date, a.criticality, a.current_status)
                        for a in valid_assets
                    ]
                    cur.executemany(query, batch_data)
            return {"inserted": len(valid_assets), "status": "SUCCESS", "errors": errors}
        except Exception as e:
            logger.error(f"Failed to commit asset registry batch: {e}")
            raise e

    @staticmethod
    def ingest_work_orders(work_orders: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Bulk updates relational historic work logs into operational analytics scopes."""
        valid_logs: List[MaintenanceRecordSchema] = []
        errors: List[Dict[str, Any]] = []

        for index, record in enumerate(work_orders):
            try:
                valid_logs.append(MaintenanceRecordSchema(**record))
            except ValidationError as e:
                errors.append({"index": index, "validation_error": e.errors()})

        if not valid_logs:
            return {"inserted": 0, "status": "SKIPPED", "errors": errors}

        query = """
            INSERT INTO maintenance_history (work_order_id, asset_id, failure_code, downtime_hours, technician_notes)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (work_order_id) DO NOTHING;
        """

        try:
            with PostgresPool.get_connection() as conn:
                with conn.cursor() as cur:
                    batch_data = [
                        (w.work_order_id, w.asset_id, w.failure_code, w.downtime_hours, w.technician_notes)
                        for w in valid_logs
                    ]
                    cur.executemany(query, batch_data)
            return {"inserted": len(valid_logs), "status": "SUCCESS", "errors": errors}
        except Exception as e:
            logger.error(f"Failed to execute maintenance log execution batch: {e}")
            raise e