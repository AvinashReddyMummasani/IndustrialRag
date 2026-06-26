import logging
from datetime import datetime
from typing import List, Dict, Any
from pydantic import BaseModel, ValidationError
from src.db.timescale_client import TimescaleManager

logger = logging.getLogger(__name__)

class TelemetryPayload(BaseModel):
    measured_at: datetime
    asset_id: str
    metric_name: str
    metric_value: float

class TelemetryPipeline:
    """Processes and validates high-velocity stream/batch telemetry data."""

    @staticmethod
    def ingest_sensor_batch(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validates raw dictionary payloads and writes them to the time-series DB.
        """
        valid_records = []
        errors = 0

        for record in payloads:
            try:
                validated = TelemetryPayload(**record)
                # Flatten to tuple for execute_values
                valid_records.append((
                    validated.measured_at,
                    validated.asset_id,
                    validated.metric_name,
                    validated.metric_value
                ))
            except ValidationError:
                errors += 1
                continue

        if not valid_records:
            logger.warning(f"Telemetry batch dropped. {errors} schema validation failures.")
            return {"inserted": 0, "errors": errors}

        try:
            inserted_count = TimescaleManager.execute_bulk_insert(valid_records)
            return {"inserted": inserted_count, "errors": errors}
        except Exception as e:
            logger.error(f"Telemetry pipeline failed during database write: {e}")
            raise e