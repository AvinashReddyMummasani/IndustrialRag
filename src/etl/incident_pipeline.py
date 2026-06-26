import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError
from psycopg2.extras import execute_values
from src.db.postgres_client import PostgresPool

logger = logging.getLogger(__name__)

class IncidentRecordSchema(BaseModel):
    incident_id: str = Field(..., description="Unique ID from the safety/EHS system")
    asset_category: str = Field(..., description="Canonical category (e.g., CENTRIFUGAL_PUMP)")
    severity: str = Field(..., description="Must be CRITICAL, HIGH, MEDIUM, or LOW")
    date_logged: datetime
    root_cause_category: Optional[str] = "UNKNOWN"
    incident_narrative: str = Field(..., min_length=10, description="The unstructured story of the failure")

class IncidentDataPipeline:
    """ETL Pipeline for hybrid Incident Reports (Metadata + Vector Narrative)."""

    def __init__(self, embedding_model):
        # Injects the same SentenceTransformer instance loaded in main.py
        self.embedding_model = embedding_model

    def ingest_incident_batch(self, payload: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validates incident metadata, generates vector embeddings for narratives, 
        and executes a bulk insert into PostgreSQL.
        """
        valid_records = []
        errors = []

        # 1. Schema Validation
        for idx, record in enumerate(payload):
            try:
                validated = IncidentRecordSchema(**record)
                valid_records.append(validated)
            except ValidationError as e:
                errors.append({"index": idx, "error": e.errors()})
                continue

        if not valid_records:
            logger.warning("Incident ingestion aborted: 0 valid records.")
            return {"inserted": 0, "errors": errors}

        # 2. Vectorization (Batching for performance)
        try:
            narratives = [r.incident_narrative for r in valid_records]
            # Encode in batch to utilize PyTorch C++ backend optimization
            embeddings = self.embedding_model.encode(narratives).tolist()
        except Exception as e:
            logger.error(f"Failed to generate embeddings for incident batch: {e}")
            raise RuntimeError("Embedding generation failed during ingestion.")

        # 3. Formulate bulk insertion tuples
        insert_data = [
            (
                r.incident_id,
                r.asset_category,
                r.severity,
                r.date_logged,
                r.root_cause_category,
                r.incident_narrative,
                embeddings[i] # 384-dimensional vector
            )
            for i, r in enumerate(valid_records)
        ]

        # 4. Transactional Database Write
        query = """
            INSERT INTO historical_incidents 
            (incident_id, asset_category, severity, date_logged, root_cause_category, incident_narrative, narrative_embedding)
            VALUES %s
            ON CONFLICT (incident_id) DO UPDATE SET
                severity = EXCLUDED.severity,
                root_cause_category = EXCLUDED.root_cause_category,
                incident_narrative = EXCLUDED.incident_narrative,
                narrative_embedding = EXCLUDED.narrative_embedding;
        """

        try:
            with PostgresPool.get_connection() as conn:
                with conn.cursor() as cur:
                    execute_values(cur, query, insert_data, page_size=500)
            
            logger.info(f"Successfully ingested {len(valid_records)} historical incidents.")
            return {"inserted": len(valid_records), "errors": errors}
            
        except Exception as e:
            logger.error(f"Incident ingestion database write failed: {e}")
            raise RuntimeError("PostgreSQL transaction failed during incident ingestion.")