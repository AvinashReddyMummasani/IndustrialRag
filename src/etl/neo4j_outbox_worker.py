import asyncio
import json
import logging

from src.core.schemas import ExtractedEntity, EntityRelationship
from src.db.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

BATCH_SIZE = 10
POLL_INTERVAL = 5

async def process_pending_jobs(pool):
    logger.info("Neo4j Outbox Worker started.")

    while True:
        try:
            async with pool.acquire() as conn:
                # FOR UPDATE SKIP LOCKED is mandatory to prevent concurrent workers 
                # from processing the same row and causing deadlocks or duplicate inserts.
                rows = await conn.fetch(
                    """
                    SELECT id, document_id, payload
                    FROM neo4j_outbox
                    WHERE status = 'PENDING'
                    ORDER BY created_at
                    LIMIT $1
                    FOR UPDATE SKIP LOCKED
                    """,
                    BATCH_SIZE,
                )

                if not rows:
                    await asyncio.sleep(POLL_INTERVAL)
                    continue

                successful_job_ids = []
                failed_job_ids = []

                # Process Neo4j inserts iteratively
                for row in rows:
                    job_id = row["id"]
                    document_id = row["document_id"]
                    payload = row["payload"]

                    try:
                        if isinstance(payload, str):
                            payload = json.loads(payload)

                        entities = [
                            ExtractedEntity(**entity) if isinstance(entity, dict) else entity
                            for entity in payload.get("entities", [])
                        ]

                        relationships = [
                            EntityRelationship(**relationship) if isinstance(relationship, dict) else relationship
                            for relationship in payload.get("relationships", [])
                        ]

                        await Neo4jClient.upsert_graph_data(
                            entities=entities,
                            relationships=relationships,
                            document_id=document_id,
                        )
                        
                        successful_job_ids.append(job_id)
                        logger.info(f"Successfully processed Neo4j outbox job {job_id} for document {document_id}.")

                    except Exception as e:
                        failed_job_ids.append(job_id)
                        logger.exception(f"Failed to process Neo4j outbox job {job_id} for document {document_id}: {e}")

                # Bulk update successful Postgres records
                if successful_job_ids:
                    await conn.execute(
                        """
                        UPDATE neo4j_outbox
                        SET status = 'COMPLETED'
                        WHERE id = ANY($1)
                        """,
                        successful_job_ids
                    )

                # Bulk update failed Postgres records
                if failed_job_ids:
                    await conn.execute(
                        """
                        UPDATE neo4j_outbox
                        SET status = 'FAILED'
                        WHERE id = ANY($1)
                        """,
                        failed_job_ids
                    )

        except asyncio.CancelledError:
            logger.info("Termination signal received. Shutting down Neo4j outbox worker.")
            break
        except Exception:
            logger.exception("Unexpected error in Neo4j outbox worker.")

        try:
            await asyncio.sleep(POLL_INTERVAL)
        except asyncio.CancelledError:
            logger.info("Termination signal received during sleep. Shutting down worker.")
            break