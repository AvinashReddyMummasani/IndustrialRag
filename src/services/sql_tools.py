import json
import logging
import asyncio
from langchain_core.tools import tool
from src.db.postgres_client import PostgresPool
from src.db.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

def _fetch_postgres_incidents_sync(asset_id: str, scan_limit: int) -> str:
    """Synchronous core for Postgres execution."""
    query = """
        SELECT work_order_id, failure_code, downtime_hours, technician_notes, execution_date
        FROM maintenance_history
        WHERE asset_id = %s
        ORDER BY execution_date DESC
        LIMIT %s;
    """
    try:
        with PostgresPool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (asset_id, scan_limit))
                rows = cur.fetchall()
        
        if not rows:
            return f"No documented operational incidents or historical context found for: {asset_id}"

        serialized_payload = [
            {
                "work_order_id": r[0],
                "failure_code": r[1],
                "downtime_hours": float(r[2]),
                "notes": r[3],
                "logged_timestamp": str(r[4])
            }
            for r in rows
        ]
        return json.dumps(serialized_payload)
    except Exception as e:
        logger.error(f"Tool execution failure inside relational scan: {e}")
        return f"Error executing relational search execution sequence: {str(e)}"

@tool
async def get_historical_asset_incidents(asset_id: str, scan_limit: int = 5) -> str:
    """
    Queries the PostgreSQL relational database to fetch structured engineering failure codes,
    downtime impact values, and prior repair context for a specific asset token.
    Use this to look up specific structural anomalies across timelines.
    """
    # Offload the synchronous DB call to a thread pool to prevent blocking the async event loop
    return await asyncio.to_thread(_fetch_postgres_incidents_sync, asset_id, scan_limit)

def _fetch_neo4j_procedures_sync(asset_type: str, observed_symptom: str) -> str:
    """Synchronous core for Neo4j execution."""

    # User has to type exact symptom (Try to change it)
    cypher_query = """
        MATCH (a:AssetType {name: $asset_type})-[:HAS_DOCUMENT]->(d:Document)
        MATCH (d)-[:CONTAINS_PROCEDURE]->(p:Procedure)
        WHERE toLower(p.symptom_description) CONTAINS toLower($observed_symptom)
        RETURN p.resolution_steps AS steps, d.filename AS source_file
        LIMIT 3
    """
    try:
        results = Neo4jClient.execute_read_query(
            cypher_query, 
            parameters={"asset_type": asset_type, "observed_symptom": observed_symptom}
        )
        
        if not results:
            return f"No OEM documentation patterns linked to Asset Type: '{asset_type}' matching symptom criteria."
            
        extracted_records = [
            f"Source: {record['source_file']} | Resolution: {record['steps']}"
            for record in results
        ]
        return "\n\n".join(extracted_records)
    except Exception as e:
        logger.error(f"Graph database query tool failure: {e}")
        return f"Error executing Graph Network traversal pattern: {str(e)}"

@tool
async def extract_knowledge_graph_oem_procedures(asset_type: str, observed_symptom: str) -> str:
    """
    Traverses the Neo4j Knowledge Graph to locate technical operational manuals, cross-referenced documentation,
    and manufacturer guidelines linked with specified asset systems and systemic symptoms.
    """
    # Offload the synchronous Graph call to a thread pool
    return await asyncio.to_thread(_fetch_neo4j_procedures_sync, asset_type, observed_symptom)