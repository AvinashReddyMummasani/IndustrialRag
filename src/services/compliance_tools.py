import json
import logging
import asyncio
from langchain_core.tools import tool
from src.db.postgres_client import PostgresPool
from src.db.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

def _fetch_regulatory_clauses_sync(standard_name: str, equipment_type: str) -> str:
    """Synchronous core for Neo4j regulatory extraction."""
    cypher_query = """
        MATCH (std:Regulation {name: $standard_name})-[:HAS_CLAUSE]->(c:Clause)
        MATCH (c)-[:APPLIES_TO]->(a:AssetType {name: $equipment_type})
        RETURN c.clause_id AS clause_id, c.text AS requirement, c.thresholds AS thresholds
        LIMIT 5
    """
    try:
        results = Neo4jClient.execute_read_query(
            cypher_query, 
            parameters={"standard_name": standard_name.upper(), "equipment_type": equipment_type}
        )
        
        if not results:
            return f"No specific clauses found in {standard_name} for {equipment_type}."
            
        return json.dumps([
            {"id": r['clause_id'], "rule": r['requirement'], "limits": r['thresholds']} 
            for r in results
        ])
    except Exception as e:
        logger.error(f"Failed to extract regulatory clauses: {e}", exc_info=True)
        return f"Graph DB Error: {str(e)}"

@tool
async def fetch_regulatory_clauses(standard_name: str, equipment_type: str) -> str:
    """
    Traverses the Neo4j Knowledge Graph to extract specific regulatory clauses, 
    environmental norms, or quality standards applicable to a given equipment type.
    Use this to determine what the legal requirement IS.
    """
    # Offload the blocking Graph network call to a worker thread
    return await asyncio.to_thread(_fetch_regulatory_clauses_sync, standard_name, equipment_type)


def _fetch_recent_inspections_sync(asset_id: str, limit: int) -> str:
    """Synchronous core for PostgreSQL inspection retrieval."""
    query = """
        SELECT inspection_date, overall_status, deviations_found, measured_parameters
        FROM asset_inspections
        WHERE asset_id = %s
        ORDER BY inspection_date DESC
        LIMIT %s;
    """
    try:
        with PostgresPool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (asset_id, limit))
                rows = cur.fetchall()
        
        if not rows:
            return f"No recorded inspections found for asset: {asset_id}"

        payload = [
            {
                "date": str(r[0]),
                "status": r[1],
                "deviations": r[2],
                "telemetry_snapshot": r[3] # JSONB field
            }
            for r in rows
        ]
        return json.dumps(payload)
    except Exception as e:
        logger.error(f"Failed to fetch inspection records: {e}", exc_info=True)
        return f"Relational DB Error: {str(e)}"

@tool
async def fetch_recent_inspections(asset_id: str, limit: int = 3) -> str:
    """
    Queries the PostgreSQL database to retrieve the latest physical inspection records, 
    quality deviations, and measured parameters for a specific asset.
    Use this to determine what the actual operational state IS.
    """
    
    return await asyncio.to_thread(_fetch_recent_inspections_sync, asset_id, limit)