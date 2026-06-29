import json
import logging
import asyncio
from langchain_core.tools import tool
from src.db.postgres_client import PostgresPool
from src.db.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

@tool
async def fetch_regulatory_clauses(standard_name: str, equipment_type: str) -> str:
    """
    Traverses the Neo4j Knowledge Graph to extract specific regulatory clauses, 
    environmental norms, or quality standards applicable to a given equipment type.
    Use this to determine what the legal requirement IS.
    """
    cypher_query = """
        MATCH (std:Regulation {name: $standard_name})-[:HAS_CLAUSE]->(c:Clause)
        MATCH (c)-[:APPLIES_TO]->(a:AssetType {name: $equipment_type})
        RETURN c.clause_id AS clause_id, c.text AS requirement, c.thresholds AS thresholds
        LIMIT 5
    """
    try:

        results = await Neo4jClient.execute_read_query(
            cypher_query, 
            parameters={
                "standard_name": standard_name.upper(), 
                "equipment_type": equipment_type
            }
        )
        
        if not results:
            return f"No specific clauses found in {standard_name} for {equipment_type}."
            
        return json.dumps([
            {
                "id": r['clause_id'], 
                "rule": r['requirement'], 
                "limits": r['thresholds']
            } 
            for r in results
        ])
    except Exception as e:
        logger.error(f"Failed to extract regulatory clauses: {e}", exc_info=True)
        return f"Graph DB Error: {str(e)}"


@tool
async def fetch_recent_inspections(asset_id: str, limit: int = 3) -> str:
    """
    Queries the PostgreSQL database to retrieve the latest physical inspection records, 
    quality deviations, and measured parameters for a specific asset.
    Use this to determine what the actual operational state IS.
    """
    
    query = """
        SELECT inspection_date, overall_status, deviations_found, measured_parameters
        FROM asset_inspections
        WHERE asset_id = $1
        ORDER BY inspection_date DESC
        LIMIT $2;
    """
    try:
        async with PostgresPool.get_connection() as conn:
            rows = await conn.fetch(query, asset_id, limit)
        
        if not rows:
            return f"No recorded inspections found for asset: {asset_id}"

        payload = []
        for r in rows:
            telemetry = r['measured_parameters']
            if isinstance(telemetry, str):
                try:
                    telemetry = json.loads(telemetry)
                except json.JSONDecodeError:
                    pass 

            payload.append({
                "date": str(r['inspection_date']),
                "status": r['overall_status'],
                "deviations": r['deviations_found'],
                "telemetry_snapshot": telemetry
            })

        return json.dumps(payload)
        
    except Exception as e:
        logger.error(f"Failed to fetch inspection records: {e}", exc_info=True)
        return f"Relational DB Error: {str(e)}"