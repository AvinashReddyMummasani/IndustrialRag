import json
import logging
from src.db.postgres_client import PostgresPool
from src.db.neo4j_client import Neo4jClient
from typing import List

logger = logging.getLogger(__name__)

async def fetch_postgres_incidents_async(asset_id: str, scan_limit: int = 5) -> str:
    """
    Native asynchronous query to PostgreSQL for historical maintenance incidents.
    """
    query = """
        SELECT work_order_id, failure_code, downtime_hours, technician_notes, execution_date
        FROM maintenance_history
        WHERE asset_id = $1
        ORDER BY execution_date DESC
        LIMIT $2;
    """
    try:

        async with PostgresPool.acquire() as conn:
            rows = await conn.fetch(query, asset_id, scan_limit)
        
        if not rows:
            return f"No documented operational incidents or historical context found for: {asset_id}"

        serialized_payload = [
            {
                "work_order_id": r['work_order_id'],
                "failure_code": r['failure_code'],
                "downtime_hours": float(r['downtime_hours']),
                "notes": r['technician_notes'],
                "logged_timestamp": str(r['execution_date'])
            }
            for r in rows
        ]
        return json.dumps(serialized_payload)
    except Exception as e:
        logger.error(f"Postgres relational scan failed for asset {asset_id}: {e}", exc_info=True)
        return f"Error executing relational search execution sequence: {str(e)}"


import logging
from src.parsers import generate_vector_async 
from src.db.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

async def fetch_neo4j_procedures_async(asset_type: str, symptom_vector: List[float]) -> str:
    """
    Executes a locally bounded vector similarity search in Neo4j.
    Traverses exact structural relationships first, then applies cosine similarity 
    to prevent cross-contamination of asset documentation.
    """
    

    # 1. Cypher: Structural bounded traversal + Cosine Similarity
    # Note: Requires Neo4j 5.x for vector.similarity.cosine
    cypher_query = """
        MATCH (a:AssetType {name: $asset_type})-[:HAS_DOCUMENT]->(d:Document)
        MATCH (d)-[:CONTAINS_PROCEDURE]->(p:Procedure)
        // Ensure the node has an embedding before calculating to prevent null errors
        WHERE p.embedding IS NOT NULL
        
        // Calculate similarity between the runtime vector and the stored node vector
        WITH p, d, vector.similarity.cosine(p.embedding, $symptom_vector) AS similarity_score
        
        // Discard low-confidence matches (Threshold tuning required based on your embedding model)
        WHERE similarity_score > 0.65 
        
        RETURN p.resolution_steps AS steps, d.filename AS source_file, similarity_score
        ORDER BY similarity_score DESC
        LIMIT 3
    """
    
    try:
        results = await Neo4jClient.execute_read_query(
            cypher_query, 
            parameters={
                "asset_type": asset_type, 
                "symptom_vector": symptom_vector
            }
        )
        
        if not results:
            return f"No OEM documentation patterns linked to Asset Type: '{asset_type}' met the semantic similarity threshold."
            
        extracted_records = [
            f"Source: {record['source_file']} (Confidence: {record['similarity_score']:.2f}) | Resolution: {record['steps']}"
            for record in results
        ]
        return "\n\n".join(extracted_records)
        
    except Exception as e:
        logger.error(f"Graph database bounded vector traversal failed for {asset_type}: {e}", exc_info=True)
        return f"Error executing Graph Network traversal pattern: {str(e)}"