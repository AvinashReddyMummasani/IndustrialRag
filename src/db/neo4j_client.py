import os
import logging
from neo4j import AsyncGraphDatabase

logger = logging.getLogger(__name__)

class Neo4jClient:
    _driver = None

    @classmethod
    async def initialize(cls):
        """Establishes and verifies persistent async socket connections to the graph database."""
        if not cls._driver:
            uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            user = os.getenv("NEO4J_USER", "neo4j")
            password = os.getenv("NEO4J_PASSWORD")
            
            if not password:
                raise ValueError("NEO4J_PASSWORD environment variable is required.")
                
            cls._driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
            await cls._verify_connectivity()

    @classmethod
    async def _verify_connectivity(cls):
        if not cls._driver:
            raise RuntimeError("Neo4j driver context was not instantiated.")
        await cls._driver.verify_connectivity()
        logger.info("Neo4j graph clustering topology verified successfully.")

    @classmethod
    async def close(cls):
        if cls._driver:
            await cls._driver.close()
            cls._driver = None

    @classmethod
    async def upsert_graph_data(cls, entities: list, relationships: list, document_id: str):
        """Atomic async write path for a single document's lineage mapping."""
        if not cls._driver:
            raise RuntimeError("Neo4j client driver uninitialized.")

        entity_query = """
        UNWIND $entities AS ent
        CALL apoc.merge.node([ent.entity_type], {entity_id: ent.entity_id}, ent.properties, ent.properties)
        YIELD node
        RETURN count(node)
        """

        relationship_query = """
        UNWIND $relationships AS rel
        MATCH (source {entity_id: rel.source_id})
        MATCH (target {entity_id: rel.target_id})
        CALL apoc.merge.relationship(source, rel.relation_type, {}, {document_id: $doc_id}, target)
        YIELD rel as relationship
        RETURN count(relationship)
        """

        # Brutally optimized extraction handling both dicts and Pydantic objects safely
        ent_params = [
            {
                "entity_id": e["entity_id"] if isinstance(e, dict) else e.entity_id,
                "entity_type": str(e["entity_type"] if isinstance(e, dict) else e.entity_type).upper(),
                "properties": e.get("properties", {}) if isinstance(e, dict) else getattr(e, "properties", {})
            }
            for e in entities
        ]

        rel_params = [
            {
                "source_id": r["source_id"] if isinstance(r, dict) else r.source_id,
                "target_id": r["target_id"] if isinstance(r, dict) else r.target_id,
                "relation_type": str(r["relation_type"] if isinstance(r, dict) else r.relation_type).upper()
            }
            for r in relationships
        ]

        async with cls._driver.session() as session:
            if ent_params:
                async def _write_entities(tx):
                    result = await tx.run(entity_query, entities=ent_params)
                    return await result.consume()
                await session.execute_write(_write_entities)
            
            if rel_params:
                async def _write_relationships(tx):
                    result = await tx.run(relationship_query, relationships=rel_params, doc_id=document_id)
                    return await result.consume()
                await session.execute_write(_write_relationships)
    
    @classmethod
    async def execute_read_query(cls, cypher_query: str, parameters: dict = None) -> list:
        """Executes a parameterized async read-only Cypher transaction."""
        if not cls._driver:
            raise RuntimeError("Neo4j client driver uninitialized. Call initialize() first.")
        
        parameters = parameters or {}
        
        async def _read_transaction(tx):
            result = await tx.run(cypher_query, **parameters)
            return [record.data() async for record in result]

        try:
            async with cls._driver.session() as session:
                return await session.execute_read(_read_transaction)
        except Exception as e:
            logger.error(f"Neo4j Read Transaction Failed: {e}")
            raise e