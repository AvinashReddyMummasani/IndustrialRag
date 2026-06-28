# import os
# import logging
# from neo4j import GraphDatabase

# logger = logging.getLogger(__name__)

# class Neo4jClient:
#     _driver = None

#     @classmethod
#     def initialize(cls):
#         """Establishes and verifies persistent socket connections to the graph database."""
#         if not cls._driver:
#             uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
#             user = os.getenv("NEO4J_USER", "neo4j")
#             password = os.getenv("NEO4J_PASSWORD")
            
#             if not password:
#                 raise ValueError("NEO4J_PASSWORD environment variable is required.")
                
#             cls._driver = GraphDatabase.driver(uri, auth=(user, password))
#             cls._verify_connectivity()

#     @classmethod
#     def _verify_connectivity(cls):
#         if not cls._driver:
#             raise RuntimeError("Neo4j driver context was not instantiated.")
#         cls._driver.verify_connectivity()
#         logger.info("Neo4j graph clustering topology verified successfully.")

#     @classmethod
#     def close(cls):
#         if cls._driver:
#             cls._driver.close()
#             cls._driver = None

#     @classmethod
#     def upsert_graph_data(cls, entities: list, relationships: list, document_id: str):
#         """Atomic batch write path for document lineage mapping."""
#         if not cls._driver:
#             raise RuntimeError("Neo4j client driver uninitialized.")

#         # Constraints verification: unique entity IDs optimize matching runtime
#         entity_query = """
#         UNWIND $entities AS ent
#         CALL apoc.merge.node([ent.entity_type], {entity_id: ent.entity_id}, ent.properties, ent.properties)
#         YIELD node
#         RETURN count(node)
#         """

#         relationship_query = """
#         UNWIND $relationships AS rel
#         MATCH (source {entity_id: rel.source_id})
#         MATCH (target {entity_id: rel.target_id})
#         CALL apoc.merge.relationship(source, rel.relation_type, {}, {document_id: $doc_id}, target)
#         YIELD rel as relationship
#         RETURN count(relationship)
#         """

#         ent_params = [
#             {
#                 "entity_id": e.entity_id,
#                 "entity_type": e.entity_type.upper() if hasattr(e.entity_type, 'upper') else e.entity_type,
#                 "properties": e.properties or {}
#             }
#             for e in entities
#         ]

#         rel_params = [
#             {
#                 "source_id": r.source_id,
#                 "target_id": r.target_id,
#                 "relation_type": r.relation_type.upper() if hasattr(r.relation_type, 'upper') else r.relation_type
#             }
#             for r in relationships
#         ]

#         with cls._driver.session() as session:
#             if ent_params:
#                 session.execute_write(lambda tx: tx.run(entity_query, entities=ent_params))
#             if rel_params:
#                 session.execute_write(lambda tx: tx.run(relationship_query, relationships=rel_params, doc_id=document_id))
    
#     @classmethod
#     def execute_read_query(cls, cypher_query: str, parameters: dict = None) -> list:
#         """
#         Executes a parameterized read-only Cypher transaction.
#         Designed for strict data retrieval by analytics and agent execution layers.
#         """
#         if not cls._driver:
#             raise RuntimeError("Neo4j client driver uninitialized. Call initialize() first.")
        
#         parameters = parameters or {}
        
#         def _read_transaction(tx):
#             result = tx.run(cypher_query, **parameters)
#             return result.data()  # .data() safely extracts the payload to native Python dicts

#         try:
#             with cls._driver.session() as session:
#                 return session.execute_read(_read_transaction)
#         except Exception as e:
#             logger.error(f"Neo4j Read Transaction Failed: {e}")
#             raise e

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
        """Atomic batch async write path for document lineage mapping."""
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

        ent_params = [
            {
                "entity_id": e.entity_id,
                "entity_type": e.entity_type.upper() if hasattr(e.entity_type, 'upper') else e.entity_type,
                "properties": e.properties or {}
            }
            for e in entities
        ]

        rel_params = [
            {
                "source_id": r.source_id,
                "target_id": r.target_id,
                "relation_type": r.relation_type.upper() if hasattr(r.relation_type, 'upper') else r.relation_type
            }
            for r in relationships
        ]

        async with cls._driver.session() as session:
            if ent_params:
                await session.execute_write(lambda tx: tx.run(entity_query, entities=ent_params))
            if rel_params:
                await session.execute_write(lambda tx: tx.run(relationship_query, relationships=rel_params, doc_id=document_id))
    
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