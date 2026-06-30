import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import asyncio

load_dotenv()

from src.db.postgres_client import PostgresPool
from src.db.neo4j_client import Neo4jClient
from src.etl.neo4j_outbox_worker import process_pending_jobs

from src.api import (
    ingestion_router, 
    copilot_router, 
    rca_agent_router, 
    compliance_router, 
    intelligence_router
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("api_main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages global application state. 
    Guarantees teardown execution even if the application crashes.
    """
    logger.info("Initiating async database connection pools...")
    try:
        await PostgresPool.initialize()
        logger.info("PostgreSQL asyncpg pool initialized successfully.")
        
        await Neo4jClient.initialize()
        logger.info("Neo4j async driver initialized successfully.")
    except Exception as e:
        logger.critical(f"Fatal boot failure. Database initialization aborted: {e}")
        raise RuntimeError("Halting boot sequence due to database connection failure.") from e

    logger.info("Loading ML models and Agent orchestrators into worker memory state...")
    try:
        from src.services.file_classifier import IngestionPipeline
        from src.services.knowledge_copilot import KnowledgeCopilot
        from src.services.rca_agent import IndustrialRCAEngine  
        from src.services.compliance_agents import RegulatoryComplianceEngine
        from src.services.failure_intelligence_agent import FailureIntelligenceEngine
        from src.etl.incident_pipeline import IncidentDataPipeline
        from sentence_transformers import SentenceTransformer 
        from langchain_groq import ChatGroq
        
        # Note: SentenceTransformer loading is CPU bound and synchronous.
        # It is acceptable here in the lifespan block before accepting traffic.
        logger.info("Loading SentenceTransformer weights (all-MiniLM-L6-v2)...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Initializing Vision API Client...")
    
        vision_client = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct", 
            temperature=0.0, 
            max_retries=3
        )
        heavy_llm = "llama-3.3-70b-versatile"
        light_llm ="llama-3.1-8b-instant"

        app.state.pipeline = IngestionPipeline(model=embedding_model, vision_client=vision_client,llm=heavy_llm)
        app.state.incident_pipeline = IncidentDataPipeline(embedding_model=embedding_model)
        
        app.state.copilot = KnowledgeCopilot(embedding_model=embedding_model,llm=heavy_llm)
        app.state.rca_engine = IndustrialRCAEngine(embedding_model=embedding_model,llm=heavy_llm)
        app.state.compliance_engine = RegulatoryComplianceEngine(llm=heavy_llm)
        app.state.intelligence_engine = FailureIntelligenceEngine(embedding_model=embedding_model,llm=heavy_llm)
        
        logger.info("ML Models, ETL Pipelines, and Agents attached to application state.")
    except Exception as e:
        logger.critical(f"Failed to instantiate system pipelines: {e}")
        raise RuntimeError("Halting boot sequence due to pipeline load failure.") from e
    
    logger.info("Starting Neo4j outbox worker...")
    # Bind using a consistent namespace
    app.state.outbox_worker = asyncio.create_task(process_pending_jobs(PostgresPool.get_pool()))
    logger.info("Neo4j outbox worker started.")

    yield 

    logger.info("Initiating graceful shutdown sequence...")
    try:
        # 1. Terminate background workers FIRST
        if hasattr(app.state, "outbox_worker"):
            app.state.outbox_worker.cancel()
            try:
                await asyncio.wait_for(app.state.outbox_worker, timeout=5.0)
                logger.info("Neo4j Outbox Worker stopped gracefully.")
            except asyncio.CancelledError:
                logger.info("Neo4j Outbox Worker cancelled cleanly.")
            except asyncio.TimeoutError:
                logger.warning("Neo4j Outbox Worker shutdown timed out. Forcing termination.")
        
        # 2. Teardown Database Connections LAST
        logger.info("Closing Neo4j async driver...")
        await Neo4jClient.close()
        logger.info("Neo4j async driver closed.")
        
        if hasattr(PostgresPool, '_pool') and PostgresPool._pool:
            logger.info("Closing PostgreSQL asyncpg connection pool...")
            await PostgresPool._pool.close()
            logger.info("PostgreSQL asyncpg connection pool closed.")
    
        # 3. Clear application state memory
        app.state.pipeline = None
        app.state.incident_pipeline = None
        app.state.copilot = None
        app.state.rca_engine = None
        app.state.compliance_engine = None
        app.state.intelligence_engine = None
        app.state.outbox_worker = None
            
    except Exception as e:
        logger.error(f"Error encountered during connection teardown: {e}")


app = FastAPI(
    title="Industrial Knowledge Intelligence API",
    description="Unified Asset & Operations Brain - Full Scope",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingestion_router.router)
app.include_router(copilot_router.router)
app.include_router(rca_agent_router.router)
app.include_router(compliance_router.router)
app.include_router(intelligence_router.router)