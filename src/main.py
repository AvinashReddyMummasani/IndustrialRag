import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from src.db.postgres_client import PostgresPool
from src.db.neo4j_client import Neo4jClient

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
    logger.info("Initiating database connection pools...")
    try:
        PostgresPool.initialize()
        logger.info("PostgreSQL pool initialized successfully.")
        
        Neo4jClient.initialize()
        logger.info("Neo4j driver initialized successfully.")
    except Exception as e:
        logger.critical(f"Fatal boot failure. Database initialization aborted: {e}")
        raise RuntimeError("Halting boot sequence due to database connection failure.") from e

    logger.info("Loading heavy ML models and Agent orchestrators into worker memory state...")
    try:

        from src.services.file_classifier import IngestionPipeline
        from src.services.knowledge_copilot import KnowledgeCopilot
        from src.services.rca_agent import IndustrialRCAEngine  
        from src.services.compliance_agents import RegulatoryComplianceEngine
        from src.services.failure_intelligence_agent import FailureIntelligenceEngine # check it
        from src.etl.incident_pipeline import IncidentDataPipeline # check it
        from sentence_transformers import SentenceTransformer 
        from langchain_groq import ChatGroq
        
        logger.info("Downloading/Loading SentenceTransformer weights (all-MiniLM-L6-v2)...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Initializing Vision API Client...")
    
        vision_client = ChatGroq(
            model="llama-3.2-90b-vision-preview", 
            temperature=0.0, 
            max_retries=3
        )
        
        app.state.pipeline = IngestionPipeline(model=embedding_model,vision_clinet=vision_client)
        app.state.incident_pipeline = IncidentDataPipeline(embedding_model=embedding_model)
        
        app.state.copilot = KnowledgeCopilot(embedding_model=embedding_model)
        app.state.rca_engine = IndustrialRCAEngine()
        app.state.compliance_engine = RegulatoryComplianceEngine()
        
        app.state.intelligence_engine = FailureIntelligenceEngine(embedding_model=embedding_model)
        
        logger.info("ML Models, ETL Pipelines, and Agents attached to application state.")
    except Exception as e:
        logger.critical(f"Failed to instantiate system pipelines: {e}")
        raise RuntimeError("Halting boot sequence due to pipeline load failure.") from e

    yield 

    # Clean up code

    logger.info("Initiating graceful shutdown sequence...")
    try:
        Neo4jClient.close()
        logger.info("Neo4j driver closed.")
        
        if hasattr(PostgresPool, '_pool') and PostgresPool._pool:
            PostgresPool._pool.closeall()
            logger.info("PostgreSQL connection pool closed.")
    
        app.state.pipeline = None
        app.state.incident_pipeline = None
        app.state.copilot = None
        app.state.rca_engine = None
        app.state.compliance_engine = None
        app.state.intelligence_engine = None
            
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