import os
import logging
from contextlib import contextmanager
from psycopg2.pool import SimpleConnectionPool

logger = logging.getLogger(__name__)


class PostgresPool:
    _pool = None

    @classmethod
    def initialize(cls):
        """Initializes the thread-safe connection pool and bootstraps schemas."""
        if not cls._pool:
            dsn = os.getenv("DATABASE_URL")
            if not dsn:
                raise ValueError("DATABASE_URL environment variable is missing.")

            cls._pool = SimpleConnectionPool(
                minconn=2,
                maxconn=20,
                dsn=dsn
            )
            cls._bootstrap_db()

    @classmethod
    @contextmanager
    def get_connection(cls):
        """Context manager providing transactional connection lending."""
        if not cls._pool:
            raise RuntimeError("Database connection pool has not been initialized. Call initialize() first.")

        conn = cls._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Postgres Transaction Rollback: {e}")
            raise e
        finally:
            cls._pool.putconn(conn)

    @classmethod
    def _bootstrap_db(cls):
        """Executes strict schema migrations and vector optimization indexing."""
        with cls.get_connection() as conn:
            with conn.cursor() as cur:

                # 1. Enable pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                # 2. Master Document Directory
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id TEXT PRIMARY KEY,
                        filename TEXT NOT NULL,
                        doc_type TEXT NOT NULL,
                        parent_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
                        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                # 3. Dense Vector Text Chunks Table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        id SERIAL PRIMARY KEY,
                        document_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
                        chunk_text TEXT NOT NULL,
                        embedding vector(1536)
                    );
                """)

                # 4. Outbox table for transactional integrity with Neo4j
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS neo4j_outbox (
                        id SERIAL PRIMARY KEY,
                        document_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
                        payload JSONB NOT NULL,
                        status TEXT NOT NULL DEFAULT 'PENDING',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                # 5. Indexing for high-throughput vector scans
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS document_chunks_hnsw_idx
                    ON document_chunks USING hnsw (embedding vector_cosine_ops);
                """)

                # 6. Industrial Assets
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS industrial_assets (
                        asset_id TEXT PRIMARY KEY,
                        asset_name TEXT NOT NULL,
                        asset_type TEXT NOT NULL,
                        installation_date DATE,
                        criticality TEXT CHECK (criticality IN ('HIGH', 'MEDIUM', 'LOW')) DEFAULT 'MEDIUM',
                        current_status TEXT DEFAULT 'OPERATIONAL'
                    );
                """)

                # 7. Maintenance History
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS maintenance_history (
                        work_order_id TEXT PRIMARY KEY,
                        asset_id TEXT REFERENCES industrial_assets(asset_id) ON DELETE CASCADE,
                        failure_code TEXT NOT NULL,
                        downtime_hours NUMERIC(6, 2) NOT NULL,
                        technician_notes TEXT NOT NULL,
                        execution_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_maintenance_history_asset_id
                    ON maintenance_history(asset_id);
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_maintenance_history_failure_code
                    ON maintenance_history(failure_code);
                """)

                # 8. Regulatory Standards
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS regulatory_standards (
                        standard_id VARCHAR(50) PRIMARY KEY,
                        issuing_body VARCHAR(100) NOT NULL,
                        title TEXT NOT NULL,
                        last_updated DATE
                    );
                """)

                # 9. Asset Inspections
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS asset_inspections (
                        inspection_id VARCHAR(50) PRIMARY KEY,
                        asset_id VARCHAR(50) REFERENCES industrial_assets(asset_id),
                        inspector_name VARCHAR(100),
                        inspection_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        overall_status VARCHAR(20) CHECK (overall_status IN ('PASS', 'FAIL', 'CONDITIONAL')),
                        deviations_found TEXT,
                        measured_parameters JSONB
                    );
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_inspections_asset
                    ON asset_inspections(asset_id);
                """)

                # 10. Historical Incidents & Lessons Learned
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS historical_incidents (
                        incident_id VARCHAR(50) PRIMARY KEY,
                        asset_category VARCHAR(100) NOT NULL,
                        severity VARCHAR(20) CHECK (severity IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW')),
                        date_logged TIMESTAMP WITH TIME ZONE,
                        root_cause_category VARCHAR(100),
                        incident_narrative TEXT NOT NULL,
                        narrative_embedding vector(384)
                    );
                """)

                # 11. User Management for RBAC & Notifications
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id TEXT PRIMARY KEY,
                        email TEXT UNIQUE NOT NULL,
                        hashed_password TEXT NOT NULL,
                        full_name TEXT NOT NULL,
                        role TEXT NOT NULL CHECK (role IN ('ADMIN', 'RELIABILITY_ENGINEER', 'TECHNICIAN')),
                        is_active BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_users_role_active 
                    ON users(role, is_active);
                """)

                # 12. Persistent Alert Records for Frontend Dashboards
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS alert_records (
                        id TEXT PRIMARY KEY,
                        asset_id TEXT REFERENCES industrial_assets(asset_id) ON DELETE CASCADE,
                        severity TEXT NOT NULL,
                        report_payload TEXT NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_alerts_asset 
                    ON alert_records(asset_id);
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_incident_category
                    ON historical_incidents(asset_category);
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_incident_embedding
                    ON historical_incidents USING hnsw (narrative_embedding vector_cosine_ops);
                """)

                logger.info("Postgres core schemas and HNSW vector indices verified.")