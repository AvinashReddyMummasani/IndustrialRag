import logging
from psycopg2.extras import execute_values
from src.db.postgres_client import PostgresPool

logger = logging.getLogger(__name__)

class TimescaleManager:
    """Manages high-velocity time-series sensor data using TimescaleDB hypertables."""

    @classmethod
    def bootstrap_telemetry_schema(cls):
        """Creates the time-series tables and converts them to partitioned hypertables."""
        schema_query = """
            CREATE TABLE IF NOT EXISTS asset_telemetry (
                measured_at TIMESTAMPTZ NOT NULL,
                asset_id TEXT NOT NULL REFERENCES industrial_assets(asset_id),
                metric_name TEXT NOT NULL,
                metric_value DOUBLE PRECISION NOT NULL
            );
            
            -- Convert to TimescaleDB hypertable (partitioned by time)
            SELECT create_hypertable('asset_telemetry', 'measured_at', if_not_exists => TRUE);
            
            -- Composite index optimized for agentic point-in-time lookups
            CREATE INDEX IF NOT EXISTS ix_asset_metric_time 
            ON asset_telemetry (asset_id, metric_name, measured_at DESC);
        """
        try:
            with PostgresPool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(schema_query)
            logger.info("TimescaleDB telemetry hypertables bootstrapped.")
        except Exception as e:
            logger.error(f"Failed to bootstrap TimescaleDB schema: {e}")
            # If TimescaleDB is not installed on the Postgres instance, this will fail.

    @classmethod
    def execute_bulk_insert(cls, records: list[tuple]) -> int:
        """
        Executes high-throughput inserts using execute_values.
        records format: [(timestamp, asset_id, metric_name, value), ...]
        """
        if not records:
            return 0

        insert_query = """
            INSERT INTO asset_telemetry (measured_at, asset_id, metric_name, metric_value)
            VALUES %s
        """
        try:
            with PostgresPool.get_connection() as conn:
                with conn.cursor() as cur:
                    # execute_values is significantly faster than executemany for large batches
                    execute_values(cur, insert_query, records, page_size=1000)
            return len(records)
        except Exception as e:
            logger.error(f"Timescale bulk insert failed: {e}")
            raise RuntimeError("Telemetry ingestion transaction failed.")