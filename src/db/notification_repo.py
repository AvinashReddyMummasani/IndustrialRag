import uuid
import logging
from typing import List, Dict
from src.db.postgres_client import PostgresPool

logger = logging.getLogger(__name__)

class NotificationRepo:
    """Synchronous data access layer for notifications. Must be called via threads."""
    
    @staticmethod
    def save_alert_and_get_targets(asset_id: str, severity: str, report: str) -> List[Dict[str, str]]:
        """
        Executes a dual operation in a single leased connection:
        1. Persists the generated LLM alert to the database.
        2. Retrieves active users who need to be notified.
        """
        alert_id = str(uuid.uuid4())
        users_to_notify = []
        
        try:
            with PostgresPool.get_connection() as conn:
                with conn.cursor() as cur:
                    # 1. Insert the alert
                    cur.execute("""
                        INSERT INTO alert_records (id, asset_id, severity, report_payload)
                        VALUES (%s, %s, %s, %s)
                    """, (alert_id, asset_id, severity, report))
                    
                    # 2. Fetch active reliability engineers and admins
                    cur.execute("""
                        SELECT email, full_name, role
                        FROM users
                        WHERE is_active = TRUE AND role IN ('RELIABILITY_ENGINEER', 'ADMIN')
                    """)
                    
                    rows = cur.fetchall()
                    users_to_notify = [{"email": r[0], "full_name": r[1], "role": r[2]} for r in rows]
                    
        except Exception as e:
            logger.error(f"Failed to process alert persistence and user fetch: {e}")
            raise
            
        return users_to_notify