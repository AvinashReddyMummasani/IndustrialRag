import logging
from typing import Optional, Dict
from rapidfuzz import process, fuzz
from src.db.postgres_client import PostgresPool

logger = logging.getLogger(__name__)

class EntityResolver:
    """
    Normalizes unstructured natural language text into canonical Database IDs.
    """
    def __init__(self):
        self._asset_cache: Dict[str, str] = {}
        self._is_hydrated: bool = False

    async def _fetch_assets_async(self) -> Dict[str, str]:
        """Native async fetch from Postgres."""
        query = "SELECT asset_id, asset_name FROM industrial_assets WHERE current_status != 'DECOMMISSIONED';"
        cache = {}
        try:
            async with PostgresPool.get_connection() as conn:
                rows = await conn.fetch(query)
            
            for row in rows:
                asset_id = row['asset_id']
                asset_name = row['asset_name']
                cache[asset_id.lower()] = asset_id
                cache[asset_name.lower()] = asset_id
                
            logger.info(f"Entity Resolver cache loaded with {len(rows)} operational assets.")
        except Exception as e:
            logger.error(f"Failed to populate entity resolver cache: {e}")
        return cache

    async def hydrate_cache(self):
        """Asynchronously loads canonical assets to prevent blocking startup."""
        if not self._is_hydrated:
            self._asset_cache = await self._fetch_assets_async()
            self._is_hydrated = True

    async def resolve_asset_id(self, raw_mention: str, threshold: float = 85.0) -> Optional[str]:
        """
        Takes a raw user input and returns the canonical PostgreSQL primary key.
        """
        await self.hydrate_cache()
        raw_mention = raw_mention.lower().strip()
        
        # 1. Fast lookup
        if raw_mention in self._asset_cache:
            return self._asset_cache[raw_mention]

        # 2. Fuzzy Match
        choices = list(self._asset_cache.keys())
        if not choices:
            return None

        best_match = process.extractOne(
            raw_mention, 
            choices, 
            scorer=fuzz.token_set_ratio,
            score_cutoff=threshold
        )

        if best_match:
            matched_key = best_match[0]
            logger.debug(f"Resolved alias '{raw_mention}' -> '{matched_key}' (Score: {best_match[1]})")
            return self._asset_cache[matched_key]

        logger.warning(f"Entity Resolution failed for mention: '{raw_mention}'")
        return None