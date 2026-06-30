import fitz
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.parsers.base_parser import BaseParser
from src.core.schemas import ParsedDocumentData
import logging

logger = logging.getLogger(__name__)

class DocumentOrchestrator:
    """Routes PDF pages to the appropriate parser strategy."""
    
    def __init__(self, parsers: List[BaseParser], max_workers: int = 4):
        # The orchestrator depends on abstractions (BaseParser), satisfying DIP.
        self.parsers = parsers
        self.max_workers = max_workers

    def _route_and_parse(self, pdf_path: Path, page_num: int, file_id: str) -> ParsedDocumentData:
            """Evaluates and routes a single page to the appropriate parser."""
            page_id = f"{file_id}_p{page_num}"
            
            # Open per thread to guarantee memory safety during concurrent reads
            doc = fitz.open(pdf_path)
            try:
                page = doc[page_num]
                
                # Strategy Routing
                for parser in self.parsers:
                    if parser.can_handle(page):
                        # Enforce positional arguments to avoid kwarg mismatches
                        return parser.parse(page, page_id)
                
                # Explicit degradation if no parser claims the page
                logger.warning(f"[{page_id}] Unroutable page. No matching parser.")
                return self._build_fallback_state(page_id, "UNROUTABLE")
                
            except Exception as e:
                logger.error(f"[{page_id}] CRITICAL extraction failure: {e}")
                return self._build_fallback_state(page_id, "FAILED")
                
            finally:
                doc.close()

    def parse_document(self, file_path: Path,file_id :str) -> List[ParsedDocumentData]:
        """Orchestrates concurrent parsing of all pages."""
        doc = fitz.open(file_path)
        total_pages = len(doc)
        doc.close()
        
        results = [None] * total_pages
        
        if total_pages == 0:
            raise ValueError(f"Document {file_id} is empty (0 pages).")

        # Creates max_workers threads and parse pages concurrently
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._route_and_parse, file_path, i, file_id): i 
                for i in range(total_pages)
            }
            
            for future in as_completed(futures):
                p_num = futures[future]
                try:
                    results[p_num] = future.result()
                except Exception as e:
                    print(f"CRITICAL: Failed to parse page {p_num}: {e}")
               
        aggregated_chunks = []
        aggregated_embeddings = []
        aggregated_entities = []
        aggregated_relationships = []
        successful_pages = 0
        used_parsers = []
        
        for page_data in results:
            if page_data is None or page_data.route_taken in ["FAILED", "UNROUTABLE"]:
                continue
                
            successful_pages += 1
            aggregated_chunks.extend(page_data.text_chunks)
            used_parsers.append(page_data.route_taken)
            
            if page_data.embeddings:
                aggregated_embeddings.extend(page_data.embeddings)
                
            if page_data.entities:
                aggregated_entities.extend(page_data.entities)
                
            if page_data.relationships:
                aggregated_relationships.extend(page_data.relationships)

        if successful_pages == 0:
            final_route = "ORCHESTRATED_BATCH_TOTAL_FAILURE"
            raise ValueError(f"Document {file_id} is empty (0 pages).")
        elif successful_pages == total_pages:
            final_route = "ORCHESTRATED_BATCH_COMPLETE, " + ", ".join(used_parsers)
        else:
            final_route = f"ORCHESTRATED_BATCH_PARTIAL_{successful_pages}_OF_{total_pages}"

        # Return a single, unified document record
        # for large pdfs it breaks
        return ParsedDocumentData(
            document_id=file_id,
            route_taken=final_route,
            raw_text="", 
            text_chunks=aggregated_chunks,
            embeddings=aggregated_embeddings,
            entities=aggregated_entities,
            relationships=aggregated_relationships
        )