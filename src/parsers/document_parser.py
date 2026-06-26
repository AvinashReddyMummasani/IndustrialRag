import fitz
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.parsers.base_parser import BaseParser
from src.core.schemas import ParsedDocumentData

class DocumentOrchestrator:
    """Routes PDF pages to the appropriate parser strategy."""
    
    def __init__(self, parsers: List[BaseParser], max_workers: int = 4):
        # The orchestrator depends on abstractions (BaseParser), satisfying DIP.
        self.parsers = parsers
        self.max_workers = max_workers

    def _route_and_parse(self, pdf_path: Path, page_num: int) -> ParsedDocumentData:
        """Finds the first capable parser and executes it."""
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        try:
            # Chain of Responsibility / Strategy Routing
            for parser in self.parsers:
                if parser.can_handle(page):
                    return parser.parse(page, page_id=f"{pdf_path.stem}_p{page_num}")
            
            raise ValueError(f"No registered parser can handle page {page_num}")
        finally:
            doc.close()

    def parse_document(self, file_path: Path) -> List[ParsedDocumentData]:
        """Orchestrates concurrent parsing of all pages."""
        doc = fitz.open(file_path)
        total_pages = len(doc)
        doc.close()
        
        results = [None] * total_pages

        # Creates max_workers threads and parse pages concurrently
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._route_and_parse, file_path, i): i 
                for i in range(total_pages)
            }
            
            for future in as_completed(futures):
                p_num = futures[future]
                try:
                    results[p_num] = future.result()
                except Exception as e:
                    print(f"CRITICAL: Failed to parse page {p_num}: {e}")
                    
        # Filter out None values from failed pages before returning
        return [res for res in results if res is not None]