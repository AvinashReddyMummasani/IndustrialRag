import fitz
from pathlib import Path
from src.parsers.base_parser import BaseParser
from src.core.schemas import ParsedDocumentData

class DigitalTextParser(BaseParser):

    def __init__(self, model):
        super().__init__(model)

    def can_handle(self, file_path: Path) -> bool:
        """
        Determines if the file is a native digital document.
        """
        valid_extensions = {'.pdf', '.txt', '.md'}
        return file_path.suffix.lower() in valid_extensions

    def parse(self, page: fitz.Page, page_id: str) -> ParsedDocumentData:
        # Implementation: Direct LLM JSON extraction
        print(f"[{page_id}] DigitalTextParser processing native text...")
        
        return ParsedDocumentData(
            document_id=page_id,
            route_taken="DIGITAL_TEXT",
            raw_text=page.get_text(),
            entities=[],
            relationships=[]
        )