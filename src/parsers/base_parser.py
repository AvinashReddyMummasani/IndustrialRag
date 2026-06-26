from abc import ABC, abstractmethod
from pathlib import Path
import fitz
from src.core.schemas import ParsedDocumentData

# Intilize embedder in router

class BaseParser(ABC):
    """Abstract Base Class enforcing the parsing contract."""
    
    def __init__(self,model):
        self.embedding_model = model

    @abstractmethod
    def can_handle(self, page: fitz.Page) -> bool:
        """Determines if this specific parser is suited for the given page."""
        pass

    @abstractmethod
    def parse(self, page: fitz.Page, page_id: str) -> ParsedDocumentData:
        """Executes the extraction logic."""
        pass
    
    def chunk_text(self, text: str) -> list[str]:
        """
        Shared utility for all parsers.
        CRITICAL: This naive character splitting should be replaced with a semantic 
        tokenizer (e.g., tiktoken) or LangChain's RecursiveCharacterTextSplitter 
        to avoid severing context mid-sentence.
        """
        return [text[i:i+1000] for i in range(0, len(text), 1000)]

    def generate_embeddings(self, chunks: list[str]) -> list[list[float]]:
        """Generates dense vector embeddings using a local model."""
        if not chunks:
            return []
        
        # .encode() handles batching internally. 
        # Convert numpy arrays back to standard Python float lists for database compatibility.
        embeddings = self.embedding_model.encode(chunks)
        return [embedding.tolist() for embedding in embeddings]