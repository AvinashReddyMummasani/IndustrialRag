from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, List, Any
import fitz
import logging
from src.core.schemas import ParsedDocumentData

logger = logging.getLogger(__name__)

class BaseParser(ABC):
    """Abstract Base Class enforcing the parsing contract across all strategies."""
    
    def __init__(self, embedding_model: Any = None):
        # Type-hinted as Any to support SentenceTransformers, HF pipelines, or API wrappers.
        self.embedding_model = embedding_model

    @abstractmethod
    def can_handle(self, input_target: Union[Path, fitz.Page]) -> bool:
        """
        Determines if this specific parser is suited for the given target.
        Must accept both raw file paths and PyMuPDF page objects to support 
        both digital text and raw image processing pipelines.
        """
        pass

    @abstractmethod
    def parse(self, input_target: Union[Path, fitz.Page], document_id: str) -> ParsedDocumentData:
        """
        Executes the extraction logic.
        Standardized signature: (input_target, document_id) prevents keyword argument 
        mismatches when dynamically routed by the orchestrator.
        """
        pass
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Sliding window chunking strategy. 
        
        Replaces the naive sequential slice. Context overlap is strictly necessary 
        for semantic retrieval (RAG) to prevent severing relationships mid-sentence.
        For production, override this in concrete classes with a token-aware 
        splitter (e.g., tiktoken) if token limits are strict.
        """
        if not text:
            return []
            
        chunks = []
        # Step size is (chunk_size - overlap) to ensure continuity
        step = max(1, chunk_size - overlap) 
        
        for i in range(0, len(text), step):
            chunks.append(text[i:i + chunk_size])
            
        return chunks

    def generate_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """Generates dense vector embeddings using the injected local model."""
        if not chunks:
            return []
            
        if not self.embedding_model:
            logger.warning("No embedding model injected. Skipping vector generation.")
            return []
        
        try:
            # .encode() handles batching internally for standard sentence-transformers.
            embeddings = self.embedding_model.encode(chunks)
            # Standardize output to Python float lists for vector DB compatibility
            return [embedding.tolist() for embedding in embeddings]
        except Exception as e:
            logger.error(f"Vector embedding generation failed: {e}")
            # Propagate the failure; silent embedding failures corrupt downstream graph mapping
            raise RuntimeError(f"Embedding failure: {e}") from e