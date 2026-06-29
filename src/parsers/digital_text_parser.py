import os
import logging
from typing import List
from pathlib import Path
import fitz
from groq import Groq
import instructor
from pydantic import BaseModel, ValidationError

# Assuming schemas are importable from your core module
from src.core.schemas import (
    ParsedDocumentData,
    ExtractedEntity,
    EntityRelationship
)
from src.parsers.base_parser import BaseParser

logger = logging.getLogger(__name__)

class ExtractionPayload(BaseModel):
    """
    Root container for the LLM to target. We do not ask the LLM to generate 
    the entire ParsedDocumentData schema (which includes embeddings/chunks), 
    only the graph data it is capable of extracting.
    """
    entities: List[ExtractedEntity]
    relationships: List[EntityRelationship]

class DigitalTextParser(BaseParser):
    def __init__(self, embedding_model = None,llm=None):
        """
        Initializes parser. Expects explicit definition of both the generative LLM
        and the embedding model to prevent namespace collisions.
        """
        # Pass the primary execution model to the base class if required
        super().__init__(embedding_model) 
        
        self.llm_model = llm
        self.embedding_model = embedding_model
        
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("CRITICAL: GROQ_API_KEY environment variable is missing.")
            
        # Wrap Groq client with instructor for native Pydantic validation and retries
        raw_client = Groq(api_key=api_key)
        self.client = instructor.from_groq(raw_client, mode=instructor.Mode.TOOLS)

        self.system_prompt = (
            "You are a strict technical data extraction system mapping engineering documentation to a graph database. "
            "Extract entities and relationships from the provided text. "
            "You must adhere strictly to the allowed taxonomies for entity_type and relation_type."
        )

    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == '.pdf'
        

    def parse(self, page: fitz.Page, page_id: str) -> ParsedDocumentData:
        raw_text = page.get_text("text").strip()
        
        if not raw_text:
            logger.warning(f"[{page_id}] Empty page. Bypassing extraction.")
            return ParsedDocumentData(
                document_id=page_id,
                route_taken="DIGITAL_TEXT_EMPTY",
                raw_text="",
                text_chunks=[],
                entities=[],
                relationships=[]
            )

        text_chunks = self.chunk_text(raw_text)
        embeddings = self.generate_embeddings(text_chunks)

        # 2. Structured LLM Extraction
        try:
            # Instructor handles the JSON schema generation, passing it as a tool,
            # and validating the response against ExtractionPayload.
            extracted_data: ExtractionPayload = self.client.chat.completions.create(
                model=self.llm_model,
                response_model=ExtractionPayload,
                max_retries=2,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": raw_text}
                ],
                temperature=0.0,
                timeout=15.0
            )
            
            return ParsedDocumentData(
                document_id=page_id,
                route_taken="DIGITAL_TEXT_STRUCTURED",
                raw_text=raw_text,
                text_chunks=text_chunks,
                embeddings=embeddings,
                entities=extracted_data.entities,
                relationships=extracted_data.relationships
            )
            
        except ValidationError as e:
            logger.error(f"[{page_id}] Pydantic validation failed after retries: {e}")
        except Exception as e:
            logger.error(f"[{page_id}] API extraction failed: {e}")
        
        # 3. Graceful Degradation
        return ParsedDocumentData(
            document_id=page_id,
            route_taken="DIGITAL_TEXT_FAILED_EXTRACTION",
            raw_text=raw_text,
            text_chunks=text_chunks,
            embeddings=embeddings,
            entities=[],
            relationships=[]
        )