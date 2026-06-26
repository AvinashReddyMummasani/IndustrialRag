import os
import json
import logging
import fitz
from pathlib import Path
from groq import Groq
from pydantic import ValidationError # Assuming your schemas use Pydantic
from src.parsers.base_parser import BaseParser
from src.core.schemas import ParsedDocumentData

# Configure module-level logger
logger = logging.getLogger(__name__)

class DigitalTextParser(BaseParser):
    def __init__(self, model: str = "llama3-70b-8192"):
        """
        Initializes the parser and Groq client. Fails fast if the environment is misconfigured.
        """
        super().__init__(model)
        
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("CRITICAL: GROQ_API_KEY environment variable is missing.")
            
        self.client = Groq(api_key=api_key)
        self.model = model

        # Define schema explicitly for the LLM to enforce structure
        self.system_prompt = (
            "You are a strict data extraction system. Extract entities and relationships from the provided text. "
            "You must output ONLY valid JSON matching this exact schema: "
            "{\"entities\": [{\"id\": \"str\", \"type\": \"str\", \"value\": \"str\"}], "
            "\"relationships\": [{\"source_id\": \"str\", \"target_id\": \"str\", \"relation\": \"str\"}]}"
        )

    def can_handle(self, file_path: Path) -> bool:
        """
        Strictly limits fitz (PyMuPDF) processing to PDFs.
        Text and Markdown should be handled by a NativeTextParser to avoid abstraction overhead.
        """
        return file_path.suffix.lower() == '.pdf'

    def parse(self, page: fitz.Page, page_id: str) -> ParsedDocumentData:
        # 1. Extraction & Pre-processing
        # 'text' flag is explicit. strip() removes trailing/leading whitespace to save tokens.
        raw_text = page.get_text("text").strip()
        
        # 2. Short-Circuit for Empty Pages
        if not raw_text:
            logger.warning(f"[{page_id}] Page is empty or image-only. Bypassing LLM.")
            return ParsedDocumentData(
                document_id=page_id,
                route_taken="DIGITAL_TEXT_EMPTY",
                raw_text="",
                entities=[],
                relationships=[]
            )

        # 3. LLM Extraction via Groq
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": raw_text}
                ],
                response_format={"type": "json_object"}, # Forces guaranteed JSON output structure
                temperature=0.0, # Deterministic outputs are mandatory for data extraction
                timeout=15.0     # Fail fast. Do not let dead connections hang the pipeline.
            )
            
            # 4. Parsing and Validation
            extracted_data = json.loads(response.choices[0].message.content)
            
            return ParsedDocumentData(
                document_id=page_id,
                route_taken="DIGITAL_TEXT_GROQ",
                raw_text=raw_text,
                entities=extracted_data.get("entities", []),
                relationships=extracted_data.get("relationships", [])
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"[{page_id}] Groq returned malformed JSON: {e}")
        except Exception as e:
            # Catches groq.APIConnectionError, groq.RateLimitError, timeouts, etc.
            logger.error(f"[{page_id}] Groq extraction failed: {e}")
        
        # 5. Graceful Degradation
        # If extraction fails, return the raw text so downstream tasks can still attempt keyword matching or fallback logic.
        return ParsedDocumentData(
            document_id=page_id,
            route_taken="DIGITAL_TEXT_FAILED_EXTRACTION",
            raw_text=raw_text,
            entities=[],
            relationships=[]
        )