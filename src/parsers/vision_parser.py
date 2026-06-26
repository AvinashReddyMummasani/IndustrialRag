import fitz
import logging
import pytesseract
import base64
import json
import re
from PIL import Image
from io import BytesIO
from pathlib import Path
from typing import Union, List, Dict, Any

from langchain_core.messages import HumanMessage

from src.parsers.base_parser import BaseParser
from src.core.schemas import ParsedDocumentData, ExtractedEntity

logger = logging.getLogger(__name__)

class VisionParser(BaseParser):
    """
    Handles unstructured image data, CAD drawings, and P&ID diagrams.
    Utilizes local OCR for entity extraction and Groq Vision for topological mapping.
    """
    
    def __init__(self, embedding_model, vision_model_client=None):
        super().__init__(embedding_model)
        self.vision_client = vision_model_client
        self.valid_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}

    def can_handle(self, input_target: Union[Path, fitz.Page]) -> bool:
        """Evaluates both raw file paths and pre-loaded PDF pages for vision suitability."""
        if isinstance(input_target, Path):
            return input_target.suffix.lower() in self.valid_extensions
            
        if isinstance(input_target, fitz.Page):
            try:
                # Heuristic: Complex engineering diagrams have high vector drawing counts.
                return len(input_target.get_drawings()) > 150
            except Exception as e:
                logger.warning(f"Failed to evaluate drawing heuristic: {e}")
                return False
                
        return False

    def _extract_image_from_target(self, target: Union[Path, fitz.Page]) -> Image.Image:
        """Standardizes inputs into a PIL Image for OCR and Vision processing."""
        if isinstance(target, Path):
            return Image.open(target)
            
        if isinstance(target, fitz.Page):
            # 144 DPI scaling for OCR clarity
            zoom_matrix = fitz.Matrix(2.0, 2.0)
            pix = target.get_pixmap(matrix=zoom_matrix)
            img_data = pix.tobytes("png")
            return Image.open(BytesIO(img_data))
            
        raise ValueError("Unsupported target type for image extraction.")

    def _perform_baseline_ocr(self, image: Image.Image) -> str:
        """Executes local OCR to capture explicit text (Equipment Tags, Line Numbers)."""
        try:
            # PSM 11 is optimized for sparse text scattered across a diagram
            custom_config = r'--oem 3 --psm 11'
            text = pytesseract.image_to_string(image, config=custom_config)
            return text.strip()
        except Exception as e:
            logger.error(f"Tesseract OCR failure: {e}")
            return ""

    def _extract_entities_via_heuristics(self, raw_text: str) -> List[ExtractedEntity]:
        """Extracts standard industrial tags using regex on the OCR output."""
        entities = []
        # Standard industrial tag pattern (e.g., P-101, V-200A, TK-99)
        tag_pattern = re.compile(r'\b([A-Z]{1,3}-\d{2,4}[A-Z]?)\b')
        found_tags = tag_pattern.findall(raw_text)
        
        for tag in set(found_tags):
            prefix = tag.split('-')[0]
            entity_type = "VALVE" if prefix == "V" else "PUMP" if prefix == "P" else "EQUIPMENT"
            entities.append(ExtractedEntity(entity_id=tag, entity_type=entity_type))
            
        return entities

    def _extract_topological_relationships(self, pil_image: Image.Image, document_id: str) -> List[Dict[str, Any]]:
        """
        Pipes the diagram to the Groq Vision model to map physical connections.
        Forces the model to return structured JSON.
        """
        if not self.vision_client:
            logger.warning(f"[{document_id}] No vision client injected. Skipping topology mapping.")
            return []

        try:
            # Compress and encode the image for API transit
            buffered = BytesIO()
            # Convert to RGB to prevent alpha channel errors during JPEG compression
            if pil_image.mode in ("RGBA", "P"):
                pil_image = pil_image.convert("RGB")
                
            pil_image.save(buffered, format="JPEG", quality=85)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            prompt_text = (
                "You are an expert industrial P&ID drafter. Analyze this engineering diagram. "
                "Identify the physical connections between equipment tags (e.g., P-101 connects to V-204). "
                "Return ONLY a valid JSON list of objects with 'source', 'target', and 'relation' keys. "
                "The relation should describe the connection type (e.g., 'PIPED_TO', 'CONTROLS'). "
                "Do not include markdown blocks, just the raw JSON array."
            )

            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
                ]
            )
            
            response = self.vision_client.invoke([message])
            raw_output = response.content.strip()
            
            # Clean potential markdown formatting from LLM output
            if raw_output.startswith("```json"):
                raw_output = raw_output.replace("```json", "").replace("```", "").strip()
            elif raw_output.startswith("```"):
                raw_output = raw_output.replace("```", "").strip()

            relationships = json.loads(raw_output)
            
            if not isinstance(relationships, list):
                logger.error(f"[{document_id}] Vision model returned non-list JSON.")
                return []
                
            return relationships

        except json.JSONDecodeError as e:
            logger.error(f"[{document_id}] Failed to parse Vision model JSON output: {e}\nRaw output: {raw_output}")
            return []
        except Exception as e:
            logger.error(f"[{document_id}] Vision API network/execution failure: {e}")
            return []

    def parse(self, input_target: Union[Path, fitz.Page], document_id: str) -> ParsedDocumentData:
        """Orchestrates the full rasterization, OCR, and VLM pipeline."""
        logger.info(f"[{document_id}] VisionParser initiated...")
        
        try:
            # 1. Rasterize
            pil_image = self._extract_image_from_target(input_target)
            
            # 2. Local OCR Extraction
            raw_text = self._perform_baseline_ocr(pil_image)
            entities = self._extract_entities_via_heuristics(raw_text)
            
            # 3. Vision API Topology Mapping
            relationships = self._extract_topological_relationships(pil_image, document_id)
            
            logger.info(f"[{document_id}] Vision extraction complete. Entities: {len(entities)}, Relationships: {len(relationships)}")
            
            return ParsedDocumentData(
                document_id=document_id,
                route_taken="VISION_TOPOLOGICAL",
                raw_text=raw_text,
                entities=entities,
                relationships=relationships
            )
            
        except Exception as e:
            logger.error(f"[{document_id}] Critical failure in VisionParser: {e}")
            raise RuntimeError(f"Vision parsing failed for document {document_id}") from e