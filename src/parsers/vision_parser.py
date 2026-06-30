import fitz
import logging
import base64
import re
import numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path
from typing import Union, List, Optional
from enum import Enum

from pydantic import BaseModel, Field, ValidationError
from langchain_core.messages import HumanMessage
from rapidocr_onnxruntime import RapidOCR

from src.parsers.base_parser import BaseParser
from src.core.schemas import (
    ParsedDocumentData, 
    ExtractedEntity, 
    EntityRelationship, 
    EntityType, 
    RelationType
)

logger = logging.getLogger(__name__)

# --- Local Routing Schemas ---
class ImageCategory(str, Enum):
    PID_DIAGRAM = "PID_DIAGRAM"
    SCANNED_DOCUMENT = "SCANNED_DOCUMENT"
    IRRELEVANT = "IRRELEVANT"

class ImageRoutingDecision(BaseModel):
    category: ImageCategory = Field(
        description="Classify the image. P&IDs have vector lines and equipment nodes. Scanned documents have dense paragraphs of text. If unreadable, select IRRELEVANT."
    )

class TopologicalMapping(BaseModel):
    """Root container required for LangChain structured output tool binding."""
    entities: List[ExtractedEntity] = Field(
        default_factory=list,
        description="List of all physical assets, personnel, systems, or regulations explicitly identified in the image. Do not leave this empty if relationships exist."
    )
    relationships: List[EntityRelationship] = Field(
        default_factory=list,
        description="List of all physical connections, flows, or semantic relationships extracted from the image."
    )

class VisionParser(BaseParser):
    """
    Handles unstructured image data, CAD drawings, and P&ID diagrams.
    Utilizes ONNX-based RapidOCR for baseline text and a 2-pass Vision LLM architecture to extract full semantic graphs.
    """
    
    def __init__(self, embedding_model, vision_model_client=None):
        super().__init__(embedding_model)
        self.vision_client = vision_model_client
        self.valid_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}
        
        # Load the ONNX model into memory once per worker instance
        logger.info("Initializing ONNX RapidOCR Engine in memory...")
        self.ocr_engine = RapidOCR()

    def can_handle(self, input_target: Union[Path, fitz.Page]) -> bool:
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
        if isinstance(target, Path):
            return Image.open(target)
            
        if isinstance(target, fitz.Page):
            zoom_matrix = fitz.Matrix(2.0, 2.0)
            pix = target.get_pixmap(matrix=zoom_matrix)
            img_data = pix.tobytes("png")
            return Image.open(BytesIO(img_data))
            
        raise ValueError("Unsupported target type for image extraction.")

    def _perform_baseline_ocr(self, image: Image.Image) -> str:
        try:
            # RapidOCR requires a NumPy array. Convert explicitly to RGB.
            img_array = np.array(image.convert('RGB'))
            
            # Execute ONNX inference
            result, _ = self.ocr_engine(img_array)
            
            if not result:
                return ""
            
            # Result format: list of tuples [([[x1,y1],...], "text", confidence)]
            extracted_text = " ".join([block[1] for block in result])
            return extracted_text.strip()
            
        except Exception as e:
            logger.error(f"RapidOCR execution failure: {e}")
            return ""

    def _extract_entities_via_heuristics(self, raw_text: str) -> List[ExtractedEntity]:
        entities = []
        if not raw_text:
            return entities
            
        tag_pattern = re.compile(r'\b([A-Z]{1,3}-\d{2,4}[A-Z]?)\b')
        found_tags = tag_pattern.findall(raw_text)
        
        for tag in set(found_tags):
            prefix = tag.split('-')[0]
            category = EntityType.COMPONENT if prefix == "V" else EntityType.EQUIPMENT if prefix == "P" else EntityType.UNKNOWN
            
            entities.append(
                ExtractedEntity(
                    entity_id=tag,
                    entity_type=category,
                    properties={"raw_mention": tag, "source": "local_ocr_heuristic"},
                    confidence=0.9
                )
            )
            
        return entities

    def _extract_vision_graph_data(self, pil_image: Image.Image, document_id: str) -> Optional[TopologicalMapping]:
        if not self.vision_client:
            logger.warning(f"[{document_id}] No vision client injected. Skipping topology mapping.")
            return None

        try:
            # 1. Rasterize payload
            buffered = BytesIO()
            if pil_image.mode in ("RGBA", "P"):
                pil_image = pil_image.convert("RGB")
                
            pil_image.save(buffered, format="JPEG", quality=85)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            image_payload = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}

            # 2. Phase 1: High-Speed Classification Routing
            router_prompt = (
                "You are an industrial image classifier. Look at this image and categorize it strictly. "
                "Output PID_DIAGRAM if it shows pipes, valves, and topological connections. "
                "Output SCANNED_DOCUMENT if it is mostly text, permits, or manuals. "
                "Output IRRELEVANT if it is a blank page or unreadable."
            )
            
            route_msg = HumanMessage(content=[{"type": "text", "text": router_prompt}, image_payload])
            router_llm = self.vision_client.with_structured_output(ImageRoutingDecision)
            decision: ImageRoutingDecision = router_llm.invoke([route_msg])
            
            logger.info(f"[{document_id}] Vision Router classified image as: {decision.category.value}")

            if decision.category == ImageCategory.IRRELEVANT:
                logger.info(f"[{document_id}] Image deemed irrelevant. Bypassing heavy extraction.")
                return None

            # 3. Phase 2: Targeted Extraction Setup
            valid_relations = [r.value for r in RelationType]
            
            if decision.category == ImageCategory.PID_DIAGRAM:
                extraction_prompt = (
                    "You are an expert industrial P&ID drafter. Extract the entire topological network from this diagram. "
                    "Explicitly define ALL equipment tags, components, and nodes in the 'entities' array. "
                    "Explicitly define the physical pipe/wire connections in the 'relationships' array. "
                    f"CRITICAL CONSTRAINT: 'relation_type' MUST strictly map to one of these values: {', '.join(valid_relations)}. "
                    "Do not generate conversational text."
                )
            else:
                extraction_prompt = (
                    "You are an industrial data extractor. Read this scanned safety/administrative document. "
                    "1. Extract ALL mentioned systems, personnel, parameters, permits, and equipment as 'entities'. "
                    "2. ANOMALY DETECTION: If there are handwritten notes or text indicating failures, glitches, or injuries (e.g., 'heat stress', 'sensor glitch'), explicitly extract them as INCIDENT entities. "
                    "3. Define their functional or administrative connections in the 'relationships' array. "
                    "4. GRAPH MATH CONSTRAINT: Pay strict attention to edge directionality. Equipment POSSESSES parameters (e.g., source: Equipment -> target: Parameter). Personnel PERFORM actions on equipment. Do not reverse source_id and target_id. "
                    f"CRITICAL CONSTRAINT: 'relation_type' MUST strictly map to one of these values: {', '.join(valid_relations)}. "
                    "Do not generate conversational text."
                )

            # 4. Phase 3: Heavy Structured Extraction (Nodes & Edges)
            extraction_msg = HumanMessage(content=[{"type": "text", "text": extraction_prompt}, image_payload])
            extractor_llm = self.vision_client.with_structured_output(TopologicalMapping)
            result: TopologicalMapping = extractor_llm.invoke([extraction_msg])
            
            return result

        except ValidationError as e:
            logger.error(f"[{document_id}] Vision model hallucinated outside schema constraints:\n{e}")
            return None
        except Exception as e:
            logger.error(f"[{document_id}] Vision API network or execution failure: {e}")
            return None

    def parse(self, input_target: Union[Path, fitz.Page], document_id: str) -> ParsedDocumentData:
        logger.info(f"[{document_id}] VisionParser initiated...")
        
        try:
            # 1. Rasterize
            pil_image = self._extract_image_from_target(input_target)
            
            # 2. Extract Baseline Local Nodes (ONNX OCR + Heuristics)
            raw_text = self._perform_baseline_ocr(pil_image)
            heuristic_entities = self._extract_entities_via_heuristics(raw_text)
            
            # 3. Extract Deep Graph (Vision API 2-Pass Router)
            vision_result = self._extract_vision_graph_data(pil_image, document_id)
            
            vision_entities = vision_result.entities if vision_result else []
            vision_relationships = vision_result.relationships if vision_result else []
            
            # 4. Merge & Deduplicate (Preventing the Orphaned Node problem)
            combined_entities = heuristic_entities + vision_entities
            
            unique_entities = {e.entity_id: e for e in combined_entities}.values()
            unique_relations = {f"{r.source_id}-{r.relation_type}-{r.target_id}": r for r in vision_relationships}.values()
            
            logger.info(f"[{document_id}] Extraction complete. Unique Nodes: {len(unique_entities)}, Unique Edges: {len(unique_relations)}")
            
            # 5. Embeddings
            embeddings = []
            chunks = self.chunk_text(raw_text) if hasattr(self, 'chunk_text') else []
            
            if hasattr(self, 'embedding_model') and self.embedding_model and chunks:
                try:
                    embeddings = self.embedding_model.encode(chunks).tolist()
                except Exception as e:
                    logger.error(f"[{document_id}] Failed to generate text embeddings: {e}")
                    
            return ParsedDocumentData(
                document_id=document_id,
                route_taken="VISION_TOPOLOGICAL",
                raw_text=raw_text,
                text_chunks=chunks,
                embeddings=embeddings,
                entities=list(unique_entities),
                relationships=list(unique_relations)
            )
            
        except Exception as e:
            logger.error(f"[{document_id}] Critical failure in VisionParser: {e}")
            raise RuntimeError(f"Vision parsing failed for document {document_id}") from e