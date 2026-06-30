import io
import fitz
import logging
import pytesseract
from PIL import Image
from pathlib import Path
from typing import Union

from src.parsers.base_parser import BaseParser
from src.core.schemas import ParsedDocumentData

logger = logging.getLogger(__name__)

class SemanticParser(BaseParser):
    """
    Handles scanned documents, forms, and images containing dense text.
    Executes Optical/Handwritten Character Recognition (OCR/HTR).
    """
    
    def __init__(self, embedding_model, htr_model=None):
        """
        Args:
            embedding_model: The SentenceTransformer model for vectorizing extracted text.
            htr_model: Optional custom HuggingFace pipeline (e.g., TrOCR) for handwriting.
        """
        super().__init__(embedding_model)
        self.htr_model = htr_model
        self.valid_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}

    def can_handle(self, input_target: Union[Path, fitz.Page]) -> bool:
        """
        Determines if the target requires Semantic/OCR processing.
        Accepts both raw image paths and PDF pages.
        """
        if isinstance(input_target, Path):
            return input_target.suffix.lower() in self.valid_extensions
            
        if isinstance(input_target, fitz.Page):
            try:
                text = input_target.get_text("text").strip()
                
                if len(text) > 200: 
                    return False
                    
                return True
                
            except Exception as e:
                logger.error(f"Failed to evaluate page for SemanticParser: {e}")
                return False
                
        return False

    def _extract_image(self, target: Union[Path, fitz.Page]) -> Image.Image:
        """Standardizes inputs into a PIL Image for OCR processing."""
        if isinstance(target, Path):
            return Image.open(target)
            
        if isinstance(target, fitz.Page):
            # Matrix scaling (2.5 ~ 180 DPI) is generally more memory-stable than dpi=300 
            # while providing sufficient resolution for standard OCR.
            zoom_matrix = fitz.Matrix(2.5, 2.5)
            pix = target.get_pixmap(matrix=zoom_matrix)
            img_data = pix.tobytes("png")
            return Image.open(io.BytesIO(img_data))
            
        raise ValueError("Unsupported target type for Semantic Parser.")

    def parse(self, input_target: Union[Path, fitz.Page], document_id: str) -> ParsedDocumentData:
        logger.info(f"[{document_id}] SemanticParser executing OCR/HTR pipeline...")
        
        try:
            # 1. Rasterize / Load Image
            img = self._extract_image(input_target)
            
            # 2. Execute Extraction
            if self.htr_model:
                # Inject custom vision model/pipeline (Later Intilize it)
                raw_extracted_text = self.htr_model(img)
                if isinstance(raw_extracted_text, list) and len(raw_extracted_text) > 0:
                    raw_extracted_text = raw_extracted_text[0].get('generated_text', '')
            else:
                # Default to system Tesseract
                
                custom_config = r'--oem 3 --psm 3'
                raw_extracted_text = pytesseract.image_to_string(img, config=custom_config)
                
            raw_extracted_text = raw_extracted_text.strip()
                
        except Exception as e:
            logger.error(f"[{document_id}] OCR pipeline failed: {e}")
            raise RuntimeError(f"Semantic parsing failed for {document_id}") from e
        
        # 3. Standardized chunking and embedding
        chunks = self.chunk_text(raw_extracted_text) if hasattr(self, 'chunk_text') and raw_extracted_text else [raw_extracted_text]
        embeddings = []
        
        # Safely execute embeddings utilizing the BaseParser's stored model
        if hasattr(self, 'embedding_model') and self.embedding_model and chunks:
            try:
                
                embeddings = self.embedding_model.encode(chunks).tolist()
            except Exception as e:
                logger.error(f"[{document_id}] Failed to generate text embeddings: {e}")

        return ParsedDocumentData(
            document_id=document_id,
            route_taken="SEMANTIC_HTR",
            raw_text=raw_extracted_text,
            text_chunks=chunks,
            embeddings=embeddings,
            entities=[],
            relationships=[]
        )