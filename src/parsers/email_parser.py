import os
import email
import logging
import uuid
from email.utils import parseaddr
from pathlib import Path
from bs4 import BeautifulSoup
from typing import Tuple, List

from groq import Groq
import instructor
from pydantic import BaseModel, Field

from src.parsers.base_parser import BaseParser
from src.core.schemas import (
    ParsedDocumentData, 
    ExtractedEntity, 
    EntityRelationship,
    EntityType,
    RelationType
)

logger = logging.getLogger(__name__)

class EmailGraphExtraction(BaseModel):
    """Root schema for instructor to enforce structured graph extraction."""
    entities: List[ExtractedEntity] = Field(
        default_factory=list,
        description="List of entities found in the email body."
    )
    relationships: List[EntityRelationship] = Field(
        default_factory=list,
        description="Topological or semantic connections, including those linking the sender/receiver to equipment."
    )

class EmailParser(BaseParser):
    def __init__(self, extract_dir: Path = Path("./temp_uploads"),model=None):
        super().__init__(model)
        
        raw_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        # To make model return pydantic object instead of string
        self.client = instructor.from_groq(raw_client, mode=instructor.Mode.TOOLS)
        
        # Guardrails
        self.max_body_chars = 10000 
        self.extraction_model = "llama3-8b-8192"
        
        self.extract_dir = extract_dir # Extract into temp uploads
        self.extract_dir.mkdir(parents=True, exist_ok=True)
        
    def can_handle(self, file_path: Path) -> bool:
        """
        Satisfies the BaseParser contract. 
        Determines if the file extension matches expected email formats.
        """
        return file_path.suffix.lower() in ['.eml', '.msg']
    
    def _extract_clean_body(self, msg: email.message.Message) -> str:
        """Safely extracts and cleans text from multipart MIME structures."""
        body_parts = []
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                disposition = str(part.get("Content-Disposition"))

                if "attachment" in disposition:
                    continue

                try:
                    payload = part.get_payload(decode=True)
                    if not payload:
                        continue
                        
                    charset = part.get_content_charset() or 'utf-8'
                    decoded_text = payload.decode(charset, errors='ignore')

                    if content_type == "text/plain":
                        body_parts.append(decoded_text)
                    elif content_type == "text/html":
                        soup = BeautifulSoup(decoded_text, "html.parser")
                        body_parts.append(soup.get_text(separator="\n", strip=True))
                except Exception as e:
                    logger.warning(f"Failed to decode email part: {e}")
        else:
            try:
                payload = msg.get_payload(decode=True)
                charset = msg.get_content_charset() or 'utf-8'
                text = payload.decode(charset, errors='ignore')
                if msg.get_content_type() == "text/html":
                    text = BeautifulSoup(text, "html.parser").get_text(separator="\n", strip=True)
                body_parts.append(text)
            except Exception:
                pass

        return "\n".join(body_parts).strip()

    def _extract_body_graph(self, text: str, sender_id: str, receiver_id: str, file_id: str) -> Tuple[List[ExtractedEntity], List[EntityRelationship]]:
        """Invokes the LLM via Instructor to extract a unified graph."""
        if not text:
            return [], []

        truncated_text = text[:self.max_body_chars]
        
        system_prompt = f"""
        You are a strict industrial knowledge graph extraction system. 
        Your task is to parse email communications and extract entities and relationships matching a specific taxonomy.
        
        CRITICAL INSTRUCTIONS:
        1. You have been provided the Sender ID ({sender_id}) and Receiver ID ({receiver_id}). 
        2. You MUST establish relationships between the Sender/Receiver and the equipment/parameters discussed in the email.
           - Example: If the sender reports fixing a pump, create a MAINTAINS relationship from the Sender ID to the Pump ID.
           - Example: If the receiver is asked to monitor a parameter, create a GOVERNS relationship from the Receiver ID to the Parameter ID.
        3. Do not create duplicate entities for the Sender or Receiver. Use the exact IDs provided.
        4. Normalize all extracted entity IDs to UPPERCASE (e.g., 'P-101', 'TEMPERATURE_SENSOR').
        5. Strictly adhere to the provided Enums for EntityType and RelationType. Do not invent new types.
        """

        try:
            extraction: EmailGraphExtraction = self.client.chat.completions.create(
                model=self.extraction_model,
                response_model=EmailGraphExtraction,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Email Body:\n{truncated_text}"}
                ],
                temperature=0.0
            )
            
            # Confidence penalty for probabilistic generation Later Can be calculated by llm
            for entity in extraction.entities:
                entity.confidence = 0.85 

            return extraction.entities, extraction.relationships

        except Exception as e:
            logger.error(f"[{file_id}] LLM graph extraction failed: {e}")
            return [], []

    def parse(self, file_path: Path, file_id: str) -> Tuple[ParsedDocumentData, List[Path]]:
        logger.info(f"[{file_id}] Executing EmailParser...")
        
        # Parse the email
        with open(file_path, 'rb') as f:
            msg = email.message_from_binary_file(f)

        # 1. Deterministic Metadata Extraction
        subject = msg.get("Subject", "No Subject")
        sender_name, sender_email = parseaddr(msg.get("From", ""))
        receiver_name, receiver_email = parseaddr(msg.get("To", ""))
        date = msg.get("Date", "")

        # 2. Extract Attachments to Disk
        extracted_attachments = []
        if msg.is_multipart():
            for part in msg.walk():
                disposition = str(part.get("Content-Disposition"))
                if "attachment" in disposition:
                    filename = part.get_filename()
                    if not filename:
                        filename = f"unnamed_attachment_{uuid.uuid4().hex[:8]}"
                    
                    safe_filename = "".join(c for c in filename if c.isalnum() or c in " ._-")
                    attachment_path = self.extract_dir / f"{file_id}_{safe_filename}"
                    
                    with open(attachment_path, "wb") as af:
                        af.write(part.get_payload(decode=True))
                    extracted_attachments.append(attachment_path)

        raw_body = self._extract_clean_body(msg)
        
        # Email data without Attachements
        full_text = (
            f"Subject: {subject}\n"
            f"From: {sender_name} <{sender_email}>\n"
            f"To: {receiver_name} <{receiver_email}>\n"
            f"Date: {date}\n\n"
            f"{raw_body}"
        )

        # 3. Build Deterministic Graph Nodes
        entities = []
        relationships = []
        
        sender_id = sender_email.upper() if sender_email else "UNKNOWN_SENDER"
        receiver_id = receiver_email.upper() if receiver_email else "UNKNOWN_RECEIVER"

        if sender_id != "UNKNOWN_SENDER":
            entities.append(ExtractedEntity(
                entity_id=sender_id,
                entity_type=EntityType.PERSONNEL,
                properties={"name": sender_name, "role": "Sender", "date": date},
                confidence=1.0
            ))
        if receiver_id != "UNKNOWN_RECEIVER":
            entities.append(ExtractedEntity(
                entity_id=receiver_id,
                entity_type=EntityType.PERSONNEL,
                properties={"name": receiver_name, "role": "Receiver"},
                confidence=1.0
            ))

        if sender_id != "UNKNOWN_SENDER" and receiver_id != "UNKNOWN_RECEIVER":
            relationships.append(EntityRelationship(
                source_id=sender_id,
                target_id=receiver_id,
                relation_type=RelationType.CONNECTS_TO,
                properties={"context": "Email Communication", "subject": subject}
            ))

        # 4. Generative Graph Extraction from body
        body_entities, body_relationships = self._extract_body_graph(
            text=raw_body, 
            sender_id=sender_id, 
            receiver_id=receiver_id, 
            file_id=file_id
        )
        
        existing_ids = {e.entity_id for e in entities}
        for be in body_entities:
            if be.entity_id not in existing_ids:
                entities.append(be)
                existing_ids.add(be.entity_id)

        relationships.extend(body_relationships)

        # Handle attachment edges deterministically
        
        for attachment in extracted_attachments:
            # Assumes attachment IDs will be based on their filename for linking
            attachment_doc_id = f"doc_{attachment.name}" 
            relationships.append(EntityRelationship(
                source_id=file_id, 
                target_id=attachment_doc_id,
                relation_type=RelationType.PART_OF,
                properties={"context": "Email Attachment"}
            ))

        chunks = self.chunk_text(full_text) if hasattr(self, 'chunk_text') else [full_text]

        parsed_data = ParsedDocumentData(
            document_id=file_id,
            route_taken="EMAIL_PARSER",
            raw_text=full_text,
            text_chunks=chunks,
            embeddings=self.mock_embeddings(chunks) if hasattr(self, 'mock_embeddings') else [],
            entities=entities,
            relationships=relationships
        )

        return parsed_data, extracted_attachments