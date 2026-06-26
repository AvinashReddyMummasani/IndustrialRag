import magic
import json
import pathlib
import logging
import uuid
from typing import Dict, Any, Optional
from psycopg2.extras import execute_values

from src.core.schemas import ParsedDocumentData
from src.db.postgres_client import PostgresPool
from src.parsers.document_parser import DocumentOrchestrator
from src.parsers.vision_parser import VisionParser
from src.parsers.semantic_parser import SemanticParser
from src.parsers.digital_text_parser import DigitalTextParser
from src.parsers.email_parser import EmailParser 
from src.parsers.spreadsheet_parser import SpreadsheetParser 
from src.parsers.archive_parser import ArchiveProcessor

logger = logging.getLogger(__name__)

MAX_RECURSION_DEPTH = 3

class IngestionPipeline:
    def __init__(self, task_queue=None,model=None,vision_clinet=None):
        self.vision_parser = VisionParser(embedding_model=model,vision_model_client=vision_clinet)
        self.semantic_parser = SemanticParser(embedding_model =model)
        self.digital_text_parser = DigitalTextParser(model=model)
        self.email_parser = EmailParser(model=model)
        self.spreadsheet_parser = SpreadsheetParser(embedding_model=model)
        
        # Documents can contain images, scanned_images, P&IDs, Normal text (pdfs)
        self.pdf_orchestrator = DocumentOrchestrator(
            parsers=[self.vision_parser, self.semantic_parser, self.digital_text_parser], 
            max_workers=4
        )
        
        self.archive_processor = ArchiveProcessor(extract_dir=pathlib.Path("./temp_uploads"))
        self.task_queue = task_queue 
        
        self.routes = {
            "application/pdf": self.pdf_orchestrator.parse_document,
            
            "image/png": self.vision_parser.parse,
            "image/jpeg": self.vision_parser.parse,
            "image/tiff": self.vision_parser.parse,
            
            "text/csv": self.spreadsheet_parser.parse,
            "application/vnd.ms-excel": self.spreadsheet_parser.parse,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": self.spreadsheet_parser.parse,
            
            "message/rfc822": self.email_parser.parse,
            "application/vnd.ms-outlook": self.email_parser.parse,
            
            "application/zip": self._handle_archive,
            "application/x-tar": self._handle_archive,
            "application/gzip": self._handle_archive,
        }

    def _handle_archive(self, file_path: pathlib.Path, file_id: str, current_depth: int):
        """Unpacks archives and prevents recursive decompression attacks."""
        if current_depth >= MAX_RECURSION_DEPTH:
            logger.error(f"Max recursion depth reached for archive {file_id}. Aborting extraction.") # tell it to user
            return

        logger.info(f"Unpacking archive {file_id} (Depth: {current_depth})")
        extracted_files = self.archive_processor.unpack(file_path, file_id)
        
        self._fan_out_files(extracted_files, parent_id=file_id, next_depth=current_depth + 1)

    def _handle_email_attachments(self, attachments: list[pathlib.Path], parent_file_id: str, current_depth: int):
        """Routes extracted attachments into the processing queue."""
        if not attachments:
            return

        if current_depth >= MAX_RECURSION_DEPTH:
            logger.error(f"Max recursion depth reached dropping attachments for email {parent_file_id}.")
            return

        logger.info(f"Fanning out {len(attachments)} attachments from email {parent_file_id}")
        self._fan_out_files(attachments, parent_id=parent_file_id, next_depth=current_depth + 1)

    def _fan_out_files(self, files: list[pathlib.Path], parent_id: str, next_depth: int): # add async upload
        """Pushes files to processes synchronously in dev mode."""
        for file in files:
            new_file_id = f"doc_{file.name}"  
            
            # on background adds one file at a time to parse
            if self.task_queue:
                self.task_queue.enqueue(
                    'process_file', 
                    str(file), 
                    new_file_id, 
                    file.name,
                    parent_id,
                    next_depth
                )
            else:
                self.process_file(file, new_file_id, file.name, parent_id, next_depth)

    def process_file(self, file_path: pathlib.Path, file_id: str, filename: str, parent_id: Optional[str] = None, depth: int = 0) -> None:
        """Main execution thread for a single file."""
        try:
            try:
                mime_type = magic.from_file(str(file_path), mime=True)
            except Exception as e:
                logger.error(f"Failed to resolve MIME signature for {filename}: {e}") # the user should Know it
                return

            handler = self.routes.get(mime_type) # returns Appropriate parser
            
            if not handler:
                logger.warning(f"Dropping unroutable file {filename} with MIME {mime_type}")
                return
                
            # Route A: Archives (Recursion)
            if mime_type in ["application/zip", "application/x-tar", "application/gzip"]:
                self._handle_archive(file_path, file_id, depth)  # currently it just uploaded files in temp_uploads
                return 

            # Route B: Emails (Yields Tuple: Data + Attachments)
            if mime_type in ["message/rfc822", "application/vnd.ms-outlook"]:
                parsed_data, attachments = handler(file_path, file_id)
                self._handle_email_attachments(attachments, file_id, depth)
            
            # Route C: Standard Documents
            else:
                parsed_data = handler(file_path, file_id)
            
            if parsed_data:
                self._write_to_database(parsed_data, filename, mime_type, parent_id)
                logger.info(f"Successfully processed document {file_id} ({filename})")
            
        except Exception as e:
            logger.error(f"Pipeline failed for {file_id} ({filename}): {str(e)}")
        finally:
            if file_path.exists():
                try:
                    file_path.unlink()
                except OSError:
                    pass

    def _write_to_database(self, parsed_data: ParsedDocumentData, filename: str, mime_type: str, parent_id: Optional[str]):
        """Transactional Outbox write."""
        with PostgresPool.get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        "INSERT INTO documents (id, filename, doc_type, parent_id) VALUES (%s, %s, %s, %s);",
                        (parsed_data.document_id, filename, mime_type, parent_id)
                    )
                    
                    if parsed_data.text_chunks:
                        chunk_records = [
                            (parsed_data.document_id, chunk, emb)
                            for chunk, emb in zip(parsed_data.text_chunks, parsed_data.embeddings)
                        ]
                        execute_values(
                            cur,
                            "INSERT INTO document_chunks (document_id, chunk_text, embedding) VALUES %s;",
                            chunk_records
                        )
                    
                    graph_payload = {
                        "entities": [e.model_dump() for e in parsed_data.entities],
                        "relationships": [r.model_dump() for r in parsed_data.relationships]
                    }
                    
                    cur.execute(
                        "INSERT INTO neo4j_outbox (document_id, payload, status) VALUES (%s, %s, 'PENDING');",
                        (parsed_data.document_id, json.dumps(graph_payload))
                    )
                    
                    conn.commit()
                except Exception as e:
                    conn.rollback()
                    raise RuntimeError(f"Database transaction failed: {e}")