import magic
import json
import pathlib
import logging
import asyncio
from typing import Dict, Any, Optional

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
    def __init__(self, task_queue=None, model=None, vision_client=None,llm=None):
        self.vision_parser = VisionParser(embedding_model=model, vision_model_client=vision_client)
        self.semantic_parser = SemanticParser(embedding_model=model)
        self.digital_text_parser = DigitalTextParser(embedding_model=model,llm=llm)
        self.email_parser = EmailParser(model=model,llm=llm)
        self.spreadsheet_parser = SpreadsheetParser(embedding_model=model,llm=llm)
        
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

    async def _handle_archive(self, file_path: pathlib.Path, file_id: str, current_depth: int):
        """Unpacks archives asynchronously and prevents recursive decompression attacks."""
        if current_depth >= MAX_RECURSION_DEPTH:
            logger.error(f"Max recursion depth reached for archive {file_id}. Aborting extraction.")
            return

        logger.info(f"Unpacking archive {file_id} (Depth: {current_depth})")
        
        extracted_files = await asyncio.to_thread(self.archive_processor.unpack, file_path, file_id)
        
        await self._fan_out_files(extracted_files, parent_id=file_id, next_depth=current_depth + 1)

    async def _handle_email_attachments(self, attachments: list[pathlib.Path], parent_file_id: str, current_depth: int):
        """Routes extracted attachments into the processing queue."""
        if not attachments:
            return

        if current_depth >= MAX_RECURSION_DEPTH:
            logger.error(f"Max recursion depth reached dropping attachments for email {parent_file_id}.")
            return

        logger.info(f"Fanning out {len(attachments)} attachments from email {parent_file_id}")
        await self._fan_out_files(attachments, parent_id=parent_file_id, next_depth=current_depth + 1)

    async def _fan_out_files(self, files: list[pathlib.Path], parent_id: str, next_depth: int):
        """Pushes files to the execution layer asynchronously."""
        for file in files:
            new_file_id = f"doc_{file.name}"  
            
            if self.task_queue:
                
                self.task_queue.enqueue(
                    'process_file', 
                    str(file), new_file_id, file.name, parent_id, next_depth
                )
            else:
                
                await self.process_file(file, new_file_id, file.name, parent_id, next_depth)

    async def process_file(self, file_path: pathlib.Path, file_id: str, filename: str, parent_id: Optional[str] = None, depth: int = 0) -> None:
        """Main execution coroutine for a single file."""
        try:
            try:
                
                mime_type = magic.from_file(str(file_path), mime=True)
            except Exception as e:
                logger.error(f"Failed to resolve MIME signature for {filename}: {e}")
                return

            handler = self.routes.get(mime_type) 
            
            if not handler:
                logger.warning(f"Dropping unroutable file {filename} with MIME {mime_type}")
                return
                
            # Route A: Archives (Recursion)
            if mime_type in ["application/zip", "application/x-tar", "application/gzip"]:
                await self._handle_archive(file_path, file_id, depth)
                return 

            # Route B: Emails (Yields Tuple: Data + Attachments)
            if mime_type in ["message/rfc822", "application/vnd.ms-outlook"]:
                parsed_data, attachments = await asyncio.to_thread(handler, file_path, file_id)
                await self._handle_email_attachments(attachments, file_id, depth)
            
            # Route C: Standard Documents
            else:
                parsed_data = await asyncio.to_thread(handler, file_path, file_id)
            
            if parsed_data:
                await self._write_to_database(parsed_data, filename, mime_type, parent_id)
                logger.info(f"Successfully processed document {file_id} ({filename})")
            
        except Exception as e:
            logger.error(f"Pipeline failed for {file_id} ({filename}): {str(e)}", exc_info=True)
        finally:
            if file_path.exists():
                try:
                    file_path.unlink()
                except OSError:
                    pass

    async def _write_to_database(self, parsed_data: ParsedDocumentData, filename: str, mime_type: str, parent_id: Optional[str]):
        """Transactional Outbox write using native asyncpg patterns."""
        try:
            async with PostgresPool.get_connection() as conn:
                
                # 1. Master Document Record
                await conn.execute(
                    "INSERT INTO documents (id, filename, doc_type, parent_id) VALUES ($1, $2, $3, $4);",
                    parsed_data.document_id, filename, mime_type, parent_id
                )
                
                # 2. Dense Vector Text Chunks
                if parsed_data.text_chunks:
                    chunk_records = [
                        (parsed_data.document_id, chunk, str(emb))
                        for chunk, emb in zip(parsed_data.text_chunks, parsed_data.embeddings)
                    ]
                    await conn.executemany(
                        "INSERT INTO document_chunks (document_id, chunk_text, embedding) VALUES ($1, $2, $3::vector);",
                        chunk_records
                    )
                
                # 3. Graph Edge Outbox
                graph_payload = {
                    "entities": [e.model_dump() for e in parsed_data.entities],
                    "relationships": [r.model_dump() for r in parsed_data.relationships]
                }
                
                await conn.execute(
                    "INSERT INTO neo4j_outbox (document_id, payload, status) VALUES ($1, $2, 'PENDING');",
                    parsed_data.document_id, json.dumps(graph_payload)
                )

        except Exception as e:
            logger.error(f"Database transaction failed for {filename}: {e}", exc_info=True)
            raise RuntimeError(f"Database transaction failed: {e}")