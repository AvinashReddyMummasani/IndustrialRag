import os
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from groq import Groq
from pydantic import BaseModel, ValidationError

from src.parsers.base_parser import BaseParser
from src.core.schemas import ParsedDocumentData, ExtractedEntity, EntityRelationship

logger = logging.getLogger(__name__)

class ColumnMappingSchema(BaseModel):
    id_column: str
    type_column: str
    source_column: str = None
    target_column: str = None
    relation_type_column: str = None

class SpreadsheetParser(BaseParser):
    def __init__(self, embedding_model, llm_client=None):
        """
        Accepts an injected Groq client to share connection pools, 
        or falls back to a local instance if none is provided.
        """
        super().__init__(embedding_model)
        self.client = llm_client or Groq(api_key=os.getenv("GROQ_API_KEY"))

    def can_handle(self, file_path: Path) -> bool:
        valid_extensions = {'.csv', '.xlsx', '.xls', '.xlsm'}
        return file_path.suffix.lower() in valid_extensions
    
    def _infer_schema(self, df: pd.DataFrame, file_id: str) -> ColumnMappingSchema:
        """Uses the LLM to map arbitrary spreadsheet columns to our domain schema."""
        # OPTIMIZATION: Do not send markdown tables. Send a JSON dictionary of the first 2 rows.
        # This prevents token bloat on spreadsheets with 50+ columns.
        columns = list(df.columns)
        sample_data = df.head(2).to_dict(orient="records")
        
        prompt = f"""
        Analyze this sample of an industrial equipment spreadsheet.
        Map the existing column headers to our internal schema.
        
        Internal Schema Requirements:
        - id_column: The unique equipment tag (e.g., P-101)
        - type_column: What the equipment is (e.g., Pump, Valve)
        - source_column (optional): Indicates a source connection.
        - target_column (optional): Indicates a target connection.
        - relation_type_column (optional): Indicates the flow or relationship type.

        Available Columns: {columns}
        Data Sample: {json.dumps(sample_data)}

        Respond ONLY in valid JSON matching this exact structure. Use exact column names. 
        If an optional column does not exist, map it to null.
        {{
            "id_column": "Exact Header Name",
            "type_column": "Exact Header Name",
            "source_column": null,
            "target_column": null,
            "relation_type_column": null
        }}
        """
        
        try:
            completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                response_format={"type": "json_object"},
                temperature=0.0
            )
            
            raw_content = completion.choices[0].message.content.strip()
            
            # Robust sanitization against markdown wrappers
            if raw_content.startswith("```json"):
                raw_content = raw_content.replace("```json", "").replace("```", "").strip()
                
            data = json.loads(raw_content)
            return ColumnMappingSchema(**data)
            
        except (Exception, ValidationError) as e:
            logger.error(f"[{file_id}] Schema inference failed: {e}")
            raise RuntimeError(f"Spreadsheet schema inference failed for {file_id}") from e

    def parse(self, file_path: Path, file_id: str) -> ParsedDocumentData:
        logger.info(f"[{file_id}] Executing Hybrid Spreadsheet Parser...")
        
        # 1. Load Data with strict type enforcement and NaN cleanup
        try:
            if file_path.suffix.lower() == ".csv":
                # dtype=str prevents Pandas from implicitly casting tags like '0012' to int 12
                df = pd.read_csv(file_path, dtype=str) 
            else:
                df = pd.read_excel(file_path, engine='openpyxl', dtype=str)
                
            df = df.dropna(how='all', axis=0).dropna(how='all', axis=1).fillna("")
        except Exception as e:
            raise RuntimeError(f"Failed to read spreadsheet file: {e}")

        if df.empty:
            raise ValueError(f"Spreadsheet {file_id} contains no readable data.")

        # 2. Infer Mapping via LLM
        mapping = self._infer_schema(df, file_id)
        
        entities = []
        relationships = []
        
        # 3. Deterministic Extraction (OPTIMIZED: NO iterrows)
        if mapping.id_column and mapping.type_column and mapping.id_column in df.columns:
            
            # Convert to list of dicts for O(1) key lookups and blazing fast iteration
            records = df.to_dict(orient='records')
            
            for row in records:
                ent_id = str(row.get(mapping.id_column, "")).strip()
                ent_type = str(row.get(mapping.type_column, "")).strip()
                
                if not ent_id:
                    continue

                # Isolate properties dynamically without copying the entire dict repeatedly
                properties = {k: v for k, v in row.items() if k not in (mapping.id_column, mapping.type_column) and v != ""}
                
                entities.append(ExtractedEntity(
                    entity_id=ent_id, 
                    entity_type=ent_type, 
                    properties=properties
                ))
                
                # Handle Optional Relationships
                if mapping.source_column and mapping.target_column and mapping.relation_type_column:
                    source = str(row.get(mapping.source_column, "")).strip()
                    target = str(row.get(mapping.target_column, "")).strip()
                    rel_type = str(row.get(mapping.relation_type_column, "")).strip()
                    
                    if source and target:
                        relationships.append(EntityRelationship(
                            source_id=source, 
                            target_id=target, 
                            relation_type=rel_type
                        ))

        # 4. Standardize text and compute actual embeddings
        raw_text = df.to_csv(index=False)
        
        # Rely on the parent BaseParser implementation for chunking and embedding
        chunks = self.chunk_text(raw_text) if hasattr(self, 'chunk_text') else [raw_text]
        embeddings = []
        
        if hasattr(self, 'embedding_model') and self.embedding_model:
            try:
                embeddings = self.embedding_model.encode(chunks).tolist()
            except Exception as e:
                logger.error(f"[{file_id}] Failed to generate text embeddings: {e}")

        return ParsedDocumentData(
            document_id=file_id,
            route_taken="HYBRID_SPREADSHEET",
            raw_text=raw_text,
            text_chunks=chunks,
            embeddings=embeddings,
            entities=entities,
            relationships=relationships
        )