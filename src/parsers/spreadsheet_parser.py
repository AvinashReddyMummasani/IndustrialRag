import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
from groq import Groq
import instructor
from pydantic import BaseModel, Field, ValidationError

from src.parsers.base_parser import BaseParser
from src.core.schemas import (
    ParsedDocumentData, 
    ExtractedEntity, 
    EntityRelationship,
    EntityType,
    RelationType
)

logger = logging.getLogger(__name__)

class ColumnMappingSchema(BaseModel):
    """Structured schema mapping arbitrary spreadsheet structures to Graph Taxonomies."""
    id_column: str = Field(description="The header name containing unique identifiers/tags (e.g., P-101).")
    type_column: str = Field(description="The header name identifying what the row/item is.")
    
    # Value mapping dictionary to bridge arbitrary spreadsheet terminology to strict Graph Enums
    type_value_mappings: Dict[str, EntityType] = Field(
        description="A dictionary mapping unique raw values from the type_column to valid EntityType enums."
    )
    
    source_column: Optional[str] = Field(default=None, description="Header for source node connections.")
    target_column: Optional[str] = Field(default=None, description="Header for target node connections.")
    relation_type_column: Optional[str] = Field(default=None, description="Header for relation types.")
    
    relation_value_mappings: Dict[str, RelationType] = Field(
        default_factory=dict,
        description="A dictionary mapping unique raw values from the relation_type_column to valid RelationType enums."
    )

class SpreadsheetParser(BaseParser):
    def __init__(self, embedding_model, llm_client=None, llm_model: str = "llama-3.3-70b-versatile"):
        """
        Convert industrial spreadsheet data into Graph data.
        Assuming spreadsheet cols have less unique values and it contain id tags column and 
        devices column.
        """
        super().__init__(embedding_model)
        self.llm_model = llm_model
        
        # Wrap raw client with instructor
        raw_client = llm_client or Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.client = instructor.from_groq(raw_client, mode=instructor.Mode.TOOLS)

    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in {'.csv', '.xlsx', '.xls', '.xlsm'}
    
    def _infer_schema(self, df: pd.DataFrame, file_id: str) -> ColumnMappingSchema:
        """Uses Instructor to cleanly map headers and translate values to Enums."""
        columns = list(df.columns)
        sample_data = df.head(3).to_dict(orient="records")
        
        potential_cat_samples = {}
        for col in columns:
            try:
                unique_vals = [str(x) for x in df[col].dropna().unique()[:15] if str(x).strip()]
                if unique_vals:
                    potential_cat_samples[col] = unique_vals
            except Exception:
                continue

        prompt = f"""
        Analyze this schema profile of an industrial spreadsheet.
        1. Identify the structural column headers.
        2. Map the raw values found in the type and relationship columns to our strict internal Enums.
        
        Valid EntityType Options: {[e.value for e in EntityType]}
        Valid RelationType Options: {[r.value for r in RelationType]}

        Available Columns: {columns}
        Data Sample (First 3 Rows): {json.dumps(sample_data)}
        Categorical Value Samples per Column: {json.dumps(potential_cat_samples)}
        """

        try:
            return self.client.chat.completions.create(
                model=self.llm_model,
                response_model=ColumnMappingSchema,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a senior data architect. Map spreadsheet columns and translate unique values precisely to allowed graph enums."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_retries=2
            )
        except Exception as e:
            logger.error(f"[{file_id}] Schema inference via instructor failed: {e}")
            raise RuntimeError(f"Spreadsheet schema inference failed for {file_id}") from e

    def parse(self, file_path: Path, file_id: str) -> ParsedDocumentData:
        logger.info(f"[{file_id}] Executing High-Performance Spreadsheet Parser...")
        
        # 1. Load Data Matrix
        try:
            if file_path.suffix.lower() == ".csv":
                df = pd.read_csv(file_path, dtype=str) 
            else:
                df = pd.read_excel(file_path, engine='openpyxl', dtype=str)
                
            df = df.dropna(how='all', axis=0).dropna(how='all', axis=1).fillna("")
        except Exception as e:
            raise RuntimeError(f"Failed to read file: {e}")

        if df.empty:
            raise ValueError(f"Spreadsheet {file_id} has zero active rows.")

        # 2. Get Structured Mapping Definitions
        mapping = self._infer_schema(df, file_id)
        
        entities = []
        relationships = []
        
        # 3. Blazing Fast Vectorized Dictionary Parsing

        records = df.to_dict(orient='records') 
        
        id_col = mapping.id_column # Tag
        type_col = mapping.type_column # device type
        src_col = mapping.source_column # from 
        tgt_col = mapping.target_column # To
        rel_type_col = mapping.relation_type_column # connection_type
        
        type_map = mapping.type_value_mappings # {device : Entity: type}
        rel_map = mapping.relation_value_mappings # {reln : actual reln}

        if id_col in df.columns and type_col in df.columns:
            for row in records:
                raw_id = str(row.get(id_col, "")).strip()
                if not raw_id:
                    continue
                
                ent_id = raw_id.upper()
                raw_type = str(row.get(type_col, "")).strip()
                resolved_type = type_map.get(raw_type, EntityType.UNKNOWN)

                # Clear column noise
                properties = {
                    k: v for k, v in row.items() 
                    if k not in (id_col, type_col, src_col, tgt_col, rel_type_col) and v != ""
                }
                
                entities.append(ExtractedEntity(
                    entity_id=ent_id, 
                    entity_type=resolved_type, 
                    properties=properties,
                    confidence=1.0  # Deterministic tabular tracking
                ))
                
                # Process edges conditionally if complete
                if src_col and tgt_col and src_col in row and tgt_col in row:
                    source = str(row.get(src_col, "")).strip().upper()
                    target = str(row.get(tgt_col, "")).strip().upper()
                    
                    if source and target:
                        raw_rel = str(row.get(rel_type_col, "")) if rel_type_col else ""
                        # Default fallback edge type if none specified or resolved
                        resolved_rel = rel_map.get(raw_rel, RelationType.CONNECTS_TO)
                        
                        relationships.append(EntityRelationship(
                            source_id=source, 
                            target_id=target, 
                            relation_type=resolved_rel
                        ))

        # 4. Standardized Output Generation
        raw_text = df.to_csv(index=False)
        chunks = self.chunk_text(raw_text) if hasattr(self, 'chunk_text') else [raw_text]
        embeddings = []
        
        if hasattr(self, 'embedding_model') and self.embedding_model:
            try:
                embeddings = self.embedding_model.encode(chunks).tolist()
            except Exception as e:
                logger.error(f"[{file_id}] Embeddings generation failed: {e}")

        return ParsedDocumentData(
            document_id=file_id,
            route_taken="HYBRID_SPREADSHEET",
            raw_text=raw_text,
            text_chunks=chunks,
            embeddings=embeddings,
            entities=entities,
            relationships=relationships
        )