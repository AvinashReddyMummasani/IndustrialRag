from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum

# --- Strict Graph Taxonomies ---
class DocumentType(str, Enum):
    PDF = "application/pdf"
    IMAGE = "image/png"
    SPREADSHEET = "text/csv"
    EMAIL = "message/rfc822"
    ARCHIVE = "application/zip"
    UNKNOWN = "unknown"

class EntityType(str, Enum):
    """Strict taxonomy to prevent Neo4j label pollution."""
    EQUIPMENT = "EQUIPMENT"
    COMPONENT = "COMPONENT"
    PARAMETER = "PARAMETER"
    REGULATION = "REGULATION"
    PERSONNEL = "PERSONNEL"
    DATE = "DATE"
    SYSTEM = "SYSTEM"
    UNKNOWN = "UNKNOWN"

class RelationType(str, Enum):
    """Strict taxonomy for Neo4j edge types."""
    FLOWS_INTO = "FLOWS_INTO"
    CONNECTS_TO = "CONNECTS_TO"
    LOCATED_IN = "LOCATED_IN"
    GOVERNS = "GOVERNS"
    MAINTAINS = "MAINTAINS"
    HAS_PARAMETER = "HAS_PARAMETER"
    PART_OF = "PART_OF"

# --- Unified Domain Models ---
class ExtractedEntity(BaseModel):
    """Unified entity model used by both Text LLMs and Vision VLMs."""
    entity_id: str = Field(
        description="Normalized unique identifier/tag, e.g., 'V-101', 'P_200'. Must be uppercase."
    )
    entity_type: EntityType = Field(
        description="Categorical classification of the entity from the allowed Enum."
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value pairs capturing metadata (status, capacity, regulatory clause)."
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence score of the extraction. 1.0 for deterministic, <1.0 for LLM/OCR."
    )

class EntityRelationship(BaseModel):
    """Unified relationship model."""
    source_id: str = Field(description="Unique identifier of the source entity.")
    target_id: str = Field(description="Unique identifier of the target entity.")
    relation_type: RelationType = Field(
        description="The nature of the topological or semantic connection."
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata about the edge itself (e.g., 'pipe_material': 'steel')."
    )

# --- Pipeline Output Models ---
class ParsedDocumentData(BaseModel):
    """The root payload passed from Parsers to the Database Clients."""
    document_id: str
    route_taken: str = Field(description="Tracks which parser strategy executed this file.")
    raw_text: str
    text_chunks: List[str]
    embeddings: List[List[float]] = Field(default_factory=list)
    entities: List[ExtractedEntity] = Field(default_factory=list)
    relationships: List[EntityRelationship] = Field(default_factory=list)

# --- Vision/VLM Specific Wrappers ---
class PIDDiagramExtractionSchema(BaseModel):
    """
    Root container strictly for LangChain's with_structured_output().
    It reuses the core ExtractedEntity and EntityRelationship models to ensure parity.
    """
    entities: List[ExtractedEntity] = Field(
        description="List of all extracted equipment, parameters, or component entities."
    )
    relationships: List[EntityRelationship] = Field(
        description="List of all topological relationships and flows between components."
    )
    text_summary: str = Field(
        description="A descriptive summary outlining the operation, components, and primary focus."
    )


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

# =====================================================================
# Strict Interface Schemas
# =====================================================================

class EntityExtractor(BaseModel):
    entities: List[str] = Field(description="Normalized uppercase equipment keys/tags found in user prompts.")

class RelevanceGrade(BaseModel):
    is_relevant: bool = Field(description="True if data is informative relative to the core query parameter.")
    rationale: str = Field(description="Root cause justification.")

class CitedAnswer(BaseModel):
    answer: str = Field(description="Comprehensive technical analysis responding to user request.")
    evidence_links: List[str] = Field(
        description="List of raw source file names or matching document IDs explicitly present in prompt metadata."
    )

class GroundednessGrade(BaseModel):
    is_grounded: bool = Field(description="True if facts correlate 1:1 with source attributes without interpolation.")
    hallucinated_facts: List[str] = Field(default_factory=list)

class UtilityGrade(BaseModel):
    fully_answers: bool = Field(description="True if generation meets technical operational clarity constraints.")

class QueryRequest(BaseModel):
    query_text: str = Field(
        ..., 
        description="The operational engineering question to route through the Graph-RAG pipeline."
    )

class QueryResponse(BaseModel):
    answer: str = Field(
        description="The synthesized, grounded response."
    )
    evidence_links: List[str] = Field(
        default_factory=list,
        description="List of verified document IDs or filenames providing provenance for the answer."
    )

class APIWebResponse(BaseModel):
    status: str
    report: str

class AuditResponse(BaseModel):
    status: str
    report: str

class AuditRequest(BaseModel):
    asset_id: str = Field(..., description="Fuzzy or explicit operational equipment tag.")
    asset_type: str = Field(..., description="Classification of the equipment.")
    target_standard: str = Field(..., description="Regulatory framework (e.g., 'OISD-144').")


class AuditReportSchema(BaseModel):
    executive_summary: str = Field(description="High-level summary of the audit.")
    regulatory_baseline_clauses: List[str] = Field(description="Specific legal thresholds found.")
    operational_deviations: List[str] = Field(description="Explicit list of deviations. Empty if none.")
    is_compliant: bool = Field(description="True if 0 deviations, False otherwise.")
    evidence_references: List[str] = Field(description="DB primary keys or file names proving the status.")