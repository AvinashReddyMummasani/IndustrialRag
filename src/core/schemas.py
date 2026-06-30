# --- Standard Library Imports ---
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

# --- Third-Party Imports ---
from pydantic import BaseModel, ConfigDict, Field, field_validator

# -----------------------------------------------------------------------------
# Strict Graph Taxonomies
# -----------------------------------------------------------------------------

class DocumentType(str, Enum):
    """MIME-type based document classification for routing extraction strategies."""
    PDF = "application/pdf"
    IMAGE = "image/png"
    SPREADSHEET = "text/csv"
    EMAIL = "message/rfc822"
    ARCHIVE = "application/zip"
    UNKNOWN = "unknown"

class EntityType(str, Enum):
    """
    Strict industrial taxonomy for asset and operations knowledge graphs.
    Descriptions must be passed to the LLM for accurate zero-shot classification.
    Adding new labels requires a schema migration and pipeline update.
    """
    
    # --- Physical Assets & Infrastructure ---
    SYSTEM = "SYSTEM"             # High-level functional groups (e.g., Secondary Cooling Water System, HVAC)
    EQUIPMENT = "EQUIPMENT"       # Major standalone machinery (e.g., Centrifugal Pump, Heat Exchanger, Turbine)
    COMPONENT = "COMPONENT"       # Sub-parts or spare parts of equipment (e.g., Mechanical Seal, O-Ring, Rotor)
    LOCATION = "LOCATION"         # Physical areas, zones, or sites (e.g., East Refinery Wing, Pump Room, Sector 4)

    # --- Operations & Physics ---
    PROCEDURE = "PROCEDURE"       # Prescribed actions or maintenance tasks (e.g., Lockout/Tagout, Calibration, Inspection)
    PARAMETER = "PARAMETER"       # Measurable operational variables or limits (e.g., Temperature, Pressure, Flow Rate)
    MATERIAL = "MATERIAL"         # Physical substances, fluids, or consumables (e.g., Crude Oil, Nitrogen, Coolant)

    # --- Failure & RCA Context ---
    INCIDENT = "INCIDENT"         # Unplanned events, anomalies, or operational states (e.g., Leak, Trip, Overheating)
    DEFECT = "DEFECT"             # Specific physical damage, wear, or degradation mechanisms (e.g., Corrosion, Pitting, Fatigue)

    # --- Administrative & Compliance ---
    PERSONNEL = "PERSONNEL"       # Specific human actors, job titles, or roles (e.g., Lead Technician, Shift Supervisor)
    ORGANIZATION = "ORGANIZATION" # Companies, regulatory bodies, vendors, or contractors (e.g., OSHA, Siemens, EPA)
    REGULATION = "REGULATION"     # Codes, standards, policies, and legal mandates (e.g., ISO 9001, API 610, CFR)
    DOCUMENT = "DOCUMENT"         # Reference materials, forms, or permits (e.g., Hot Work Permit, OEM Manual, P&ID)
    DATE = "DATE"                 # Temporal markers or timestamps
    
    # --- Fallback ---
    UNKNOWN = "UNKNOWN"           # Unclassifiable concepts. Mandatory to prevent Pydantic validation crashes.


class ExtractedEntity(BaseModel):
    entity_id: str = Field(description="A unique, deterministic identifier for the entity.")
    entity_type: EntityType = Field(
        ..., 
        description="Must strictly map to the defined EntityType enum. If the concept is ambiguous, classify as UNKNOWN."
    )
    properties: dict = Field(
        default_factory=dict, 
        description="Key-value pairs of extracted attributes (e.g., {'type': 'centrifugal', 'manufacturer': 'Flowserve'})."
    )
    confidence: float = Field(
        ge=0.0, le=1.0, 
        description="Calibrated probability of accurate extraction. Output < 0.85 if the text is ambiguous."
    )

class RelationType(str, Enum):
    """Strict taxonomy for Neo4j edge types. Dictates topological traversal logic."""
    FLOWS_INTO = "FLOWS_INTO"
    CONNECTS_TO = "CONNECTS_TO"
    LOCATED_IN = "LOCATED_IN"
    GOVERNS = "GOVERNS"
    MAINTAINS = "MAINTAINS"
    HAS_PARAMETER = "HAS_PARAMETER"
    PART_OF = "PART_OF"

# -----------------------------------------------------------------------------
# Unified Domain Models
# -----------------------------------------------------------------------------

class ExtractedEntity(BaseModel):
    """
    Unified, immutable entity model used across text LLMs, Vision VLMs, and DB mappings.
    Frozen to ensure immutability during asynchronous pipeline execution.
    """
    model_config = ConfigDict(use_enum_values=True, frozen=True)

    entity_id: str = Field(
        ..., 
        description="Normalized unique identifier/tag (e.g., 'V-101', 'P-200'). Must be uppercase."
    )
    raw_mention: Optional[str] = Field(
        default=None,
        description="The exact text string as it appears in the source for NLP span mapping."
    )
    entity_type: EntityType = Field(
        default=EntityType.UNKNOWN,
        description="Categorical classification mapped directly to Neo4j node labels."
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value metadata (e.g., status, capacity). Schema-less by design for flexibility."
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Extraction confidence score. Used for downstream threshold filtering."
    )

    @field_validator("entity_id", mode="before")
    @classmethod
    def normalize_id(cls, v: Any) -> str:
        """Strip whitespaces and cast to uppercase to prevent cache/DB lookup misses."""
        if not isinstance(v, str):
            raise ValueError(f"entity_id must be a string, received {type(v)}")
        return v.strip().upper()


class EntityRelationship(BaseModel):
    """
    Unified, immutable edge representation linking two known entities.
    """
    model_config = ConfigDict(use_enum_values=True, frozen=True)

    source_id: str = Field(..., description="Normalized unique identifier of the source entity.")
    target_id: str = Field(..., description="Normalized unique identifier of the target entity.")
    relation_type: RelationType = Field(
        ..., 
        description="The topological or semantic connection acting as the edge label."
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Edge metadata (e.g., 'pipe_material': 'steel')."
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, 
        description="Extraction confidence score for the relationship."
    )

    @field_validator("source_id", "target_id", mode="before")
    @classmethod
    def normalize_edge_ids(cls, v: Any) -> str:
        """
        CRITICAL: Edge IDs must undergo the exact same normalization as Entity IDs.
        Failure to do this results in disjointed graphs and orphaned nodes in Neo4j.
        """
        if not isinstance(v, str):
            raise ValueError(f"Edge IDs must be strings, received {type(v)}")
        return v.strip().upper()

# -----------------------------------------------------------------------------
# Extraction Specific Wrappers
# -----------------------------------------------------------------------------

class PIDDiagramExtractionSchema(BaseModel):
    """
    Root container for LangChain's with_structured_output().
    Enforces a strict contract for vision-language models parsing schematics.
    """
    entities: List[ExtractedEntity] = Field(
        description="List of all extracted equipment, parameters, or component entities."
    )
    relationships: List[EntityRelationship] = Field(
        description="List of all topological relationships and flows between components."
    )
    text_summary: str = Field(
        description="Descriptive operational summary of the parsed diagram."
    )

class EmailGraphExtraction(BaseModel):
    """Enforces structured graph extraction from unstructured email bodies."""
    entities: List[ExtractedEntity] = Field(
        default_factory=list,
        description="List of entities found in the email body."
    )
    relationships: List[EntityRelationship] = Field(
        default_factory=list,
        description="Connections linking the sender/receiver to equipment or concepts."
    )

class ColumnMappingSchema(BaseModel):
    """Structured schema mapping arbitrary spreadsheet structures to Graph Taxonomies."""
    id_column: str = Field(description="Header name containing unique identifiers/tags.")
    type_column: str = Field(description="Header name identifying the entity category.")
    type_value_mappings: Dict[str, EntityType] = Field(
        description="Maps raw spreadsheet type values to strict EntityType enums."
    )
    source_column: Optional[str] = Field(default=None, description="Header for source node IDs.")
    target_column: Optional[str] = Field(default=None, description="Header for target node IDs.")
    relation_type_column: Optional[str] = Field(default=None, description="Header for relation types.")
    relation_value_mappings: Dict[str, RelationType] = Field(
        default_factory=dict,
        description="Maps raw spreadsheet relation values to strict RelationType enums."
    )

# -----------------------------------------------------------------------------
# Pipeline State & Internal Data Transport
# -----------------------------------------------------------------------------

class ParsedDocumentData(BaseModel):
    """
    The root payload passed from parsers to database clients.
    WARNING: Passing `embeddings` as List[List[float]] in memory for large documents
    will cause massive serialization bottlenecks. Consider passing references or storing
    these directly to the vector DB rather than holding them in the Pydantic model.
    """
    document_id: str
    route_taken: str = Field(description="Tracks which parser strategy executed this file.")
    raw_text: str = Field(default="", description="Full raw text. Keep empty for large files to optimize RAM.")
    text_chunks: List[str]
    embeddings: List[List[float]] = Field(default_factory=list)
    entities: List[ExtractedEntity] = Field(default_factory=list)
    relationships: List[EntityRelationship] = Field(default_factory=list)

class AgentState(TypedDict):
    """State machine representation for LangGraph cyclic execution."""
    query: str
    combined_context: str
    generation: str
    evidence: List[str]
    is_relevant: bool
    is_grounded: bool
    is_useful: bool
    retries: int

class EntityExtractor(BaseModel):
    """Payload capturing industrial tags and extraction rationale from user queries."""
    entities: List[ExtractedEntity] = Field(
        default_factory=list,
        description="Unique industrial identifiers matched within the instruction text."
    )
    reasoning_log: str = Field(
        ...,
        description="Engineering rationale explaining why these identifiers were grouped."
    )

# -----------------------------------------------------------------------------
# Grading & Auditing Interfaces
# -----------------------------------------------------------------------------

class RelevanceGrade(BaseModel):
    is_relevant: bool = Field(description="True if data is informative relative to the query parameter.")
    rationale: str = Field(description="Root cause justification for the relevance boolean.")

class GroundednessGrade(BaseModel):
    is_grounded: bool = Field(description="True if facts correlate 1:1 with source attributes without interpolation.")
    hallucinated_facts: List[str] = Field(default_factory=list)

class UtilityGrade(BaseModel):
    fully_answers: bool = Field(description="True if generation meets technical operational clarity constraints.")

class CitedAnswer(BaseModel):
    answer: str = Field(description="Comprehensive technical analysis responding to user request.")
    evidence_links: List[str] = Field(
        description="List of raw source file names or document IDs explicitly present in prompt metadata."
    )

class QueryRequest(BaseModel):
    query_text: str = Field(
        ..., 
        description="The operational engineering question to route through the Graph-RAG pipeline."
    )

class QueryResponse(BaseModel):
    answer: str = Field(description="The synthesized, grounded response.")
    evidence_links: List[str] = Field(
        default_factory=list,
        description="Verified document IDs or filenames providing provenance for the answer."
    )

class APIWebResponse(BaseModel):
    status: str
    report: Dict[str, Any]

class AuditRequest(BaseModel):
    asset_id: str = Field(..., description="Fuzzy or explicit operational equipment tag.")
    asset_type: str = Field(..., description="Classification of the equipment.")
    target_standard: str = Field(..., description="Regulatory framework (e.g., 'OISD-144').")

class AuditResponse(BaseModel):
    status: str
    report: str

class AuditReportSchema(BaseModel):
    executive_summary: str = Field(description="High-level summary of the audit findings.")
    regulatory_baseline_clauses: List[str] = Field(description="Specific legal thresholds evaluated.")
    operational_deviations: List[str] = Field(description="Explicit list of deviations. Empty if none.")
    is_compliant: bool = Field(description="True if 0 deviations, False otherwise.")
    evidence_references: List[str] = Field(description="DB primary keys or file names proving the status.")
