"""Input document representation - universal format for all input sources."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class InputDocument(BaseModel):
    """Universal input document format that all input providers convert to.
    
    This model serves as the standardized bridge between external data sources
    (files, web forms, APIs, etc.) and the internal Example format used by
    the processing pipeline. All input providers must convert their data into
    this format before further processing.
    
    Fields:
        doc_id: Unique document identifier. Typically a PMID (PubMed ID) for
                biomedical literature, but can be any unique string for other
                sources. Generated automatically if not provided by the source.
                Examples: "PMID_12345678", "doc_abc123", "web_form_20240101_001"
        
        text: The main document content. For PubMed abstracts, this is the
              abstract text. For full-text articles, this may be the full body.
              Must be non-empty. Whitespace-only strings are not allowed.
        
        title: Optional document title. Populated when available from the source.
               For PubMed articles, this is the article title. For file uploads,
               may be derived from filename or metadata. None if not available.
        
        source: Identifier for the data source system. Used for tracking and
                debugging. Common values:
                - "pubtator": From PubTator API or database
                - "file": From uploaded file (JSON, TXT, etc.)
                - "web_form": From web interface form submission
                - "api": From REST API endpoint
                - "pubmed": Direct from PubMed API
                - "bioc": From BioC XML format
        
        bioc_annotations: Optional list of pre-existing entity annotations in
                          BioC XML format. Populated when the input source is
                          BioC XML or when entities are provided upfront.
                          Structure: List of dictionaries, where each dict
                          represents an annotation with keys like:
                          - "id": Annotation ID
                          - "type": Entity type (e.g., "Gene", "Disease")
                          - "text": Mentioned text span
                          - "locations": List of location dicts with "offset"
                            and "length" keys
                          - "infons": Additional information dict
                          Example:
                          [
                              {
                                  "id": "T1",
                                  "type": "Gene",
                                  "text": "EGFR",
                                  "locations": [{"offset": 0, "length": 4}],
                                  "infons": {"gene_id": "1956"}
                              }
                          ]
                          None if the document is raw text without annotations.
        
        metadata: Extensible dictionary for additional document metadata.
                  Common fields include:
                  - "publication_date": Publication date (ISO format or string)
                  - "journal": Journal name
                  - "authors": List of author names
                  - "pmid": PubMed ID (if different from doc_id)
                  - "doi": Digital Object Identifier
                  - "file_path": Original file path (for file sources)
                  - "upload_timestamp": When document was uploaded/processed
                  - "language": Document language code (e.g., "en")
                  - Any other source-specific metadata
    
    Example:
        >>> doc = InputDocument(
        ...     doc_id="PMID_12345678",
        ...     text="The EGFR protein is associated with cancer development.",
        ...     title="EGFR in Cancer",
        ...     source="pubtator",
        ...     metadata={
        ...         "publication_date": "2024-01-15",
        ...         "journal": "Nature",
        ...         "authors": ["Smith, J.", "Doe, A."]
        ...     }
        ... )
    """
    
    doc_id: str = Field(
        ...,
        min_length=1,
        description="Unique document identifier (PMID or generated ID)"
    )
    text: str = Field(
        ...,
        min_length=1,
        description="Document text content (abstract or full text)"
    )
    title: Optional[str] = Field(
        default=None,
        description="Document title"
    )
    source: str = Field(
        ...,
        description="Data source identifier (e.g., 'pubtator', 'file', 'web_form')"
    )
    bioc_annotations: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Pre-existing entity annotations in BioC XML format"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible metadata dictionary"
    )
    
    model_config = {
        "str_strip_whitespace": True,
        "json_schema_extra": {
            "example": {
                "doc_id": "PMID_12345678",
                "text": (
                    "The epidermal growth factor receptor (EGFR) is a "
                    "transmembrane protein that plays a crucial role in cell "
                    "proliferation and survival. Mutations in EGFR have been "
                    "linked to various types of cancer, including lung cancer "
                    "and glioblastoma."
                ),
                "title": "EGFR Mutations in Cancer Development",
                "source": "pubtator",
                "bioc_annotations": [
                    {
                        "id": "T1",
                        "type": "Gene",
                        "text": "EGFR",
                        "locations": [
                            {"offset": 4, "length": 4}
                        ],
                        "infons": {
                            "gene_id": "1956",
                            "NCBI Gene": "1956",
                            "species": "9606"
                        }
                    },
                    {
                        "id": "T2",
                        "type": "Disease",
                        "text": "lung cancer",
                        "locations": [
                            {"offset": 145, "length": 11}
                        ],
                        "infons": {
                            "MESH": "D008175"
                        }
                    }
                ],
                "metadata": {
                    "publication_date": "2024-01-15",
                    "journal": "Nature Cancer",
                    "authors": ["Smith, John", "Doe, Alice"],
                    "pmid": "12345678",
                    "doi": "10.1038/s41587-024-01234-5",
                    "language": "en"
                }
            }
        }
    }
    
    @field_validator("doc_id")
    @classmethod
    def validate_doc_id_not_empty(cls, v: str) -> str:
        """Validate that doc_id is not empty after stripping."""
        if not v or not v.strip():
            raise ValueError("doc_id cannot be empty")
        return v.strip()
    
    @field_validator("text")
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        """Validate that text is not empty after stripping."""
        if not v or not v.strip():
            raise ValueError("text cannot be empty")
        return v.strip()
    
    @field_validator("source")
    @classmethod
    def validate_source_not_empty(cls, v: str) -> str:
        """Validate that source is not empty after stripping."""
        if not v or not v.strip():
            raise ValueError("source cannot be empty")
        return v.strip()

