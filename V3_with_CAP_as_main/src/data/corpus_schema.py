# LexAgent v3.0 | corpus_schema.py
"""Data models for LexAgent v3.0 legal corpus with CAP integration.

Defines the canonical document and citation record schemas used throughout
the pipeline: retrieval, debate agents, CCE, and Adaptive Memory.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any


@dataclass
class CorpusDocument:
    """A single document in the LexAgent legal corpus.
    
    Attributes:
        doc_id: Unique identifier (e.g., "CAP_410_0001_01" or "SCOTUS_hash").
        text: Full document or chunk text.
        source: Origin dataset — "CAP" | "SCOTUS" | "CaseHOLD" | "LegalBench".
        case_name: Extracted case name if available.
        year: Decision year (0 if unknown).
        court: Court name (e.g., "Supreme Court of the United States").
        volume: Reporter volume (e.g., "410").
        reporter: Reporter abbreviation (e.g., "U.S.").
        first_page: Starting page number in the official reporter.
        metadata: Additional attributes.
    """
    doc_id: str
    text: str
    source: str = "CAP"
    case_name: str = ""
    year: int = 0
    court: str = ""
    volume: str = ""
    reporter: str = ""
    first_page: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CorpusDocument':
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass
class CitationRecord:
    """A citation extracted from a debate agent's response.
    
    Attributes:
        case_name: Cited case name (e.g., "Roe v. Wade" or "Smith v. Jones").
        court: Court where the case was decided.
        year: Year of the decision.
        holding: The claimed holding or legal rule.
        self_confidence: LLM's stated confidence in this citation (0.0-1.0).
        agent_source: Which debate agent produced this citation ("prosecutor" | "defense").
        reporter_cite: Official volume/reporter string if cited (e.g., "410 U.S. 113").
    """
    case_name: str
    court: str
    year: int
    holding: str
    self_confidence: float = 1.0
    agent_source: str = "prosecutor"
    reporter_cite: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CitationRecord':
        """Deserialize from dictionary."""
        return cls(**data)
