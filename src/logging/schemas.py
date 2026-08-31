"""Schemas de eventos de auditoria (dataclasses tipadas)."""
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional


@dataclass
class AuditEvent:
    """Evento base de auditoria."""
    event_type: str
    session_id: str
    user_id: str = "anonymous"
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LLMCallEvent(AuditEvent):
    """Chamada a um LLM."""
    agent: str = ""
    model: str = ""
    input_preview: str = ""
    output_preview: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    estimated_cost_usd: float = 0.0


@dataclass
class RAGRetrievalEvent(AuditEvent):
    """Retrieval de chunks de vector store."""
    rag_source: str = ""
    query: str = ""
    n_chunks: int = 0
    top_scores: list = field(default_factory=list)


@dataclass
class GuardrailDecisionEvent(AuditEvent):
    """Decisão de guardrail aplicada."""
    rule: str = ""
    decision: str = ""
    message: str = ""


@dataclass
class HITLDecisionEvent(AuditEvent):
    """Decisão humana."""
    action: str = ""
    texto_original: str = ""
    texto_editado: Optional[str] = None
    edit_reason: Optional[str] = None


@dataclass
class DocumentGeneratedEvent(AuditEvent):
    """Documento PDF gerado."""
    document_type: str = ""
    file_path: str = ""
    sha256: str = ""
    signed_by: str = ""