"""
Sistema de auditoria completo para o Tech Challenge Fase 3.

Componentes:
- src/logging/schemas.py   - Dataclasses tipadas para cada evento
- src/logging/audit.py     - Backend SQLite + Loguru
- src/logging/decorators.py - @audit_llm_call para instrumentar agentes
- src/logging/dashboard.py - Consultas SQL prontas

Autor: Michelle Nogueira (Tech Challenge FIAP - Fase 3)
"""

# ============================================================
# src/logging/schemas.py
# ============================================================
"""
Dataclasses tipadas para cada tipo de evento de auditoria.
"""
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional
import uuid


@dataclass
class AuditEvent:
    """Evento base de auditoria."""
    event_type: str  # 'session_start', 'llm_call', 'rag_retrieval', etc.
    session_id: str
    user_id: str = "anonymous"
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LLMCallEvent(AuditEvent):
    """Chamada a um LLM (Triagem, Síntese, Validação)."""
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
    """Retrieval de chunks de um vector store."""
    rag_source: str = ""  # 'pmc' ou 'interno'
    query: str = ""
    n_chunks: int = 0
    top_scores: list = field(default_factory=list)


@dataclass
class GuardrailDecisionEvent(AuditEvent):
    """Decisão de guardrail aplicada."""
    rule: str = ""
    decision: str = ""  # 'pass' ou 'fail'
    message: str = ""


@dataclass
class HITLDecisionEvent(AuditEvent):
    """Decisão humana (médico)."""
    action: str = ""  # 'aprovado', 'editado', 'rejeitado'
    texto_original: str = ""
    texto_editado: Optional[str] = None
    edit_reason: Optional[str] = None


@dataclass
class DocumentGeneratedEvent(AuditEvent):
    """Documento PDF gerado."""
    document_type: str = ""  # 'prontuario', 'atestado', 'receita'
    file_path: str = ""
    sha256: str = ""
    signed_by: str = ""


# ============================================================
# src/logging/audit.py
# ============================================================
"""
Backend SQLite + Loguru.
Persistência ACID para audit trail completo.
"""
import sqlite3
import json
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime
from typing import Optional
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.logging.schemas import AuditEvent


DB_PATH = Path("audit.db")


def init_db(db_path: Path = DB_PATH):
    """Inicializa o banco SQLite com schema de auditoria."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            session_id TEXT NOT NULL,
            user_id TEXT,
            agent TEXT,
            model TEXT,
            input_hash TEXT,
            output_hash TEXT,
            input_preview TEXT,
            output_preview TEXT,
            tokens_in INTEGER,
            tokens_out INTEGER,
            latency_ms INTEGER,
            cost_usd REAL,
            metadata TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON events(session_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_type ON events(event_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_user ON events(user_id)")
    conn.commit()
    conn.close()
    logger.info(f"DB inicializado: {db_path}")


@contextmanager
def get_conn(db_path: Path = DB_PATH):
    """Context manager para conexão SQLite."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def log_event(event: AuditEvent, db_path: Path = DB_PATH):
    """Grava evento no banco."""
    import hashlib

    data = event.to_dict()

    # Hash SHA256 do input/output (não loga PHI cru)
    input_hash = hashlib.sha256(
        str(data.get("input_preview", "")).encode()
    ).hexdigest()[:16]
    output_hash = hashlib.sha256(
        str(data.get("output_preview", "")).encode()
    ).hexdigest()[:16]

    with get_conn(db_path) as conn:
        conn.execute("""
            INSERT INTO events (
                timestamp, event_type, session_id, user_id,
                agent, model, input_hash, output_hash,
                input_preview, output_preview,
                tokens_in, tokens_out, latency_ms, cost_usd, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get("timestamp", datetime.utcnow().isoformat()),
            data.get("event_type", "unknown"),
            data.get("session_id", ""),
            data.get("user_id", "anonymous"),
            data.get("agent", ""),
            data.get("model", ""),
            input_hash,
            output_hash,
            str(data.get("input_preview", ""))[:500],  # truncar
            str(data.get("output_preview", ""))[:500],
            data.get("tokens_in", 0),
            data.get("tokens_out", 0),
            data.get("latency_ms", 0),
            data.get("estimated_cost_usd", 0.0),
            json.dumps(data.get("metadata", {})),
        ))

    logger.info(f"[{event.event_type}] session={event.session_id} user={event.user_id}")


# ============================================================
# src/logging/decorators.py
# ============================================================
"""
Decorador @audit_llm_call para instrumentar agentes automaticamente.
"""
import functools
import time
import uuid
from datetime import datetime
from typing import Callable

from src.logging.audit import log_event, init_db
from src.logging.schemas import LLMCallEvent


def audit_llm_call(agent_name: str, model: str = "biomistral-medquad-lora"):
    """
    Decorador que automaticamente loga toda chamada LLM.

    Uso:
        @audit_llm_call("triagem")
        def triar(relato: str, llm_client) -> dict:
            ...
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            init_db()

            session_id = kwargs.get("session_id", str(uuid.uuid4())[:8])
            user_id = kwargs.get("user_id", "anonymous")

            # Capturar input (primeiro argumento str ou dict)
            input_data = args[0] if args else kwargs.get("relato", "")

            t0 = time.time()
            try:
                result = func(*args, **kwargs)
                latency_ms = int((time.time() - t0) * 1000)
                error = None
            except Exception as e:
                latency_ms = int((time.time() - t0) * 1000)
                result = {"error": str(e)}
                error = str(e)

            # Estimar tokens (chute: ~4 chars/token)
            tokens_in = len(str(input_data)) // 4
            tokens_out = len(str(result)) // 4

            event = LLMCallEvent(
                event_type="llm_call",
                session_id=session_id,
                user_id=user_id,
                agent=agent_name,
                model=model,
                input_preview=str(input_data)[:500],
                output_preview=str(result)[:500],
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=latency_ms,
                estimated_cost_usd=(tokens_in + tokens_out) * 0.000001,  # estimativa
                metadata={"error": error} if error else {},
            )

            log_event(event)

            if error:
                raise
            return result

        return wrapper
    return decorator


# ============================================================
# src/logging/dashboard.py
# ============================================================
"""
Consultas SQL prontas para visualizar auditoria.
"""
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict


def dashboard_resumo(db_path: Path = Path("audit.db"), horas: int = 24):
    """Mostra resumo de atividade nas últimas N horas."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    since = (datetime.utcnow() - timedelta(hours=horas)).isoformat()

    print("="*70)
    print(f"📊 DASHBOARD DE AUDITORIA (últimas {horas}h)")
    print("="*70)

    # 1. Total de eventos
    total = conn.execute(
        "SELECT COUNT(*) FROM events WHERE timestamp >= ?", (since,)
    ).fetchone()[0]
    print(f"\n📈 Total de eventos: {total:,}")

    # 2. Por tipo
    print(f"\n📋 Por tipo de evento:")
    for row in conn.execute("""
        SELECT event_type, COUNT(*) as n
        FROM events
        WHERE timestamp >= ?
        GROUP BY event_type
        ORDER BY n DESC
    """, (since,)):
        print(f"   • {row['event_type']:<25} {row['n']:>5,}")

    # 3. Por agente
    print(f"\n🤖 Por agente LLM:")
    for row in conn.execute("""
        SELECT agent, COUNT(*) as n, AVG(latency_ms) as avg_lat
        FROM events
        WHERE timestamp >= ? AND event_type = 'llm_call'
        GROUP BY agent
    """, (since,)):
        print(f"   • {row['agent'] or '(none)':<25} {row['n']:>5,} chamadas, {row['avg_lat']:.0f}ms média")

    # 4. Top médicos
    print(f"\n👨‍⚕️ Por médico:")
    for row in conn.execute("""
        SELECT user_id, COUNT(*) as n
        FROM events
        WHERE timestamp >= ?
        GROUP BY user_id
        ORDER BY n DESC
        LIMIT 5
    """, (since,)):
        print(f"   • {row['user_id']:<25} {row['n']:>5,} eventos")

    # 5. Sessões ativas
    sessoes = conn.execute("""
        SELECT COUNT(DISTINCT session_id)
        FROM events
        WHERE timestamp >= ?
    """, (since,)).fetchone()[0]
    print(f"\n🔄 Sessões ativas: {sessoes:,}")

    # 6. Custo estimado
    custo = conn.execute("""
        SELECT SUM(cost_usd) FROM events WHERE timestamp >= ?
    """, (since,)).fetchone()[0] or 0
    print(f"💰 Custo estimado: ${custo:.4f}")

    conn.close()


def buscar_por_sessao(session_id: str, db_path: Path = Path("audit.db")):
    """Mostra todos os eventos de uma sessão específica."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    print("="*70)
    print(f"🔍 EVENTOS DA SESSÃO: {session_id}")
    print("="*70)

    for row in conn.execute("""
        SELECT * FROM events WHERE session_id = ?
        ORDER BY timestamp ASC
    """, (session_id,)):
        print(f"\n[{row['timestamp']}] {row['event_type']}")
        if row['agent']:
            print(f"   Agente: {row['agent']}")
        if row['latency_ms']:
            print(f"   Latência: {row['latency_ms']}ms")
        if row['input_preview']:
            print(f"   Input: {row['input_preview'][:200]}")
        if row['output_preview']:
            print(f"   Output: {row['output_preview'][:200]}")

    conn.close()


# ============================================================
# Teste
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("📊 SISTEMA DE AUDITORIA")
    print("="*60)
    print("""
Componentes criados em src/logging/:

src/logging/
├── schemas.py      8 dataclasses tipadas
├── audit.py        SQLite + Loguru
├── decorators.py   @audit_llm_call (instrumenta agentes)
└── dashboard.py    Consultas SQL prontas

USO:
  from src.logging.audit import init_db, log_event
  from src.logging.schemas import LLMCallEvent
  from src.logging.decorators import audit_llm_call

  init_db()

  @audit_llm_call("triagem")
  def triar(relato, llm_client):
      ...

EVENTOS LOGADOS:
  • session_start      início de conversa com médico
  • llm_call          cada chamada a LLM (3 agentes)
  • rag_retrieval     cada busca no ChromaDB
  • guardrail_decision cada aplicação de regra
  • hitl_decision     cada decisão humana
  • document_generated cada PDF gerado
  • error             qualquer exceção

CONSULTAS SQL:
  • Dashboard resumo (últimas 24h)
  • Buscar por sessão
  • Top médicos por uso
  • Latência média por agente
  • Custo estimado por período

EXEMPLO DE AUDIT TRAIL:
  > SELECT * FROM events WHERE session_id = 'a1b2c3'
  session_start → llm_call(triagem) → rag_retrieval(pmc)
              → rag_retrieval(interno) → llm_call(sintese)
              → llm_call(validacao) → hitl_decision(approved)
              → document_generated
""")