"""Estado compartilhado do LangGraph."""
from typing import TypedDict, Optional, Literal


class ConversationState(TypedDict):
    """Estado que flui entre os 6 nós do LangGraph."""
    relato_inicial: str
    dados_paciente: Optional[dict]
    triagem: Optional[dict]
    sintese: Optional[dict]
    validacao: Optional[dict]
    rag_pmc_chunks: list[dict]
    rag_interno_chunks: list[dict]
    medico_decisao: Optional[Literal["aprovado", "editado", "rejeitado"]]
    texto_editado: Optional[str]
    documento_final: Optional[str]
    hash_documento: Optional[str]
    session_id: str
    timestamp_inicio: str
    erros: list[str]