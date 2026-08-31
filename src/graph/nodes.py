"""6 nós do grafo LangGraph."""
import hashlib


EMERGENCY_KEYWORDS = [
    "dor torácica", "infarto", "avc", "derrame", "dispneia",
    "sepse", "choque", "parada cardíaca", "anafilaxia",
    "sangramento ativo", "hemorragia",
]


def detect_emergency(relato: str) -> bool:
    return any(kw in relato.lower() for kw in EMERGENCY_KEYWORDS)


def node_triagem(state, llm_client):
    from src.agents.triagem import triar
    triagem_result = triar(state["relato_inicial"], llm_client)
    return {
        **state,
        "triagem": triagem_result,
        "rag_pmc_chunks": [],
        "rag_interno_chunks": [],
    }


def node_retrieval(state, retriever):
    query = state["relato_inicial"]
    rag_pmc = retriever.retrieve_pmc(query, k=4)
    rag_interno = retriever.retrieve_interno(query, k=4)
    return {
        **state,
        "rag_pmc_chunks": rag_pmc,
        "rag_interno_chunks": rag_interno,
    }


def node_sintese(state, llm_client):
    from src.agents.sintese import sintetizar
    sintese_result = sintetizar(
        state["relato_inicial"],
        state["rag_pmc_chunks"],
        state["rag_interno_chunks"],
        llm_client,
    )
    return {**state, "sintese": sintese_result}


def node_validacao(state, llm_client):
    from src.agents.validacao import validar
    validated = validar(state["sintese"], state["triagem"], llm_client)
    return {**state, "validacao": validated}


def node_hitl(state):
    """Pausa aguardando decisão humana."""
    return state


def node_gerar_docs(state, doc_generator):
    if state.get("medico_decisao") == "rejeitado":
        return state
    texto_final = state.get("texto_editado") or state.get("validacao")
    doc_path = doc_generator.gerar_prontuario(
        texto_final,
        dados_paciente=state.get("dados_paciente", {}),
    )
    with open(doc_path, "rb") as f:
        doc_hash = hashlib.sha256(f.read()).hexdigest()
    return {
        **state,
        "documento_final": doc_path,
        "hash_documento": doc_hash,
    }