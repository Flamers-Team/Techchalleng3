"""Monta o StateGraph completo com 6 nós."""
from langgraph.graph import StateGraph, END
from src.graph.state import ConversationState
from src.graph.nodes import (
    node_triagem, node_retrieval, node_sintese,
    node_validacao, node_hitl, node_gerar_docs,
)


def criar_workflow(llm_client, retriever, doc_generator):
    """Monta e retorna o StateGraph compilado."""
    workflow = StateGraph(ConversationState)

    workflow.add_node("triagem", lambda s: node_triagem(s, llm_client))
    workflow.add_node("retrieval", lambda s: node_retrieval(s, retriever))
    workflow.add_node("sintese", lambda s: node_sintese(s, llm_client))
    workflow.add_node("validacao", lambda s: node_validacao(s, llm_client))
    workflow.add_node("hitl", node_hitl)
    workflow.add_node("gerar_docs", lambda s: node_gerar_docs(s, doc_generator))

    workflow.set_entry_point("triagem")
    workflow.add_edge("triagem", "retrieval")
    workflow.add_edge("retrieval", "sintese")
    workflow.add_edge("sintese", "validacao")
    workflow.add_edge("validacao", "hitl")

    def pos_hitl(state):
        decisao = state.get("medico_decisao")
        if decisao in ("aprovado", "editado"):
            return "gerar_docs"
        return END

    workflow.add_conditional_edges("hitl", pos_hitl, {
        "gerar_docs": "gerar_docs",
        END: END,
    })
    workflow.add_edge("gerar_docs", END)

    return workflow.compile()