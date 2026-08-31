"""
ETAPA 3: Estrutura do LangGraph com 3 agentes.

Componentes:
- src/agents/triagem.py      - Agente 1: classifica urgência
- src/agents/sintese.py      - Agente 2: hipóteses + condutas
- src/agents/validacao.py    - Agente 3: aplica guardrails
- src/graph/state.py         - Estado compartilhado
- src/graph/nodes.py         - 6 nós do grafo
- src/graph/workflow.py      - StateGraph completo

Autor: Michelle Nogueira (Tech Challenge FIAP - Fase 3)
"""

# ============================================================
# src/graph/state.py
# ============================================================
"""
Estado compartilhado do LangGraph.
TypedDict define o schema do estado que flui entre os nós.
"""
from typing import TypedDict, Optional, Literal


class ConversationState(TypedDict):
    """
    Estado que flui entre os 6 nós do LangGraph.

    Campos:
    - relato_inicial: input do médico (queixa + sintomas)
    - triagem: resultado do Agente 1
    - rag_pmc_chunks: contexto da literatura (RAG #1)
    - rag_interno_chunks: contexto da base interna (RAG #2)
    - sintese: resultado do Agente 2 (hipóteses + condutas)
    - validacao: resultado do Agente 3 (com guardrails)
    - medico_decisao: 'aprovado', 'editado', 'rejeitado'
    - documento_final: PDF gerado
    - erros: lista de erros encontrados
    """
    # Input do médico
    relato_inicial: str
    dados_paciente: Optional[dict]

    # Resultados dos agentes
    triagem: Optional[dict]
    sintese: Optional[dict]
    validacao: Optional[dict]

    # RAG contexts
    rag_pmc_chunks: list[dict]
    rag_interno_chunks: list[dict]

    # Decisão humana
    medico_decisao: Optional[Literal["aprovado", "editado", "rejeitado"]]
    texto_editado: Optional[str]

    # Documento final
    documento_final: Optional[str]
    hash_documento: Optional[str]

    # Auditoria
    session_id: str
    timestamp_inicio: str
    erros: list[str]


# ============================================================
# src/agents/triagem.py
# ============================================================
"""
Agente 1: TRIAGEM
Detecta emergências e classifica urgência.
Temperature BAIXA (0.3) — precisa ser determinístico.
"""
TRIAGEM_SYSTEM_PROMPT = """Você é um assistente de TRIAGEM CLÍNICA de um hospital.

TAREFA: Classificar a urgência de um relato clínico.

CATEGORIAS (escolha EXATAMENTE uma):
- EMERGENCIA: risco iminente de vida (sepse, IAM, AVC, anafilaxia, parada cardíaca)
- URGENTE: necessita atenção em horas (dor intensa, sangramento moderado, febre alta persistente)
- ROTINA: pode aguardar consulta agendada (sintomas leves, crônicos estáveis)

RESPONDA EM JSON ESTRITO:
{
  "categoria": "EMERGENCIA|URGENTE|ROTINA",
  "justificativa": "<máx 200 caracteres explicando>",
  "red_flags": ["<sinal de alerta 1>", "<sinal de alerta 2>"],
  "confianca": "alta|media|baixa"
}

⚠️ IMPORTANTE: Em caso de dúvida entre URGENTE e EMERGENCIA, escolha EMERGENCIA
(é mais seguro superestimar urgência do que subestimar).

⚠️ NÃO prescreva nada. NÃO dê diagnóstico. Apenas CLASSIFIQUE a urgência."""


def triar(relato: str, llm_client) -> dict:
    """Chama LLM para classificar urgência."""
    import json, re

    messages = [
        {"role": "system", "content": TRIAGEM_SYSTEM_PROMPT},
        {"role": "user", "content": f"Relato: {relato}"},
    ]

    response = llm_client.invoke(messages)
    text = response if isinstance(response, str) else response.content

    # Parse JSON (try/except com fallback)
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback seguro: assume URGENTE
    return {
        "categoria": "URGENTE",
        "justificativa": "Falha no parsing — assumindo URGENTE por segurança",
        "red_flags": [],
        "confianca": "baixa",
    }


# ============================================================
# src/agents/sintese.py
# ============================================================
"""
Agente 2: SÍNTESE
Cruza relato + RAG PMC + RAG interno → hipóteses + condutas.
Temperature MÉDIA (0.7) — precisa ser criativo mas responsável.
"""
SINTESE_SYSTEM_PROMPT = """Você é um assistente de SÍNTESE CLÍNICA de um hospital.

TAREFA: Com base no relato do paciente + literatura médica + protocolos internos,
sugerir:
1. HIPÓTESES DIAGNÓSTICAS (ordenadas por probabilidade, com CID-10 quando possível)
2. EXAMES COMPLEMENTARES sugeridos
3. MEDICAÇÕES POTENCIAIS (sempre com aviso de validação)

REGRAS IMPORTANTES:
- SEMPRE cite a FONTE de cada afirmação: [Fonte: PMC-XXXX] ou [Fonte: SOP-XXX]
- Cada hipótese DEVE ter nível de confiança (alta, média, baixa)
- Medicações são SUGESTÕES — NUNCA prescrições definitivas
- Use APENAS o contexto fornecido (NÃO invente conhecimento externo)
- Se não houver evidência, diga "informação insuficiente"

RESPONDA EM JSON ESTRITO:
{
  "hipoteses": [
    {
      "cid10": "X.XX",
      "nome": "<nome da condição>",
      "probabilidade": "alta|media|baixa",
      "justificativa": "<máx 150 chars>",
      "fonte": "PMC-XXXX ou SOP-XXX"
    }
  ],
  "exames_sugeridos": [
    {"nome": "<exame>", "justificativa": "<por quê>", "fonte": "PMC-XXXX"}
  ],
  "medicacoes_sugeridas": [
    {
      "nome": "<medicamento>",
      "dose": "<dose>",
      "frequencia": "<frequência>",
      "NOTA": "VALIDAÇÃO MÉDICA OBRIGATÓRIA"
    }
  ],
  "observacoes": "<máx 300 chars>"
}"""


def sintetizar(relato: str, rag_pmc: list[dict], rag_interno: list[dict], llm_client) -> dict:
    """Chama LLM para gerar hipóteses e condutas."""
    import json, re

    # Formatar contexto RAG
    contexto_pmc = "\n".join(
        f"[Fonte: {c.get('source', 'PMC')}] {c.get('content', '')[:300]}"
        for c in rag_pmc[:3]
    ) or "(sem resultados relevantes)"

    contexto_interno = "\n".join(
        f"[Fonte: {c.get('source', 'SOP')}] {c.get('content', '')[:300]}"
        for c in rag_interno[:3]
    ) or "(sem protocolos específicos)"

    messages = [
        {"role": "system", "content": SINTESE_SYSTEM_PROMPT},
        {"role": "user", "content": f"""
RELATO: {relato}

=== LITERATURA (PMC) ===
{contexto_pmc}

=== PROTOCOLOS INTERNOS ===
{contexto_interno}

Resposta JSON:"""},
    ]

    response = llm_client.invoke(messages)
    text = response if isinstance(response, str) else response.content

    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except (json.JSONDecodeError, AttributeError):
        pass

    return {
        "hipoteses": [],
        "exames_sugeridos": [],
        "medicacoes_sugeridas": [],
        "observacoes": "Falha no parsing da síntese.",
        "_erro": True,
    }


# ============================================================
# src/agents/validacao.py
# ============================================================
"""
Agente 3: VALIDAÇÃO
Aplica guardrails + formata saída final com disclaimer e citações.
Temperature BAIXA (0.2) — precisa ser conservador.
"""
VALIDACAO_SYSTEM_PROMPT = """Você é o agente de VALIDAÇÃO FINAL do assistente médico.

TAREFA: Receber a síntese clínica e:
1. Adicionar DISCLAIMER OBRIGATÓRIO no início
2. Verificar que cada afirmação clínica TEM FONTE CITADA
3. Se houver prescrição direta, ADICIONAR aviso de validação humana
4. Formatar em markdown limpo

⚕️ ATENÇÃO: Esta resposta foi gerada por IA e constitui APENAS
uma sugestão. A validação do médico assistente é obrigatória
antes de qualquer conduta clínica.

REGRAS:
- Preservar estrutura JSON se possível
- Garantir que TODA medicação tenha "VALIDAÇÃO OBRIGATÓRIA"
- Se falta fonte em alguma afirmação, adicionar "[Fonte: informação insuficiente]"
- Tom: profissional mas conservador (não superestimar confiança)"""


def validar(sintese: dict, triagem: dict, llm_client) -> dict:
    """Aplica guardrails e formata saída final."""
    import json, re

    # Garantir campos obrigatórios
    validated = dict(sintese)

    # Adicionar disclaimer se faltar
    if "disclaimer" not in validated:
        validated["disclaimer"] = (
            "⚕️ ATENÇÃO: Sugestão gerada por IA. "
            "Validação humana obrigatória antes de qualquer conduta."
        )

    # Marcar todas medicações com aviso
    for med in validated.get("medicacoes_sugeridas", []):
        if "NOTA" not in med or "VALIDAÇÃO" not in med.get("NOTA", ""):
            med["NOTA"] = "⚠️ VALIDAÇÃO MÉDICA OBRIGATÓRIA"

    # Adicionar info da triagem
    validated["triagem"] = triagem

    return validated


# ============================================================
# src/graph/nodes.py
# ============================================================
"""
Os 6 nós do grafo LangGraph.
"""
from datetime import datetime
import uuid

# Guardrails keywords
EMERGENCY_KEYWORDS = [
    "dor torácica", "infarto", "avc", "derrame", "dispneia",
    "sepse", "choque", "parada cardíaca", "anafilaxia",
    "sangramento ativo", "hemorragia",
]


def detect_emergency(relato: str) -> bool:
    """Detecta se há emergência no relato."""
    return any(kw in relato.lower() for kw in EMERGENCY_KEYWORDS)


def node_triagem(state, llm_client):
    """Nó 1: Triagem."""
    from src.agents.triagem import triar

    relato = state["relato_inicial"]
    triagem_result = triar(relato, llm_client)

    return {
        **state,
        "triagem": triagem_result,
        "rag_pmc_chunks": [],
        "rag_interno_chunks": [],
    }


def node_retrieval(state, retriever):
    """Nó 2: Retrieval RAG (literatura + base interna)."""
    query = state["relato_inicial"]

    # RAG #1: Literatura (PMC)
    rag_pmc = retriever.retrieve_pmc(query, k=4)

    # RAG #2: Base interna (ANVISA + Synthetic)
    rag_interno = retriever.retrieve_interno(query, k=4)

    return {
        **state,
        "rag_pmc_chunks": rag_pmc,
        "rag_interno_chunks": rag_interno,
    }


def node_sintese(state, llm_client):
    """Nó 3: Síntese clínica."""
    from src.agents.sintese import sintetizar

    sintese_result = sintetizar(
        state["relato_inicial"],
        state["rag_pmc_chunks"],
        state["rag_interno_chunks"],
        llm_client,
    )

    return {**state, "sintese": sintese_result}


def node_validacao(state, llm_client):
    """Nó 4: Validação + guardrails."""
    from src.agents.validacao import validar

    validated = validar(state["sintese"], state["triagem"], llm_client)
    return {**state, "validacao": validated}


def node_hitl(state):
    """Nó 5: HITL — PAUSA aguardando médico."""
    # Este nó SEMPRE pausa o grafo.
    # O médico recebe a resposta via UI (Gradio) e decide:
    # - Aprovar como está → gerar_docs
    # - Editar texto → voltar para sintese com edição
    # - Rejeitar → END

    # Em produção, isso seria implementado com interrupt():
    # from langgraph.checkpoint import interrupt
    # decision = interrupt({"validation": state["validacao"]})

    # Por ora, retorna o state e a transição é feita externamente
    return state


def node_gerar_docs(state, doc_generator):
    """Nó 6: Geração de documentos (PDF)."""
    if state.get("medico_decisao") == "rejeitado":
        return state

    # Usar texto_validado (editado ou original)
    texto_final = state.get("texto_editado") or state.get("validacao")

    doc_path = doc_generator.gerar_prontuario(
        texto_final,
        dados_paciente=state.get("dados_paciente", {}),
    )

    import hashlib
    with open(doc_path, "rb") as f:
        doc_hash = hashlib.sha256(f.read()).hexdigest()

    return {
        **state,
        "documento_final": doc_path,
        "hash_documento": doc_hash,
    }


# ============================================================
# src/graph/workflow.py
# ============================================================
"""
Monta o StateGraph completo com 6 nós e transições.
"""
from langgraph.graph import StateGraph, END
from src.graph.state import ConversationState
from src.graph.nodes import (
    node_triagem, node_retrieval, node_sintese,
    node_validacao, node_hitl, node_gerar_docs,
)


def criar_workflow(llm_client, retriever, doc_generator):
    """Monta e retorna o StateGraph compilado."""

    workflow = StateGraph(ConversationState)

    # Adicionar nós
    workflow.add_node("triagem", lambda s: node_triagem(s, llm_client))
    workflow.add_node("retrieval", lambda s: node_retrieval(s, retriever))
    workflow.add_node("sintese", lambda s: node_sintese(s, llm_client))
    workflow.add_node("validacao", lambda s: node_validacao(s, llm_client))
    workflow.add_node("hitl", node_hitl)
    workflow.add_node("gerar_docs", lambda s: node_gerar_docs(s, doc_generator))

    # Definir transições
    workflow.set_entry_point("triagem")
    workflow.add_edge("triagem", "retrieval")
    workflow.add_edge("retrieval", "sintese")
    workflow.add_edge("sintese", "validacao")
    workflow.add_edge("validacao", "hitl")

    # Após HITL: decisão condicional
    def pos_hitl(state):
        decisao = state.get("medico_decisao")
        if decisao == "aprovado" or decisao == "editado":
            return "gerar_docs"
        return END

    workflow.add_conditional_edges("hitl", pos_hitl, {
        "gerar_docs": "gerar_docs",
        END: END,
    })
    workflow.add_edge("gerar_docs", END)

    return workflow.compile()


# ============================================================
# Exemplo de uso
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("🤖 ESTRUTURA LANGGRAPH — 3 AGENTES")
    print("="*60)
    print("""
Estrutura criada em src/:

src/
├── graph/
│   ├── state.py          ConversationState (TypedDict)
│   ├── nodes.py          6 nós do grafo
│   └── workflow.py       StateGraph + transições
└── agents/
    ├── triagem.py        Agente 1: classifica urgência
    ├── sintese.py        Agente 2: hipóteses + condutas
    └── validacao.py      Agente 3: guardrails

FLUXO:
  triagem → retrieval → sintese → validacao → hitl → gerar_docs

AGENTES:
  - Triagem:    temp 0.3 (determinístico)
  - Síntese:    temp 0.7 (criativo mas responsável)
  - Validação:  temp 0.2 (conservador)

HITL: Médico sempre ratifica antes de gerar documentos.
""")