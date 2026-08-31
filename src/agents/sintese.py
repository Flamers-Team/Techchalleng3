"""Agente 2: SÍNTESE — gera hipóteses diagnósticas + condutas."""
import json, re


SINTESE_SYSTEM_PROMPT = """Você é um assistente de SÍNTESE CLÍNICA de um hospital.

TAREFA: Com base no relato do paciente + literatura médica + protocolos internos,
sugerir:
1. HIPÓTESES DIAGNÓSTICAS (com CID-10 quando possível)
2. EXAMES COMPLEMENTARES
3. MEDICAÇÕES POTENCIAIS (sempre com aviso de validação)

REGRAS:
- SEMPRE cite a FONTE: [Fonte: PMC-XXXX] ou [Fonte: SOP-XXX]
- Cada hipótese DEVE ter nível de confiança (alta, média, baixa)
- Medicações são SUGESTÕES — NUNCA prescrições definitivas
- Use APENAS o contexto fornecido

RESPONDA EM JSON ESTRITO:
{
  "hipoteses": [{"cid10": "X.XX", "nome": "...", "probabilidade": "alta|media|baixa", "justificativa": "...", "fonte": "..."}],
  "exames_sugeridos": [{"nome": "...", "justificativa": "...", "fonte": "..."}],
  "medicacoes_sugeridas": [{"nome": "...", "dose": "...", "frequencia": "...", "NOTA": "VALIDAÇÃO MÉDICA OBRIGATÓRIA"}],
  "observacoes": "..."
}"""


def sintetizar(relato: str, rag_pmc: list, rag_interno: list, llm_client) -> dict:
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
        "observacoes": "Falha no parsing.",
        "_erro": True,
    }