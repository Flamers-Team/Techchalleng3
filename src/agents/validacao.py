"""Agente 3: VALIDAÇÃO — aplica guardrails e formata saída final."""


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
- Preservar estrutura JSON
- Garantir que TODA medicação tenha "VALIDAÇÃO OBRIGATÓRIA"
- Se falta fonte em alguma afirmação, adicionar "[Fonte: informação insuficiente]"
- Tom: profissional mas conservador"""


def validar(sintese: dict, triagem: dict, llm_client) -> dict:
    """Aplica guardrails e formata saída final."""
    validated = dict(sintese)

    # Adicionar disclaimer
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