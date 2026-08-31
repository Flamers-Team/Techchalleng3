"""
Agente 1: TRIAGEM — classifica urgência de relatos clínicos.
Usa LLM real (BioMistral fine-tuned) se disponível, senão mock.
"""

import json
import re
from src.llm.client import get_llm


TRIAGEM_SYSTEM_PROMPT = """Você é um assistente de TRIAGEM CLÍNICA de um hospital.

TAREFA: Classificar a urgência de um relato clínico.

CATEGORIAS (escolha EXATAMENTE uma):
- EMERGENCIA: risco iminente de vida (sepse, IAM, AVC, anafilaxia, parada cardíaca)
- URGENTE: necessita atenção em horas (dor intensa, sangramento moderado, febre alta persistente)
- ROTINA: pode aguardar consulta agendada (sintomas leves, crônicos estáveis)

RESPONDA EM JSON ESTRITO:
{
  "categoria": "EMERGENCIA|URGENTE|ROTINA",
  "justificativa": "<máx 200 caracteres>",
  "red_flags": ["<sinal de alerta 1>", "<sinal de alerta 2>"],
  "confianca": "alta|media|baixa"
}

⚠️ Em caso de dúvida entre URGENTE e EMERGENCIA, escolha EMERGENCIA.
⚠️ NÃO prescreva nada. NÃO dê diagnóstico. Apenas CLASSIFIQUE a urgência."""


def triar(relato: str) -> dict:
    """Classifica urgência usando LLM real ou fallback."""
    llm = get_llm()

    messages = [
        {"role": "system", "content": TRIAGEM_SYSTEM_PROMPT},
        {"role": "user", "content": f"Relato: {relato}"},
    ]

    text = llm.invoke(messages)

    # Parse JSON (try/except com fallback)
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            result = json.loads(match.group(0))
            # Validar campos
            if "categoria" in result:
                return result
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback seguro
    return {
        "categoria": "URGENTE",
        "justificativa": "Falha no parsing — assumindo URGENTE por segurança",
        "red_flags": [],
        "confianca": "baixa",
    }