"""
LLM client: carrega o modelo BioMistral fine-tuned (se existir) ou usa fallback.

Autor: Michelle Nogueira (Tech Challenge FIAP - Fase 3)
"""

from pathlib import Path
from typing import Optional
import os


# Paths padrão
DEFAULT_LORA_PATH = Path("biomistral-medquad-lora")
DEFAULT_BASE_MODEL = "BioMistral/BioMistral-7B"


class LLMClient:
    """Wrapper que carrega BioMistral-7B + LoRA adapters (se existirem)."""

    def __init__(
        self,
        base_model: str = DEFAULT_BASE_MODEL,
        lora_path: Path = DEFAULT_LORA_PATH,
        device: str = "auto",
        load_in_4bit: bool = True,
    ):
        self.base_model = base_model
        self.lora_path = Path(lora_path)
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None
        self.use_mock = True  # até conseguir carregar o modelo

        # Tentar carregar modelo real
        if self.lora_path.exists():
            try:
                self._carregar_modelo_real()
                self.use_mock = False
            except Exception as e:
                print(f"⚠️  Não foi possível carregar modelo: {e}")
                print("   Usando modo MOCK (respostas pré-definidas)")
        else:
            print(f"ℹ️  LoRA adapters não encontrados em {self.lora_path}")
            print(f"   Usando modo MOCK até você rodar o fine-tuning")

    def _carregar_modelo_real(self):
        """Carrega BioMistral + LoRA do disco."""
        try:
            from unsloth import FastLanguageModel
            import torch
        except ImportError:
            raise ImportError(
                "Instale: pip install unsloth transformers peft bitsandbytes"
            )

        print(f"🔄 Carregando {self.base_model} + LoRA de {self.lora_path}...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(self.lora_path),  # LoRA dir
            max_seq_length=4096,
            dtype=None,
            load_in_4bit=self.load_in_4bit,
        )
        FastLanguageModel.for_inference(self.model)
        print("✅ Modelo carregado!")

    def invoke(self, messages: list) -> str:
        """Chama o LLM com uma lista de mensagens (formato OpenAI).

        messages = [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
        ]
        """
        if self.use_mock:
            # Fallback: retorna resposta mock inteligente baseada em keywords
            return self._mock_response(messages)

        # Inferência real
        try:
            # Formatar prompt
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )
            return response.strip()
        except Exception as e:
            print(f"Erro na inferência: {e}")
            return self._mock_response(messages)

    def _mock_response(self, messages: list) -> str:
        """Resposta mock inteligente baseada em keywords (pra demo sem GPU)."""
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        user_lower = user_msg.lower()

        # Detectar triagem
        if any(kw in user_lower for kw in ["classifica", "urgência", "triagem", "emergência"]):
            if any(kw in user_lower for kw in ["dor torácica", "infarto", "avc", "dispneia"]):
                return '{"categoria": "EMERGENCIA", "justificativa": "Sinais de emergência detectados", "red_flags": ["dor torácica"], "confianca": "alta"}'
            return '{"categoria": "ROTINA", "justificativa": "Sem sinais críticos", "red_flags": [], "confianca": "media"}'

        # Síntese (default)
        return '''{
  "hipoteses": [
    {"cid10": "I21", "nome": "Infarto Agudo do Miocárdio", "probabilidade": "alta", "justificativa": "Padrão clínico compatível", "fonte": "PMC-12345"}
  ],
  "exames_sugeridos": [
    {"nome": "ECG 12 derivações", "justificativa": "Investigar isquemia", "fonte": "PMC-12345"}
  ],
  "medicacoes_sugeridas": [
    {"nome": "AAS", "dose": "200mg", "frequencia": "dose única", "NOTA": "VALIDAÇÃO MÉDICA OBRIGATÓRIA"}
  ],
  "observacoes": "Caso requer investigação adicional antes de conduta definitiva."
}'''


# ============================================================
# SINGLETON GLOBAL (pra evitar carregar modelo várias vezes)
# ============================================================
_llm_instance: Optional[LLMClient] = None


def get_llm() -> LLMClient:
    """Retorna instância singleton do LLM."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMClient()
    return _llm_instance


# ============================================================
# TESTE
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("🤖 TESTANDO LLM CLIENT")
    print("="*60)

    llm = get_llm()

    # Testar triagem
    resp = llm.invoke([
        {"role": "system", "content": "Você é um assistente de triagem."},
        {"role": "user", "content": "Paciente com dor torácica há 3h. Classifique a urgência."},
    ])
    print(f"\n--- TRIAGEM ---")
    print(resp)

    # Testar síntese
    resp = llm.invoke([
        {"role": "system", "content": "Você é um assistente de síntese."},
        {"role": "user", "content": "Paciente com dor torácica, irradiação para braço esquerdo, sudorese."},
    ])
    print(f"\n--- SÍNTESE ---")
    print(resp)