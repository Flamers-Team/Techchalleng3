"""
Assistente Médico com Fine-Tuning + Tradução PT-BR <-> EN
=========================================================

Pipeline:
  Pergunta PT-BR
      ↓
  [MarianMT PT → EN]
      ↓
  [BioMistral Fine-Tuned] (GPU A100 ou CPU com quantização)
      ↓
  [MarianMT EN → PT]
      ↓
  Resposta PT-BR (terminologia médica brasileira)

Modelos:
  - LLM: biomistral-medquad-lora (seu fine-tuning) ou BioMistral/BioMistral-7B base
  - Tradutor EN→PT: Helsinki-NLP/opus-mt-tc-big-en-pt
  - Tradutor PT→EN: Helsinki-NLP/opus-mt-tc-big-pt-en

Uso:
  python assistente_traduzido.py
  ou
  from assistente_traduzido import AssistenteTraduzido
  bot = AssistenteTraduzido()
  print(bot.perguntar("O que é diabetes?"))
"""

import torch
from transformers import MarianMTModel, MarianTokenizer
from unsloth import FastLanguageModel
import time


class Tradutor:
    """Wrapper para tradução bidirecional PT-BR <-> EN."""

    def __init__(self, device: str = "cuda", cache_dir: str = None):
        self.device = device

        print("📥 Carregando tradutor PT → EN...")
        self.tokenizer_pt_en = MarianTokenizer.from_pretrained(
            "Helsinki-NLP/opus-mt-tc-big-pt-en"
        )
        self.model_pt_en = MarianMTModel.from_pretrained(
            "Helsinki-NLP/opus-mt-tc-big-pt-en"
        ).to(device)
        self.model_pt_en.eval()

        print("📥 Carregando tradutor EN → PT...")
        self.tokenizer_en_pt = MarianTokenizer.from_pretrained(
            "Helsinki-NLP/opus-mt-tc-big-en-pt"
        )
        self.model_en_pt = MarianMTModel.from_pretrained(
            "Helsinki-NLP/opus-mt-tc-big-en-pt"
        ).to(device)
        self.model_en_pt.eval()

        print("✅ Tradutores carregados")

    def traduzir(self, texto: str, direcao: str = "pt_en") -> str:
        """
        Traduz texto entre PT-BR e EN.

        Args:
            texto: Texto a traduzir
            direcao: "pt_en" ou "en_pt"

        Returns:
            Texto traduzido
        """
        if direcao == "pt_en":
            tokenizer, model = self.tokenizer_pt_en, self.model_pt_en
        elif direcao == "en_pt":
            tokenizer, model = self.tokenizer_en_pt, self.model_en_pt
        else:
            raise ValueError(f"Direção inválida: {direcao}")

        # Tokeniza
        inputs = tokenizer(
            [texto],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        # Gera tradução
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=512,
                num_beams=4,
                early_stopping=True,
                do_sample=False,
            )

        # Decodifica
        traducao = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return traducao.strip()

    def pt_para_en(self, texto_pt: str) -> str:
        """Traduz PT-BR → EN."""
        return self.traduzir(texto_pt, "pt_en")

    def en_para_pt(self, texto_en: str) -> str:
        """Traduz EN → PT-BR."""
        return self.traduzir(texto_en, "en_pt")


class AssistenteTraduzido:
    """Assistente médico com tradução automática PT-BR <-> EN."""

    def __init__(
        self,
        modelo_path: str = "biomistral-medquad-lora",
        device: str = "cuda",
        max_new_tokens: int = 512,
        temperature: float = 0.5,
        top_p: float = 0.9,
    ):
        self.device = device

        # Carrega tradutor
        self.tradutor = Tradutor(device=device)

        # Carrega LLM fine-tuned
        print(f"📥 Carregando modelo: {modelo_path}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=modelo_path,
            max_seq_length=4096,
            dtype=None,
            load_in_4bit=(device == "cuda"),  # 4-bit se GPU, FP32 se CPU
        )
        FastLanguageModel.for_inference(self.model)
        print("✅ Modelo carregado")

        # Parâmetros de geração
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        # Template Alpaca (do seu fine-tuning)
        self.prompt_template = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n"
            "### Input:\n{input}\n\n"
            "### Response:\n"
        )

    def perguntar(
        self,
        pergunta_pt: str,
        topico_en: str = "",
        verbose: bool = False,
    ) -> str:
        """
        Faz pergunta em PT-BR, retorna resposta em PT-BR.

        Args:
            pergunta_pt: Pergunta em português brasileiro
            topico_en: Tópico em inglês (opcional, para RAG)
            verbose: Se True, mostra os passos intermediários

        Returns:
            Resposta em português brasileiro
        """
        t0 = time.time()

        # 1. Traduz pergunta PT → EN
        if verbose:
            print(f"  🇧🇷 Pergunta original: {pergunta_pt}")
        pergunta_en = self.tradutor.pt_para_en(pergunta_pt)
        if verbose:
            print(f"  🇺🇸 Tradução EN: {pergunta_en}")

        # 2. Gera resposta em inglês
        if topico_en:
            contexto = f"Context / Topic: {topico_en}"
        else:
            contexto = ""

        prompt = self.prompt_template.format(
            instruction=pergunta_en,
            input=contexto,
        )

        inputs = self.tokenizer(
            [prompt],
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                repetition_penalty=1.3,
            )

        resposta_en = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        resposta_en = resposta_en.split("### Response:")[-1].strip()

        if verbose:
            print(f"  🤖 LLM EN: {resposta_en[:200]}...")

        # 3. Traduz resposta EN → PT
        resposta_pt = self.tradutor.en_para_pt(resposta_en)
        if verbose:
            print(f"  🇧🇷 Tradução PT: {resposta_pt[:200]}...")

        t_total = time.time() - t0
        if verbose:
            print(f"  ⏱️ Tempo total: {t_total:.1f}s")

        return resposta_pt

    def conversar(self):
        """Modo conversacional interativo."""
        print("=" * 70)
        print("🏥 ASSISTENTE MÉDICO COM TRADUÇÃO AUTOMÁTICA")
        print("=" * 70)
        print("Faça perguntas em português brasileiro.")
        print("Digite 'sair' ou 'exit' para encerrar.")
        print("=" * 70)

        while True:
            try:
                pergunta = input("\n❓ Pergunta: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Até logo!")
                break

            if not pergunta:
                continue
            if pergunta.lower() in ("sair", "exit", "quit"):
                print("👋 Até logo!")
                break

            resposta = self.perguntar(pergunta, verbose=True)
            print(f"\n💊 Resposta:\n{resposta}")
            print("─" * 70)


def main():
    """Demonstração rápida."""
    import sys

    # Detecta device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Device: {device}")

    # Inicializa assistente
    bot = AssistenteTraduzido(
        modelo_path="biomistral-medquad-lora",
        device=device,
    )

    # Testes rápidos
    perguntas_teste = [
        "O que é diabetes?",
        "Quais são os sintomas de pneumonia?",
        "Como funciona a vacina de mRNA?",
        "O que causa AVC?",
        "Quais são os tratamentos para hipertensão?",
    ]

    print("\n" + "=" * 70)
    print("🧪 TESTES RÁPIDOS")
    print("=" * 70)

    for pergunta in perguntas_teste:
        print(f"\n❓ {pergunta}")
        resposta = bot.perguntar(pergunta, verbose=False)
        print(f"💊 {resposta[:300]}...")
        print("─" * 70)

    # Inicia modo conversacional
    print("\n💬 Entrando em modo conversacional...")
    bot.conversar()


if __name__ == "__main__":
    main()
