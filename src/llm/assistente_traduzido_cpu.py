"""
Versão CPU-only do Assistente Traduzido.
Para rodar localmente sem GPU.

Uso:
  python assistente_traduzido_cpu.py
"""
import torch
from transformers import MarianMTModel, MarianTokenizer, AutoModelForCausalLM, AutoTokenizer
import time


class Tradutor:
    """Wrapper para tradução bidirecional PT-BR <-> EN (CPU-friendly)."""

    def __init__(self, device: str = "cpu"):
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
        if direcao == "pt_en":
            tokenizer, model = self.tokenizer_pt_en, self.model_pt_en
        elif direcao == "en_pt":
            tokenizer, model = self.tokenizer_en_pt, self.model_en_pt
        else:
            raise ValueError(f"Direção inválida: {direcao}")

        inputs = tokenizer(
            [texto],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=512,
                num_beams=2,  # Reduzido para CPU (mais rápido)
                early_stopping=True,
                do_sample=False,
            )

        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


class AssistenteCPU:
    """Assistente médico otimizado para CPU."""

    def __init__(
        self,
        modelo_path: str = "biomistral-medquad-lora",
        max_new_tokens: int = 256,
    ):
        # Tradutor
        self.tradutor = Tradutor(device="cpu")

        # Modelo base BioMistral em 4-bit (CPU não suporta, então usa FP16 se possível)
        print(f"📥 Carregando modelo: {modelo_path}")
        print("⚠️  CPU é lento. Considere GPU ou API HuggingFace.")

        self.tokenizer = AutoTokenizer.from_pretrained(modelo_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            modelo_path,
            torch_dtype=torch.float32,  # CPU = float32
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        self.max_new_tokens = max_new_tokens
        self.prompt_template = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n"
            "### Input:\n{input}\n\n"
            "### Response:\n"
        )

    def perguntar(self, pergunta_pt: str, verbose: bool = False) -> str:
        t0 = time.time()

        pergunta_en = self.tradutor.traduzir(pergunta_pt, "pt_en")
        if verbose:
            print(f"  EN: {pergunta_en}")

        prompt = self.prompt_template.format(instruction=pergunta_en, input="")

        inputs = self.tokenizer([prompt], return_tensors="pt").to("cpu")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.5,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.3,
            )

        resposta_en = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        resposta_en = resposta_en.split("### Response:")[-1].strip()

        resposta_pt = self.tradutor.traduzir(resposta_en, "en_pt")

        if verbose:
            print(f"  PT: {resposta_pt[:200]}...")
            print(f"  ⏱️ {time.time()-t0:.1f}s")

        return resposta_pt


def main():
    print("🖥️ Modo CPU")
    bot = AssistenteCPU()

    testes = [
        "O que é diabetes?",
        "Quais os sintomas de pneumonia?",
    ]

    for p in testes:
        print(f"\n❓ {p}")
        r = bot.perguntar(p, verbose=True)
        print(f"💊 {r[:300]}")


if __name__ == "__main__":
    main()
